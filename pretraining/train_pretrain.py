"""
Pretraining script for visual primitive grounding.

This script trains the model to output bounding boxes and points
using next-token prediction on large-scale detection/grounding data.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_pretrain import JSONLGroundingDataset, COCODetectionDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from utils.logging import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["model_name_or_path"],
        use_spatial_compression=cfg.get("use_spatial_compression", False),
        freeze_vision_tower=cfg.get("freeze_vision_tower", True),
        freeze_llm=cfg.get("freeze_llm", False),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
    )

    if cfg.get("use_lora", False):
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_cfg)
        model.vlm.print_trainable_parameters()

    return model


def build_dataloader(cfg: dict, tokenizer):
    ds_type = cfg["dataset_type"]
    if ds_type == "jsonl":
        dataset = JSONLGroundingDataset(
            jsonl_path=cfg["data_path"],
            image_root=cfg.get("image_root"),
        )
    elif ds_type == "coco":
        dataset = COCODetectionDataset(
            image_root=cfg["image_root"],
            annotation_path=cfg["annotation_path"],
        )
    else:
        raise ValueError(f"Unknown dataset_type: {ds_type}")

    transform = ImageTransform(image_size=cfg.get("image_size", 448))
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=transform,
        max_length=cfg.get("max_length", 2048),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )
    return loader


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger, grad_accum=1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pixel_values = None
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        num_batches += 1
        pbar.set_postfix({"loss": loss.item() * grad_accum, "lr": scheduler.get_last_lr()[0]})

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("pretrain", log_file=str(output_dir / "train.log"))
    logger.info(f"Config: {cfg}")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    model.to(device)
    tokenizer = model.tokenizer

    # Build dataloader
    dataloader = build_dataloader(cfg, tokenizer)

    # Optimizer & scheduler
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(dataloader) // grad_accum) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    for epoch in range(start_epoch, cfg["epochs"]):
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger, grad_accum)
        global_step += len(dataloader) // grad_accum

        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(save_dir))
            logger.info(f"Model saved to {save_dir}")

    # Save final
    model.save_pretrained(str(output_dir / "final"))
    logger.info("Pretraining complete.")


if __name__ == "__main__":
    main()
