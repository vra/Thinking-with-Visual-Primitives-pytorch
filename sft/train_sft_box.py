"""
Specialized SFT for Thinking with Grounding (FTwG).

Trains the model to output bounding boxes as visual primitives during reasoning.
Tasks: Counting, Spatial Reasoning, General VQA.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_sft import JSONLSFTDataset, MixedSFTDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from utils.logging import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_box.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_box")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["model_name_or_path"],
        freeze_vision_tower=cfg.get("freeze_vision_tower", True),
        freeze_llm=cfg.get("freeze_llm", False),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
    )
    if cfg.get("use_lora", False):
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj", "o_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_cfg)
        model.vlm.print_trainable_parameters()
    return model


def build_dataloader(cfg: dict, tokenizer):
    datasets = []
    weights = []

    for ds_cfg in cfg["datasets"]:
        ds = JSONLSFTDataset(
            jsonl_path=ds_cfg["path"],
            image_root=ds_cfg.get("image_root"),
            task_type=ds_cfg["task_type"],
        )
        datasets.append(ds)
        weights.append(ds_cfg.get("weight", 1.0))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = MixedSFTDataset(
            datasets=datasets,
            weights=weights,
            total_samples=cfg.get("total_samples", 10000),
        )

    transform = ImageTransform(image_size=cfg.get("image_size", 448))
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=transform,
        max_length=cfg.get("max_length", 4096),
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
        # Move all tensor items to device and pass to model
        model_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**model_inputs)
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

    logger = setup_logger("sft_box", log_file=str(output_dir / "train.log"))
    logger.info(f"Config: {cfg}")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg)
    model.to(device)
    tokenizer = model.tokenizer

    dataloader = build_dataloader(cfg, tokenizer)

    total_steps = len(dataloader) * cfg["epochs"]
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
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    for epoch in range(start_epoch, cfg["epochs"]):
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger, grad_accum)
        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(save_dir))
            logger.info(f"Model saved to {save_dir}")

    model.save_pretrained(str(output_dir / "final"))
    logger.info("SFT (box) complete.")


if __name__ == "__main__":
    main()
