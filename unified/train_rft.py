"""
Unified RFT (Rejection Fine-Tuning) training.

Initialize from the base pretrained model and train on RFT data
retained from expert rollouts (Normal-Level + 5% Easy-Level).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from utils.logging import setup_logger


class RFTDataset(Dataset):
    """Dataset from RFT generated JSONL."""

    def __init__(self, jsonl_path: str, min_reward: float = 0.5):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("avg_reward", 0) >= min_reward:
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rft.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/rft")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["base_model"],
        freeze_vision_tower=cfg.get("freeze_vision_tower", True),
        freeze_llm=cfg.get("freeze_llm", False),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
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
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("rft", log_file=str(output_dir / "train.log"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg)
    model.to(device)
    tokenizer = model.tokenizer

    dataset = RFTDataset(cfg["rft_data_path"], min_reward=cfg.get("min_reward", 0.5))
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=ImageTransform(cfg.get("image_size", 448)),
        max_length=cfg.get("max_length", 4096),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
    )

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(dataloader) // grad_accum) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

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
            pbar.set_postfix({"loss": loss.item() * grad_accum})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(save_dir))

    model.save_pretrained(str(output_dir / "final"))
    logger.info("RFT training complete.")


if __name__ == "__main__":
    main()
