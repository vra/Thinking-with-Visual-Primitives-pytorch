"""
On-Policy Distillation (OPD).

Distill expert models ETwG and ETwP into a single unified student model
using reverse KL divergence on the student's own trajectories.

Loss: L_OPD = Σ w_i * D_KL(π_θ || π_Ei)
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_sft import JSONLSFTDataset, MixedSFTDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/opd.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/opd")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["student_model"],
        freeze_vision_tower=cfg.get("freeze_vision_tower", True),
        freeze_llm=cfg.get("freeze_llm", False),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
    )
    return model


def build_teacher_models(cfg, device):
    teachers = []
    for path in cfg["teacher_models"]:
        t = VisualPrimitiveVLM(
            model_name_or_path=path,
            torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
            device_map=str(device),
            load_in_4bit=cfg.get("load_in_4bit", False),
            load_in_8bit=cfg.get("load_in_8bit", False),
        )
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        teachers.append(t)
    return teachers


def build_dataloader(cfg, tokenizer):
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
    dataset = MixedSFTDataset(datasets, weights, total_samples=cfg.get("total_samples", 5000))
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=ImageTransform(cfg.get("image_size", 448)),
        max_length=cfg.get("max_length", 2048),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
    )


def compute_kl_loss(student_logits, teacher_logits, attention_mask):
    """
    Reverse KL: D_KL(π_θ || π_E) = Σ π_θ(x) * (log π_θ(x) - log π_E(x)).
    Per the paper: L_OPD = Σ w_i * D_KL(π_θ || π_Ei).
    We compute per-token KL and average over valid tokens.
    """
    # student_logits, teacher_logits: (B, L, V)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()

    # Reverse KL: D_KL(student || teacher) = Σ p_student * (log p_student - log p_teacher)
    kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    mask = attention_mask.float()
    kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    return kl


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("opd", log_file=str(output_dir / "train.log"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    student = build_model(cfg)
    student.to(device)
    teachers = build_teacher_models(cfg, device)
    teacher_weights = cfg.get("teacher_weights", [1.0 / len(teachers)] * len(teachers))

    tokenizer = student.tokenizer
    dataloader = build_dataloader(cfg, tokenizer)

    total_steps = len(dataloader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(cfg["epochs"]):
        student.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Move all tensor items to device
            model_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Student forward
            student_outputs = student(**model_inputs)
            student_logits = student_outputs.logits

            # Teacher forwards (no grad) — remove labels for inference
            teacher_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
            kl_losses = []
            with torch.no_grad():
                for teacher in teachers:
                    teacher_outputs = teacher(**teacher_inputs)
                    kl = compute_kl_loss(student_logits, teacher_outputs.logits, attention_mask)
                    kl_losses.append(kl)

            # Weighted KL loss
            kl_loss = sum(w * kl for w, kl in zip(teacher_weights, kl_losses))
            # Optionally add the standard CE loss to prevent forgetting
            loss = cfg.get("ce_coeff", 0.1) * student_outputs.loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "kl": kl_loss.item()})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            student.save_pretrained(str(save_dir))

    student.save_pretrained(str(output_dir / "final"))
    logger.info("OPD training complete.")


if __name__ == "__main__":
    main()