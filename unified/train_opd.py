"""
On-Policy Distillation (OPD).

Distill expert models ETwG and ETwP into a single unified student model.
Uses forward KL divergence with temperature scaling for stable offline distillation.

Loss: L_OPD = Σ w_i * D_KL(π_Ei || π_θ) + ce_coeff * L_CE
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


TASK_TO_TEACHER = {
    "counting": 0,
    "spatial": 0,
    "grounding": 0,
    "maze": 1,
    "path": 1,
}


def compute_distill_loss(student_logits, teacher_logits, labels, temperature=2.0):
    """
    Forward KL: D_KL(teacher || student) with temperature scaling.
    Only computed on assistant tokens (where labels != -100).
    """
    s_logits = student_logits[:, :-1, :] / temperature
    t_logits = teacher_logits[:, :-1, :] / temperature

    teacher_probs = F.softmax(t_logits, dim=-1)
    student_log_probs = F.log_softmax(s_logits, dim=-1)

    per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)

    labels_shifted = labels[:, 1:]
    mask = (labels_shifted != -100).float()
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

    return loss * (temperature ** 2)


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
    temperature = cfg.get("temperature", 2.0)
    ce_coeff = cfg.get("ce_coeff", 0.3)
    grad_accum = cfg.get("gradient_accumulation_steps", 16)

    tokenizer = student.tokenizer
    dataloader = build_dataloader(cfg, tokenizer)

    total_steps = (len(dataloader) // grad_accum) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    logger.info(f"Config: temperature={temperature}, ce_coeff={ce_coeff}, "
                f"grad_accum={grad_accum}, lr={cfg['learning_rate']}, "
                f"total_steps={total_steps}")

    for epoch in range(cfg["epochs"]):
        student.train()
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            model_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Student forward (with gradients)
            student_outputs = student(**model_inputs)
            student_logits = student_outputs.logits
            ce_loss = student_outputs.loss

            # Determine which teacher to use based on task type
            metadata_list = batch.get("metadata", [{}])
            task_type = metadata_list[0].get("task_type", "") if metadata_list else ""
            teacher_idx = TASK_TO_TEACHER.get(task_type, None)

            teacher_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
            if teacher_idx is not None and teacher_idx < len(teachers):
                # Route to the expert teacher for this task
                with torch.no_grad():
                    teacher_outputs = teachers[teacher_idx](**teacher_inputs)
                    teacher_logits = teacher_outputs.logits.detach()
                kl_loss = compute_distill_loss(
                    student_logits, teacher_logits,
                    model_inputs["labels"], temperature
                )
                del teacher_outputs, teacher_logits
            else:
                # Unknown task: average all teachers
                kl_losses = []
                for teacher in teachers:
                    with torch.no_grad():
                        teacher_outputs = teacher(**teacher_inputs)
                        t_logits = teacher_outputs.logits.detach()
                    kl = compute_distill_loss(
                        student_logits, t_logits,
                        model_inputs["labels"], temperature
                    )
                    kl_losses.append(kl)
                    del teacher_outputs, t_logits
                kl_loss = sum(w * kl for w, kl in zip(teacher_weights, kl_losses))

            loss = ce_coeff * ce_loss + kl_loss
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            batch_loss = loss.item() * grad_accum
            total_loss += batch_loss
            total_kl += kl_loss.item()
            total_ce += ce_loss.item()
            num_batches += 1
            pbar.set_postfix({
                "loss": f"{batch_loss:.3f}",
                "kl": f"{kl_loss.item():.3f}",
                "ce": f"{ce_loss.item():.3f}",
            })

            if (step + 1) % 50 == 0:
                avg_l = total_loss / num_batches
                avg_k = total_kl / num_batches
                avg_c = total_ce / num_batches
                logger.info(
                    f"Epoch {epoch} Step {step+1}/{len(dataloader)} | "
                    f"loss={batch_loss:.4f} | avg_loss={avg_l:.4f} | "
                    f"avg_kl={avg_k:.4f} | avg_ce={avg_c:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_kl = total_kl / max(num_batches, 1)
        avg_ce = total_ce / max(num_batches, 1)
        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f} (kl={avg_kl:.4f}, ce={avg_ce:.4f})")

        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            student.save_pretrained(str(save_dir))

    student.save_pretrained(str(output_dir / "final"))
    logger.info("OPD training complete.")


if __name__ == "__main__":
    main()
