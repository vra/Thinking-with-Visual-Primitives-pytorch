"""
Checkpoint saving/loading utilities.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    save_dir: str,
    additional_info: Optional[dict] = None,
):
    """Save a training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if additional_info:
        checkpoint.update(additional_info)

    path = save_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, path)
    # Also save latest
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    print(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    strict: bool = True,
):
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    print(f"Loaded checkpoint from epoch {epoch}, step {step}")
    return epoch, step


def save_lora_checkpoint(model, save_dir: str):
    """Save only LoRA adapters."""
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model.save_pretrained(save_dir)
    else:
        raise ValueError("Model is not a PEFT model")
