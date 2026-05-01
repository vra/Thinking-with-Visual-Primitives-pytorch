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
from peft import LoraConfig, get_peft_model, PeftModel
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


def diagnose_nan_in_model(model, model_inputs, logger):
    """
    When loss is NaN, run a diagnostic forward pass with hooks
    to locate which layer produces the first NaN/Inf.
    """
    logger.error("=" * 60)
    logger.error("NAN/INF DIAGNOSTIC — Locating source of numerical instability")
    logger.error("=" * 60)

    # 1. Print input tensor statistics
    logger.error("[Inputs]")
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            nan_count = v.isnan().sum().item()
            inf_count = v.isinf().sum().item()
            if torch.is_floating_point(v) or torch.is_complex(v):
                stats = (
                    f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
                    f"mean={v.mean().item():.4f}, "
                )
            else:
                stats = "dtype=integer/bool (skipped mean), "
            logger.error(
                f"  {k}: shape={list(v.shape)}, dtype={v.dtype}, device={v.device}, "
                f"{stats}nan={nan_count}, inf={inf_count}"
            )

    # 2. Register forward hooks on all modules
    first_nan_layer = [None]
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if first_nan_layer[0] is not None:
                return
            # Check output
            tensors_to_check = []
            if isinstance(output, torch.Tensor):
                tensors_to_check.append(("output", output))
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        tensors_to_check.append((f"output[{i}]", o))
            elif isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        tensors_to_check.append((f"output['{k}']", v))
            for tensor_name, tensor in tensors_to_check:
                if tensor.isnan().any() or tensor.isinf().any():
                    first_nan_layer[0] = (name, module.__class__.__name__, tensor_name)
                    nan_count = tensor.isnan().sum().item()
                    inf_count = tensor.isinf().sum().item()
                    logger.error(
                        f"[FIRST NAN] Layer: {name} ({module.__class__.__name__}) | {tensor_name} | "
                        f"shape={list(tensor.shape)}, nan={nan_count}, inf={inf_count}, "
                        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
                    )
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 3. Re-run forward with no_grad (diagnostic only)
    with torch.no_grad():
        try:
            _ = model(**model_inputs)
        except Exception as e:
            logger.error(f"Diagnostic forward failed: {e}")

    # 4. Cleanup hooks
    for h in hooks:
        h.remove()

    if first_nan_layer[0] is None:
        logger.error("[Result] No NaN/Inf found in any layer's output during diagnostic forward.")
        logger.error("         NaN may originate from: loss computation, gradient, or backward pass.")
    else:
        logger.error(f"[Result] First NaN/Inf detected at: {first_nan_layer[0][0]} ({first_nan_layer[0][1]})")

    logger.error("=" * 60)
    raise RuntimeError(f"Training stopped for NaN diagnosis. See logs above. First NaN layer: {first_nan_layer[0]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="visual-primitives")
    parser.add_argument("--wandb_run_name", type=str, default="pretrain")
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, skip_lora: bool = False):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["model_name_or_path"],
        use_spatial_compression=cfg.get("use_spatial_compression", False),
        freeze_vision_tower=cfg.get("freeze_vision_tower", True),
        freeze_llm=cfg.get("freeze_llm", False),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
    )

    if cfg.get("use_lora", False) and not skip_lora:
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


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger, grad_accum=1, log_every=50, wandb_run=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    nan_streak = 0          # consecutive NaN losses
    batch_has_nan = False   # whether current batch triggered NaN

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # Move all tensor items to device and pass to model
        model_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**model_inputs)
        loss = outputs.loss

        # ---- NaN loss handling: skip batch, but diagnose on first occurrence ----
        if not torch.isfinite(loss):
            nan_streak += 1
            batch_has_nan = True
            logger.error(
                f"Epoch {epoch} Step {step+1}: loss is nan/inf (streak={nan_streak})"
            )
            if nan_streak == 1 or nan_streak % 10 == 0:
                # Run diagnosis only occasionally to avoid flooding logs
                try:
                    diagnose_nan_in_model(model, model_inputs, logger)
                except Exception as diag_err:
                    logger.error(f"NaN diagnosis failed: {diag_err}")
            if nan_streak >= 50:
                raise RuntimeError(
                    f"Training diverged: {nan_streak} consecutive NaN losses. Stopping."
                )
            # Skip backward/step for this batch
            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                optimizer.zero_grad()
            continue
        else:
            nan_streak = 0
            batch_has_nan = False

        loss = loss / grad_accum
        # Compute backward in fp32 to avoid bf16 numerical instability
        loss.float().backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            # Check for nan gradients before stepping
            has_nan_grad = False
            bad_param_name = None
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    bad_param_name = name
                    break

            if has_nan_grad:
                logger.error(
                    f"Epoch {epoch} Step {step+1}: NaN/Inf gradient in '{bad_param_name}' — "
                    "skipping step and zeroing gradients"
                )
                optimizer.zero_grad()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if grad_norm > max_grad_norm * 0.9:
                    logger.warning(
                        f"Epoch {epoch} Step {step+1}: grad_norm={grad_norm:.2f} is close to clip threshold"
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        num_batches += 1
        pbar.set_postfix({"loss": loss.item() * grad_accum, "lr": scheduler.get_last_lr()[0]})

        # Log intermediate loss
        if (step + 1) % log_every == 0:
            current_loss = total_loss / max(num_batches, 1)
            lr = scheduler.get_last_lr()[0]
            msg = f"Epoch {epoch} Step {step+1}/{len(dataloader)} | loss={loss.item() * grad_accum:.4f} | avg_loss={current_loss:.4f} | lr={lr:.2e}"
            logger.info(msg)
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss.item() * grad_accum,
                    "train/avg_loss": current_loss,
                    "train/lr": lr,
                    "train/step": step + 1,
                    "train/epoch": epoch,
                })

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
    if wandb_run is not None:
        wandb_run.log({"train/epoch_avg_loss": avg_loss, "train/epoch": epoch})
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

    # Optional: wandb
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=cfg,
            )
            logger.info("Wandb initialized.")
        except Exception as e:
            logger.warning(f"Wandb init failed: {e}")

    # Build model
    model = build_model(cfg)
    # 4-bit/8-bit models use device_map="auto", already on GPU. Skip .to() to avoid OOM.
    if not (cfg.get("load_in_4bit", False) or cfg.get("load_in_8bit", False)):
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
        resume_path = Path(args.resume)
        # Check if resume path is a PEFT adapter directory (adapter-only save)
        if resume_path.is_dir() and (resume_path / "adapter_config.json").exists():
            import gc
            from peft import PeftModel
            # model.vlm is already wrapped by get_peft_model in build_model.
            # We need to replace it with a PeftModel loaded from the saved adapter.
            # First, unwrap to get the base model and release the old adapter's GPU memory.
            base_model = model.vlm.model if isinstance(model.vlm, PeftModel) else model.vlm
            model.vlm = base_model  # drop the old PeftModel wrapper
            del base_model
            gc.collect()
            torch.cuda.empty_cache()

            model.vlm = PeftModel.from_pretrained(model.vlm, str(resume_path), is_trainable=True)
            # Try to infer epoch from directory name (e.g., epoch_0 -> start at epoch 1)
            if resume_path.name.startswith("epoch_"):
                start_epoch = int(resume_path.name.split("_")[-1]) + 1
            logger.info(f"Resumed PEFT adapter from {resume_path}, starting at epoch {start_epoch}")
        else:
            start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
            start_epoch += 1

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

    log_every = cfg.get("log_every", 50)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg.get("max_grad_norm", 0.5)
    for epoch in range(start_epoch, cfg["epochs"]):
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger, grad_accum, log_every, wandb_run, max_grad_norm)
        global_step += len(dataloader) // grad_accum

        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(save_dir))
            logger.info(f"Model saved to {save_dir}")

    # Save final
    model.save_pretrained(str(output_dir / "final"))
    logger.info("Pretraining complete.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
