"""
Specialized RL training for Thinking with Grounding (ETwG).

Uses GRPO to optimize the box-specialized SFT model.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_sft import JSONLSFTDataset, MixedSFTDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from rl.grpo_trainer import GRPOTrainer
from rl.reward_models import build_reward_models
from utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rl_point.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/rl_point")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model = VisualPrimitiveVLM(
        model_name_or_path=cfg["model_name_or_path"],
        freeze_vision_tower=True,
        freeze_llm=False,
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
    )
    return model


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

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = MixedSFTDataset(datasets, weights, total_samples=cfg.get("total_samples", 5000))

    transform = ImageTransform(image_size=cfg.get("image_size", 448))
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=transform,
        max_length=cfg.get("max_length", 2048),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("rl_point", log_file=str(output_dir / "train.log"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load policy and reference models
    policy = build_model(cfg)
    policy.to(device)

    ref_model = build_model(cfg)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tokenizer = policy.tokenizer
    dataloader = build_dataloader(cfg, tokenizer)

    reward_fns = build_reward_models(task_type="mixed")

    trainer = GRPOTrainer(
        model=policy.vlm,
        ref_model=ref_model.vlm,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        group_size=cfg.get("group_size", 4),
        kl_coeff=cfg.get("kl_coeff", 0.04),
        lr=cfg.get("learning_rate", 1e-6),
        device=device,
    )

    for epoch in range(cfg["epochs"]):
        stats = trainer.train_epoch(dataloader, max_new_tokens=cfg.get("max_new_tokens", 512))
        logger.info(f"Epoch {epoch}: {stats}")
        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_dir = output_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            policy.save_pretrained(str(save_dir))

    policy.save_pretrained(str(output_dir / "final"))
    logger.info("RL (point) training complete.")


if __name__ == "__main__":
    main()
