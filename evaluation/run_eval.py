"""
Unified evaluation script for all tasks.

Usage:
  python evaluation/run_eval.py \
    --model_path outputs/opd/final \
    --task counting \
    --data_path data/eval/counting_eval.jsonl \
    --output results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_sft import JSONLSFTDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from evaluation import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=["counting", "spatial", "maze", "path", "all"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _get_generate_kwargs(batch, device):
    """Extract kwargs for model.generate from batch, including VL-specific tensors."""
    kwargs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    pixel_values = batch.get("pixel_values")
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device)
    for key in ("image_grid_thw", "image_rotary_emb", "pixel_values_videos", "video_grid_thw"):
        if key in batch:
            kwargs[key] = batch[key].to(device)
    return kwargs


def evaluate_counting(model, dataloader, device) -> Dict:
    em_count = 0
    ra_count = 0
    total = 0
    for batch in tqdm(dataloader, desc="Eval counting"):
        kwargs = _get_generate_kwargs(batch, device)
        texts = model.generate(
            max_new_tokens=256,
            **kwargs,
        )
        metadata_list = batch.get("metadata", [{}] * len(texts))
        for text, meta in zip(texts, metadata_list):
            gt = meta.get("count", 0)
            if metrics.counting_exact_match(text, gt):
                em_count += 1
            if metrics.counting_relative_accuracy(text, gt, threshold=0.1):
                ra_count += 1
            total += 1

    return {
        "exact_match": em_count / max(total, 1),
        "ra@10": ra_count / max(total, 1),
        "total": total,
    }


def evaluate_spatial(model, dataloader, device) -> Dict:
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Eval spatial"):
        kwargs = _get_generate_kwargs(batch, device)
        texts = model.generate(
            max_new_tokens=256,
            **kwargs,
        )
        metadata_list = batch.get("metadata", [{}] * len(texts))
        for text, meta in zip(texts, metadata_list):
            gt = meta.get("answer", "")
            if metrics.vqa_accuracy(text, gt):
                correct += 1
            total += 1

    return {"accuracy": correct / max(total, 1), "total": total}


def evaluate_maze(model, dataloader, device) -> Dict:
    answer_correct = 0
    path_valid = 0
    total = 0
    for batch in tqdm(dataloader, desc="Eval maze"):
        kwargs = _get_generate_kwargs(batch, device)
        texts = model.generate(
            max_new_tokens=512,
            **kwargs,
        )
        metadata_list = batch.get("metadata", [{}] * len(texts))
        for text, meta in zip(texts, metadata_list):
            solvable = meta.get("solvable", True)
            if metrics.maze_answer_correctness(text, solvable):
                answer_correct += 1
            if metrics.maze_path_validity(text):
                path_valid += 1
            total += 1

    return {
        "answer_accuracy": answer_correct / max(total, 1),
        "path_validity": path_valid / max(total, 1),
        "total": total,
    }


def evaluate_path(model, dataloader, device) -> Dict:
    answer_correct = 0
    endpoint_acc = 0
    traj_scores = []
    total = 0
    for batch in tqdm(dataloader, desc="Eval path"):
        kwargs = _get_generate_kwargs(batch, device)
        texts = model.generate(
            max_new_tokens=512,
            **kwargs,
        )
        metadata_list = batch.get("metadata", [{}] * len(texts))
        for text, meta in zip(texts, metadata_list):
            gt_label = meta.get("end_label", "")
            gt_points = meta.get("points", [])
            if metrics.path_answer_correctness(text, gt_label):
                answer_correct += 1
            if gt_points and metrics.path_endpoint_accuracy(text, gt_points[-1]):
                endpoint_acc += 1
            if gt_points:
                traj_scores.append(metrics.path_trajectory_similarity(text, gt_points))
            total += 1

    return {
        "answer_accuracy": answer_correct / max(total, 1),
        "endpoint_accuracy": endpoint_acc / max(total, 1),
        "trajectory_similarity": sum(traj_scores) / max(len(traj_scores), 1),
        "total": total,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = VisualPrimitiveVLM.from_pretrained(args.model_path, device_map=str(device))
    model.eval()
    tokenizer = model.tokenizer

    image_root = args.image_root or Path(args.data_path).parent
    dataset = JSONLSFTDataset(
        jsonl_path=args.data_path,
        image_root=image_root,
        task_type=args.task if args.task != "all" else "spatial",
    )
    collator = ConversationCollator(
        tokenizer=tokenizer,
        image_transform=ImageTransform(448),
        max_length=2048,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    task_fn = {
        "counting": evaluate_counting,
        "spatial": evaluate_spatial,
        "maze": evaluate_maze,
        "path": evaluate_path,
    }

    if args.task == "all":
        results = {}
        for task, fn in task_fn.items():
            results[task] = fn(model, dataloader, device)
    else:
        results = task_fn[args.task](model, dataloader, device)

    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()