"""
Generate RFT (Rejection Fine-Tuning) data using expert models.

1. Load expert models ETwG and ETwP.
2. For each sample in the data pool, generate N rollouts.
3. Compute rewards using reward models.
4. Categorize into Easy / Normal / Hard.
5. Retain Normal-Level + 5% Easy-Level samples.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from data.datasets_sft import JSONLSFTDataset, MixedSFTDataset
from data.collators import ConversationCollator
from data.transforms import ImageTransform
from rl.reward_models import build_reward_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_box", type=str, default="outputs/rl_box/final")
    parser.add_argument("--expert_point", type=str, default="outputs/rl_point/final")
    parser.add_argument("--data_pool", type=str, default="data/sft")
    parser.add_argument("--output", type=str, default="data/rft/rft_data.jsonl")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_expert_model(path: str, device: str):
    model = VisualPrimitiveVLM.from_pretrained(path, device_map=device)
    model.eval()
    return model


def generate_rollouts(model, dataloader, num_rollouts: int, max_new_tokens: int):
    """Generate rollouts for all samples in dataloader."""
    all_results = []
    for batch in tqdm(dataloader, desc="Generating rollouts"):
        input_ids = batch["input_ids"].to(model.vlm.device)
        attention_mask = batch["attention_mask"].to(model.vlm.device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(model.vlm.device)

        batch_size = input_ids.shape[0]
        for _ in range(num_rollouts):
            texts = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )
            # Split back to per-sample
            for i in range(batch_size):
                all_results.append({
                    "input_ids": input_ids[i].cpu(),
                    "attention_mask": attention_mask[i].cpu(),
                    "pixel_values": pixel_values[i].cpu() if pixel_values is not None else None,
                    "text": texts[i],
                })
    return all_results


def categorize_difficulty(rewards: List[float], num_rollouts: int) -> str:
    """
    Categorize based on number of correct rollouts.
    Easy: all correct; Normal: 1 <= k < N; Hard: all incorrect.
    """
    # Use a threshold to determine "correct"
    threshold = 0.7
    correct_count = sum(1 for r in rewards if r >= threshold)
    if correct_count == num_rollouts:
        return "easy"
    elif correct_count == 0:
        return "hard"
    else:
        return "normal"


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load experts
    expert_box = load_expert_model(args.expert_box, args.device)
    expert_point = load_expert_model(args.expert_point, args.device)

    # Build data pool (all SFT data)
    ds_tasks = []
    for task in ["counting", "spatial", "maze", "path"]:
        jsonl = Path(args.data_pool) / task / f"{task}_data.jsonl"
        img_root = Path(args.data_pool) / task
        if jsonl.exists():
            ds_tasks.append(JSONLSFTDataset(jsonl, img_root, task_type=task))

    if not ds_tasks:
        print("No data pool found.")
        return

    dataset = MixedSFTDataset(ds_tasks, weights=[1.0] * len(ds_tasks), total_samples=5000)
    collator = ConversationCollator(
        tokenizer=expert_box.tokenizer,
        image_transform=ImageTransform(448),
        max_length=2048,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    reward_fns = build_reward_models("mixed")

    # Generate with both experts and collect results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open(output_path, "w")

    sample_idx = 0
    for batch in tqdm(dataloader, desc="RFT generation"):
        metadata_list = batch.get("metadata", [{}] * args.batch_size)

        for expert_name, expert in [("box", expert_box), ("point", expert_point)]:
            texts = expert.generate(
                pixel_values=batch.get("pixel_values"),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

            for i, text in enumerate(texts):
                meta = metadata_list[i] if i < len(metadata_list) else {}
                # Compute reward
                rewards = [fn(text, meta) for fn in reward_fns]
                avg_reward = sum(rewards) / len(rewards)

                rec = {
                    "sample_idx": sample_idx,
                    "expert": expert_name,
                    "text": text,
                    "rewards": rewards,
                    "avg_reward": avg_reward,
                    "metadata": meta,
                }
                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_idx += 1

    writer.close()
    print(f"RFT data saved to {output_path}")


if __name__ == "__main__":
    main()
