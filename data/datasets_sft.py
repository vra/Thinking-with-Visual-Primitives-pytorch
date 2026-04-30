"""Datasets for supervised fine-tuning."""

import json
import random
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset


class JSONLSFTDataset(Dataset):
    """Dataset from JSONL SFT annotations."""

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, task_type: str = "vqa"):
        self.samples = []
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root else self.jsonl_path.parent
        self.task_type = task_type

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_rel = item.get("image", "")
        image_path = self.image_root / image_rel

        # Build metadata from task-specific fields
        metadata = {
            "task_type": self.task_type,
            "image": image_rel,
        }
        if "count" in item:
            metadata["count"] = item["count"]
        if "answer" in item:
            metadata["answer"] = item["answer"]
        if "solvable" in item:
            metadata["solvable"] = item["solvable"]
        if "end_label" in item:
            metadata["end_label"] = item["end_label"]
        if "boxes" in item:
            metadata["boxes"] = item["boxes"]
        if "points" in item:
            metadata["points"] = item["points"]

        return {
            "image_path": str(image_path) if image_path.exists() else None,
            "question": item.get("question", ""),
            "thinking": item.get("thinking", ""),
            "answer": item.get("answer", ""),
            "text": item.get("thinking", "") + "\n" + item.get("answer", ""),
            "metadata": metadata,
        }


class MixedSFTDataset(Dataset):
    """Weighted mixture of multiple SFT datasets."""

    def __init__(self, datasets: List[Dataset], weights: List[float], total_samples: int = 10000):
        self.datasets = datasets
        self.weights = weights
        self.total_samples = total_samples

        # Normalize weights
        total_weight = sum(weights)
        self.probs = [w / total_weight for w in weights]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Sample a dataset according to weights
        ds_idx = random.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]
        dataset = self.datasets[ds_idx]
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]
