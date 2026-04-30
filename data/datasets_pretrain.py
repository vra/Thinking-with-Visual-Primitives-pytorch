"""Datasets for pretraining visual primitive grounding."""

import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


class JSONLGroundingDataset(Dataset):
    """Dataset from JSONL grounding annotations."""

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None):
        self.samples = []
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root else self.jsonl_path.parent

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

        # Build text from boxes/points for pretraining
        boxes = item.get("boxes", [])
        points = item.get("points", [])
        label = item.get("label", "")

        text_parts = []
        if label:
            text_parts.append(f"<|ref|>{label}<|/ref|>")
        if boxes:
            box_str = ",".join(f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in boxes)
            text_parts.append(f"<|box|>[{box_str}]<|/box|>")
        if points:
            pt_str = ",".join(f"[{p[0]},{p[1]}]" for p in points)
            text_parts.append(f"<|point|>[{pt_str}]<|/point|>")

        return {
            "image_path": str(image_path) if image_path.exists() else None,
            "text": " ".join(text_parts),
            "label": label,
            "boxes": boxes,
            "points": points,
            "metadata": item,
        }


class COCODetectionDataset(Dataset):
    """Dataset from COCO detection annotations."""

    def __init__(self, image_root: str, annotation_path: str):
        self.image_root = Path(image_root)
        with open(annotation_path, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

        # Group annotations by image
        self.img_annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_annotations:
                self.img_annotations[img_id] = []
            self.img_annotations[img_id].append(ann)

        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.image_root / img_info["file_name"]
        W, H = img_info.get("width", 1), img_info.get("height", 1)

        anns = self.img_annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            from model.special_tokens import normalize_coordinate
            boxes.append([
                normalize_coordinate(x, W),
                normalize_coordinate(y, H),
                normalize_coordinate(x + w, W),
                normalize_coordinate(y + h, H),
            ])
            labels.append(self.categories.get(ann["category_id"], ""))

        return {
            "image_path": str(img_path) if img_path.exists() else None,
            "boxes": boxes,
            "labels": labels,
            "metadata": {"image_id": img_id},
        }
