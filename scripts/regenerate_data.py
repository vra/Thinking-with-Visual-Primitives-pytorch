#!/usr/bin/env python3
"""Regenerate pretrain and SFT data with normalized [0,999] coordinates."""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.special_tokens import normalize_coordinate
from utils.coco_categories import COCO_CATS

random.seed(42)

def regenerate_pretrain(coco_json_path, output_path):
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    records = []
    for img_info in coco["images"]:
        img_id = img_info["id"]
        W, H = img_info["width"], img_info["height"]
        anns = img_anns.get(img_id, [])
        if not anns:
            continue

        by_cat = defaultdict(list)
        for ann in anns:
            cat_id = ann["category_id"]
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, min(x, W))
            y1 = max(0.0, min(y, H))
            x2 = max(0.0, min(x + w, W))
            y2 = max(0.0, min(y + h, H))
            box = (
                normalize_coordinate(x1, W),
                normalize_coordinate(y1, H),
                normalize_coordinate(x2, W),
                normalize_coordinate(y2, H),
            )
            by_cat[cat_id].append(box)

        for cat_id, boxes in by_cat.items():
            records.append({
                "image": str(Path("images") / img_info["file_name"]),
                "label": str(cat_id),  # keep as numeric string for compatibility
                "boxes": boxes,
                "points": [],
                "normalized": True,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Pretrain data: {len(records)} samples -> {output_path}")
    # Verify
    max_coord = max(max(max(b) for b in rec["boxes"]) for rec in records)
    print(f"  Max coordinate: {max_coord} (should be <= 999)")
    return records


def regenerate_sft(pretrain_records, output_path, neg_ratio=0.30, max_samples=10000):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build samples
    all_samples = []
    for rec in pretrain_records:
        # label is numeric category id (string)
        label_id = int(rec["label"])
        category = COCO_CATS.get(label_id, f"object_{label_id}")

        boxes = [tuple(b) for b in rec["boxes"]]
        all_samples.append({
            "image": rec["image"],
            "category": category,
            "label_id": label_id,
            "boxes": boxes,
        })

    # Group by image for negative sampling
    img_to_labels = defaultdict(set)
    for s in all_samples:
        img_to_labels[s["image"]].add(s["label_id"])

    MAX_BOXES_PER_SAMPLE = 8
    positive_samples = []
    for s in all_samples:
        if not s["boxes"]:
            continue
        boxes = s["boxes"][:MAX_BOXES_PER_SAMPLE]
        positive_samples.append({
            "image": s["image"],
            "question": f"Locate the {s['category']} in the image.",
            "thinking": build_positive_thinking(s["category"], boxes),
            "answer": build_positive_answer(s["category"], boxes),
            "boxes": boxes,
            "points": [],
        })

    all_label_ids = list(COCO_CATS.keys())
    img_list = list(img_to_labels.keys())
    negative_samples = []
    for img_rel in img_list:
        present = img_to_labels[img_rel]
        absent = [lid for lid in all_label_ids if lid not in present]
        if absent:
            n_neg = min(2, len(absent))
            for neg_label in random.sample(absent, n_neg):
                category = COCO_CATS[neg_label]
                negative_samples.append({
                    "image": img_rel,
                    "question": f"Locate the {category} in the image.",
                    "thinking": build_negative_thinking(category),
                    "answer": build_negative_answer(category),
                    "boxes": [],
                    "points": [],
                })

    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    n_pos_target = int(max_samples * (1 - neg_ratio))
    n_neg_target = int(max_samples * neg_ratio)

    pos_selected = positive_samples[:n_pos_target]
    neg_selected = negative_samples[:n_neg_target]

    combined = pos_selected + neg_selected
    random.shuffle(combined)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"SFT data: {len(combined)} samples -> {out_path}")
    print(f"  Positive: {len(pos_selected)}")
    print(f"  Negative: {len(neg_selected)}")


def format_box_token(boxes):
    if not boxes:
        return ""
    inner = ",".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in boxes)
    return f"<|box|>[{inner}]<|/box|>"


def build_positive_thinking(category, boxes):
    box_token = format_box_token(boxes)
    refs = "\n".join(
        f"I see a <|ref|>{category}<|/ref|>{box_token}."
        for _ in (boxes if boxes else [])
    )
    if not refs:
        refs = f"I see a <|ref|>{category}<|/ref|>{box_token}."
    return (
        f"1. **Analyzing the request**\n"
        f"The user asks me to locate the {category} in this image.\n"
        f"2. **Object grounding**\n"
        f"{refs}\n"
        f"3. **Conclusion**\n"
        f"The {category} is located at the specified coordinates."
    )


def build_positive_answer(category, boxes):
    if not boxes:
        return f"The {category} is not visible in the image."
    box_str = ",".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in boxes)
    return f"The {category} is located at [{box_str}]."


def build_negative_thinking(category):
    return (
        f"1. **Analyzing the request**\n"
        f"The user asks me to locate the {category} in this image.\n"
        f"2. **Object grounding**\n"
        f"After carefully scanning the entire image, I do not see any {category} present.\n"
        f"3. **Conclusion**\n"
        f"There is no {category} in this image."
    )


def build_negative_answer(category):
    return f"There is no {category} in the image."


def main():
    coco_json = "data/coco/val/annotations.json"
    pretrain_out = Path("data/pretrain/grounding.jsonl")
    sft_out = Path("data/sft/grounding/sft_grounding.jsonl")

    print("Regenerating pretrain data with [0,999] normalized coordinates...")
    records = regenerate_pretrain(coco_json, pretrain_out)

    print("\nRegenerating SFT data...")
    regenerate_sft(records, sft_out)

    print("\nDone!")


if __name__ == "__main__":
    main()
