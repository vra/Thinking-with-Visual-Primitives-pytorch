#!/usr/bin/env python3
"""
Generate SFT grounding data with negative samples for improved precision and rejection ability.

Produces:
    - data/sft/grounding/sft_grounding.jsonl
    - 70% positive samples with full CoT thinking template
    - 30% negative samples (object not in image -> empty response)

Usage:
    python scripts/generate_sft_grounding_data.py \
        --coco_jsonl data/pretrain/grounding.jsonl \
        --image_root data/coco/val \
        --output data/sft/grounding/sft_grounding.jsonl \
        --neg_ratio 0.30
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.coco_categories import COCO_CATS

random.seed(42)

IRREGULAR_PLURALS = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "knife": "knives",
    "child": "children",
}


def pluralize(word: str) -> str:
    low = word.lower()
    if low in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[low]
    if " " in word:
        parts = word.rsplit(" ", 1)
        return parts[0] + " " + pluralize(parts[1])
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


GROUNDING_TEMPLATES = [
    "Locate the {category} in the image.",
    "Locate the {category} in this image.",
    "Find the {category} in the image.",
    "Where is the {category} in the image?",
    "Where is the {category}?",
    "Show me the {category} in the image.",
    "Point out the {category}.",
    "Locate the {plural} in the image.",
]


def format_box_token(boxes):
    """Format boxes into <|box|>[[x1,y1,x2,y2],...]<|/box|>."""
    if not boxes:
        return ""
    inner = ",".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in boxes)
    return f"<|box|>[{inner}]<|/box|>"


def build_positive_thinking(category, boxes):
    """Build CoT thinking for positive sample."""
    refs = "\n".join(
        f"I see a <|ref|>{category}<|/ref|><|box|>[[{b[0]},{b[1]},{b[2]},{b[3]}]]<|/box|>."
        for b in (boxes if boxes else [])
    )
    if not refs:
        refs = f"I see a <|ref|>{category}<|/ref|><|box|>[]<|/box|>."

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_jsonl", type=str, default="data/pretrain/grounding.jsonl")
    parser.add_argument("--image_root", type=str, default="data/coco/val")
    parser.add_argument("--output", type=str, default="data/sft/grounding/sft_grounding.jsonl")
    parser.add_argument("--neg_ratio", type=float, default=0.30)
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all positive samples
    print("Loading COCO grounding data...")
    all_samples = []
    with open(args.coco_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            img_rel = item.get("image", "")
            label_raw = item.get("label", 0)
            try:
                label_id = int(label_raw)
                category = COCO_CATS.get(label_id, f"object_{label_id}")
            except (ValueError, TypeError):
                # label is already a string (category name)
                category = str(label_raw)
            boxes = [tuple(b) for b in item.get("boxes", [])]
            all_samples.append({
                "image": img_rel,
                "category": category,
                "label_id": label_id,
                "boxes": boxes,
            })

    # Group by image for negative sampling
    img_to_labels = defaultdict(set)
    for s in all_samples:
        img_to_labels[s["image"]].add(s["label_id"])

    # Build positive SFT samples
    # Cap boxes at 8 per sample to keep sequence length manageable for 12G VRAM
    MAX_BOXES_PER_SAMPLE = 8
    positive_samples = []
    for s in all_samples:
        if not s["boxes"]:
            continue
        boxes = s["boxes"][:MAX_BOXES_PER_SAMPLE]
        cat = s["category"]
        plural = pluralize(cat)
        question = random.choice(GROUNDING_TEMPLATES).format(category=cat, plural=plural)
        positive_samples.append({
            "image": s["image"],
            "question": question,
            "thinking": build_positive_thinking(cat, boxes),
            "answer": build_positive_answer(cat, boxes),
            "boxes": boxes,
            "points": [],
        })

    # Build negative SFT samples
    all_label_ids = list(COCO_CATS.keys())
    img_list = list(img_to_labels.keys())
    negative_samples = []
    for img_rel in img_list:
        present = img_to_labels[img_rel]
        absent = [lid for lid in all_label_ids if lid not in present]
        if absent:
            # Sample 1-2 negative categories per image
            n_neg = min(2, len(absent))
            for neg_label in random.sample(absent, n_neg):
                category = COCO_CATS[neg_label]
                plural = pluralize(category)
                question = random.choice(GROUNDING_TEMPLATES).format(category=category, plural=plural)
                negative_samples.append({
                    "image": img_rel,
                    "question": question,
                    "thinking": build_negative_thinking(category),
                    "answer": build_negative_answer(category),
                    "boxes": [],
                    "points": [],
                })

    # Shuffle and sample
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # Determine counts based on neg_ratio
    n_pos_target = int(args.max_samples * (1 - args.neg_ratio))
    n_neg_target = int(args.max_samples * args.neg_ratio)

    pos_selected = positive_samples[:n_pos_target]
    neg_selected = negative_samples[:n_neg_target]

    # Combine and shuffle
    combined = pos_selected + neg_selected
    random.shuffle(combined)

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(combined)} SFT samples:")
    print(f"  Positive: {len(pos_selected)}")
    print(f"  Negative: {len(neg_selected)}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
