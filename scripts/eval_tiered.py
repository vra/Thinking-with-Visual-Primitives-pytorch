#!/usr/bin/env python3
"""
Tiered evaluation of grounding capability on COCO data.
Tests: Easy / Medium / Hard / Negative samples

Usage:
    python scripts/eval_tiered.py \
        --model_path outputs/pretrain/final \
        --coco_jsonl data/pretrain/grounding.jsonl \
        --image_root data/coco/val \
        --output_dir outputs/tiered_eval
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from model.special_tokens import parse_box_token
from utils.coco_categories import COCO_CATS

random.seed(42)


def load_model(model_path: str, device: str = "cuda"):
    print(f"[LOAD] {model_path}")
    model = VisualPrimitiveVLM.from_pretrained(
        model_path, device_map=device, load_in_4bit=False, freeze_vision_tower=True
    )
    model.eval()
    return model


def unload_model(model):
    import gc
    if hasattr(model, 'vlm'):
        model.vlm = model.vlm.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_inference(model, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
    tokenizer = model.tokenizer
    conv = [
        {"role": "system", "content": "You are a helpful assistant that can understand images and reason with visual primitives."},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    from transformers import AutoProcessor
    processor_path = getattr(model, "base_model_path", tokenizer.name_or_path)
    try:
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    except Exception:
        processor = None

    if processor is not None:
        image = Image.open(image_path).convert("RGB")
        # Pre-resize to control visual token count (same as training collator)
        image = image.resize((336, 336), Image.LANCZOS)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    else:
        from data.transforms import ImageTransform
        transform = ImageTransform(image_size=224)
        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0)
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {**encoded, "pixel_values": pixel_values}

    inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output_ids = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0].strip()


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return 0.0
    ious = []
    for gt in gt_boxes:
        best = max(box_iou(pred, gt) for pred in pred_boxes)
        ious.append(best)
    return sum(ious) / len(ious)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/pretrain/final")
    parser.add_argument("--coco_jsonl", type=str, default="data/pretrain/grounding.jsonl")
    parser.add_argument("--image_root", type=str, default="data/coco/val")
    parser.add_argument("--num_per_tier", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/tiered_eval")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all samples
    print("Loading COCO grounding data...")
    all_samples = []
    with open(args.coco_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            img_path = str(Path(args.image_root) / item.get("image", ""))
            label_id = int(item.get("label", 0))
            category = COCO_CATS.get(label_id, f"object_{label_id}")
            boxes = [tuple(b) for b in item.get("boxes", [])]
            all_samples.append({
                "image_path": img_path,
                "label_id": label_id,
                "category": category,
                "gt_boxes": boxes,
            })

    # Group by image for negative sampling
    img_to_labels = defaultdict(set)
    for s in all_samples:
        img_to_labels[s["image_path"]].add(s["label_id"])

    # Categorize by difficulty
    def classify_difficulty(sample):
        boxes = sample["gt_boxes"]
        n = len(boxes)
        total_area = sum(box_area(b) for b in boxes)
        img_area = 999 * 999
        ratio = total_area / img_area

        if n == 1 and ratio > 0.05:
            return "easy"
        elif n <= 3 and ratio > 0.02:
            return "medium"
        else:
            return "hard"

    tiers = {"easy": [], "medium": [], "hard": []}
    for s in all_samples:
        tier = classify_difficulty(s)
        tiers[tier].append(s)

    # Sample from each tier
    test_samples = []
    for tier_name in ["easy", "medium", "hard"]:
        pool = tiers[tier_name]
        if len(pool) > args.num_per_tier:
            sampled = random.sample(pool, args.num_per_tier)
        else:
            sampled = pool
        for s in sampled:
            s["tier"] = tier_name
            test_samples.append(s)

    # Generate negative samples
    all_label_ids = list(COCO_CATS.keys())
    img_list = list(img_to_labels.keys())
    neg_samples = []
    for img_path in random.sample(img_list, min(args.num_per_tier, len(img_list))):
        present = img_to_labels[img_path]
        absent = [lid for lid in all_label_ids if lid not in present]
        if absent:
            neg_label = random.choice(absent)
            neg_samples.append({
                "image_path": img_path,
                "label_id": neg_label,
                "category": COCO_CATS[neg_label],
                "gt_boxes": [],
                "tier": "negative",
            })

    test_samples.extend(neg_samples)
    print(f"Total test samples: {len(test_samples)}")
    print(f"  Easy: {sum(1 for s in test_samples if s['tier']=='easy')}")
    print(f"  Medium: {sum(1 for s in test_samples if s['tier']=='medium')}")
    print(f"  Hard: {sum(1 for s in test_samples if s['tier']=='hard')}")
    print(f"  Negative: {sum(1 for s in test_samples if s['tier']=='negative')}")

    # Run inference
    model = load_model(args.model_path)
    results = []
    for s in tqdm(test_samples, desc="Evaluating"):
        prompt = f"Locate the {s['category']} in the image."
        try:
            resp = run_inference(model, s["image_path"], prompt, args.max_new_tokens)
        except Exception as e:
            print(f"\n[ERROR] {s['image_path']} {s['category']}: {e}")
            resp = ""
        boxes = parse_box_token(resp)
        iou = best_iou(boxes, s["gt_boxes"])
        has_primitive = "<|box|>" in resp
        results.append({
            **s,
            "response": resp,
            "pred_boxes": boxes,
            "iou": iou,
            "has_primitive": has_primitive,
        })
    unload_model(model)

    # Compute tier stats
    tier_stats = defaultdict(lambda: {"count": 0, "iou_sum": 0.0, "has_box": 0, "has_prim": 0,
                                       "empty_gt_with_box": 0, "empty_gt_correct": 0})
    for r in results:
        tier = r["tier"]
        st = tier_stats[tier]
        st["count"] += 1
        st["iou_sum"] += r["iou"]
        st["has_box"] += int(bool(r["pred_boxes"]))
        st["has_prim"] += int(r["has_primitive"])
        if not r["gt_boxes"]:
            st["empty_gt_with_box"] += int(bool(r["pred_boxes"]))
            st["empty_gt_correct"] += int(not bool(r["pred_boxes"]))

    # Print report
    print("\n" + "=" * 70)
    print("TIERED EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Tier':<12} {'Count':>6} {'Avg IoU':>10} {'Has Box':>8} {'Has <|box|>':>12}")
    print("-" * 70)
    for tier in ["easy", "medium", "hard", "negative"]:
        st = tier_stats[tier]
        if st["count"] == 0:
            continue
        avg_iou = st["iou_sum"] / st["count"]
        print(f"{tier:<12} {st['count']:>6} {avg_iou:>10.3f} {st['has_box']:>8} {st['has_prim']:>12}")

    # Negative sample detail
    print("\n--- Negative Sample Detail ---")
    neg_correct = tier_stats["negative"]["empty_gt_correct"]
    neg_wrong = tier_stats["negative"]["empty_gt_with_box"]
    neg_total = tier_stats["negative"]["count"]
    print(f"  Correctly rejected (no box): {neg_correct}/{neg_total}")
    print(f"  Wrongly output box:          {neg_wrong}/{neg_total}")

    # Per-sample detail
    print("\n--- Per-Sample Detail (top 20) ---")
    for i, r in enumerate(results[:20]):
        status = "✅" if r["iou"] > 0.5 else ("⚠️" if r["iou"] > 0.2 else "❌")
        if r["tier"] == "negative":
            status = "✅" if not r["pred_boxes"] else "❌"
        print(f"{status} [{r['tier']:8s}] {r['category']:12s} IoU={r['iou']:.3f} boxes={len(r['pred_boxes'])} | {Path(r['image_path']).name}")

    # Save JSON
    with open(out_dir / "tiered_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_dir / 'tiered_results.json'}")


if __name__ == "__main__":
    main()
