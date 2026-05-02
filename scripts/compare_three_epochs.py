#!/usr/bin/env python3
"""
Compare inference quality across epoch_0, epoch_1, and final (epoch_2).
Loads models sequentially to fit in 12G VRAM.

Usage:
    python scripts/compare_three_epochs.py \
        --num_samples 5 \
        --output_dir outputs/three_epoch_comparison
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from model.special_tokens import parse_box_token
from utils.visualization import draw_boxes
from utils.coco_categories import COCO_CATS


def load_model(model_path: str, device: str = "cuda"):
    print(f"\n[LOAD] {model_path}")
    model = VisualPrimitiveVLM.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=False,
        freeze_vision_tower=True,
    )
    model.eval()
    return model


def unload_model(model):
    import gc
    # Move model to CPU first to free GPU memory
    if hasattr(model, 'vlm'):
        model.vlm = model.vlm.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[UNLOAD] Model freed")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")


def run_inference(model, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
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
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0].strip()


def load_coco_samples(jsonl_path: str, image_root: str, num_samples: int):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line.strip())
            img_path = Path(image_root) / item.get("image", "")
            label_id = item.get("label", "")
            category = COCO_CATS.get(int(label_id), f"object_{label_id}") if str(label_id).isdigit() else str(label_id)
            prompt = f"Locate the {category} in the image."
            samples.append({
                "image_path": str(img_path),
                "prompt": prompt,
                "gt_boxes": [tuple(b) for b in item.get("boxes", [])],
            })
    return samples


def make_comparison_image(image_path, gt_boxes, results, boxes_dict, output_path):
    """Create side-by-side: GT | epoch_0 | epoch_1 | final"""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    target_h = 350
    scale = target_h / H
    target_w = int(W * scale)
    img_r = img.resize((target_w, target_h), Image.LANCZOS)

    labels = ["Ground Truth", "epoch_0", "epoch_1", "final"]
    colors = ["lime", "cyan", "orange", "red"]
    panels = []

    for label, color in zip(labels, colors):
        p = img_r.copy()
        if label == "Ground Truth" and gt_boxes:
            p = draw_boxes(p, gt_boxes, labels=["GT"]*len(gt_boxes), color=color, width=3, normalized=True)
        elif label in boxes_dict and boxes_dict[label]:
            p = draw_boxes(p, boxes_dict[label], labels=[label]*len(boxes_dict[label]), color=color, width=3, normalized=True)
        panels.append((label, p))

    total_w = target_w * len(panels) + 15 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, target_h + 50), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    x_off = 0
    for title, panel in panels:
        canvas.paste(panel, (x_off, 50))
        bbox = draw.textbbox((0,0), title, font=font)
        tw = bbox[2]-bbox[0]
        draw.text((x_off + (target_w-tw)//2, 18), title, fill="black", font=font)
        x_off += target_w + 15
    canvas.save(output_path)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


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
    parser.add_argument("--epoch_0", type=str, default="outputs/pretrain/epoch_0")
    parser.add_argument("--epoch_1", type=str, default="outputs/pretrain/epoch_1")
    parser.add_argument("--final", type=str, default="outputs/pretrain/final")
    parser.add_argument("--coco_jsonl", type=str, default="data/pretrain/grounding.jsonl")
    parser.add_argument("--image_root", type=str, default="data/coco/val")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/three_epoch_comparison")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = {
        "epoch_0": args.epoch_0,
        "epoch_1": args.epoch_1,
        "final": args.final,
    }

    print("=" * 70)
    print("Three-Epoch Comparison: epoch_0 vs epoch_1 vs final")
    print("=" * 70)

    samples = load_coco_samples(args.coco_jsonl, args.image_root, args.num_samples)
    print(f"Loaded {len(samples)} test samples\n")

    # Run each epoch sequentially
    all_results = {}
    for name, path in epochs.items():
        model = load_model(path)
        results = []
        for s in tqdm(samples, desc=f"Running {name}"):
            resp = run_inference(model, s["image_path"], s["prompt"], args.max_new_tokens)
            boxes = parse_box_token(resp)
            results.append({"response": resp, "boxes": boxes})
        all_results[name] = results
        unload_model(model)

    # Print comparison table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    summary = {name: {"has_box": 0, "has_primitive": 0, "mean_iou": []} for name in epochs}

    for i, s in enumerate(samples):
        print(f"\n--- Sample {i+1}: {Path(s['image_path']).name} | {s['prompt']} ---")
        print(f"GT boxes: {s['gt_boxes']}")

        boxes_dict = {"Ground Truth": s["gt_boxes"]}
        for name in epochs:
            r = all_results[name][i]
            has_box = bool(r["boxes"])
            has_prim = "<|box|>" in r["response"]
            iou = best_iou(r["boxes"], s["gt_boxes"]) if s["gt_boxes"] else -1

            summary[name]["has_box"] += int(has_box)
            summary[name]["has_primitive"] += int(has_prim)
            if iou >= 0:
                summary[name]["mean_iou"].append(iou)

            boxes_dict[name] = r["boxes"]
            print(f"  [{name:8s}] boxes={r['boxes']}, iou={iou:.3f}, prim={has_prim}")
            resp_short = r["response"][:120] + "..." if len(r["response"]) > 120 else r["response"]
            print(f"            resp: {resp_short}")

        # Save comparison image
        make_comparison_image(
            s["image_path"], s["gt_boxes"],
            all_results, boxes_dict,
            str(out_dir / f"compare_{i:03d}.png"),
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(samples)
    print(f"{'Metric':<25} {'epoch_0':>10} {'epoch_1':>10} {'final':>10}")
    print("-" * 70)
    for metric in ["has_box", "has_primitive"]:
        vals = [f"{summary[name][metric]}/{n}" for name in epochs]
        print(f"{metric:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    for metric in ["mean_iou"]:
        vals = []
        for name in epochs:
            ious = summary[name][metric]
            vals.append(f"{sum(ious)/len(ious):.3f}" if ious else "N/A")
        print(f"{metric:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # Save JSON
    json_out = []
    for i, s in enumerate(samples):
        entry = {
            "image": s["image_path"],
            "prompt": s["prompt"],
            "gt_boxes": s["gt_boxes"],
        }
        for name in epochs:
            entry[name] = all_results[name][i]
        json_out.append(entry)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
