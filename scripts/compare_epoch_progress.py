#!/usr/bin/env python3
"""
Compare inference quality between epoch checkpoints.
Loads models sequentially to fit in 12G VRAM.

Usage:
    python scripts/compare_epoch_progress.py \
        --epoch_a outputs/pretrain/epoch_0 \
        --epoch_b outputs/pretrain/epoch_1 \
        --num_samples 5 \
        --output_dir outputs/epoch_comparison
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

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
    del model
    torch.cuda.empty_cache()
    print("[UNLOAD] Model freed")


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


def make_comparison_image(image_path, gt_boxes, resp_a, resp_b, boxes_a, boxes_b, output_path, label_a, label_b):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    target_h = 400
    scale = target_h / H
    target_w = int(W * scale)
    img_r = img.resize((target_w, target_h), Image.LANCZOS)

    panels = []
    p_gt = img_r.copy()
    if gt_boxes:
        p_gt = draw_boxes(p_gt, gt_boxes, labels=["GT"]*len(gt_boxes), color="lime", width=3, normalized=True)
    panels.append(("Ground Truth", p_gt))

    p_a = img_r.copy()
    if boxes_a:
        p_a = draw_boxes(p_a, boxes_a, labels=[label_a]*len(boxes_a), color="cyan", width=3, normalized=True)
    panels.append((label_a, p_a))

    p_b = img_r.copy()
    if boxes_b:
        p_b = draw_boxes(p_b, boxes_b, labels=[label_b]*len(boxes_b), color="red", width=3, normalized=True)
    panels.append((label_b, p_b))

    total_w = target_w * len(panels) + 20 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, target_h + 60), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    x_off = 0
    for title, panel in panels:
        canvas.paste(panel, (x_off, 60))
        bbox = draw.textbbox((0,0), title, font=font)
        tw = bbox[2]-bbox[0]
        draw.text((x_off + (target_w-tw)//2, 20), title, fill="black", font=font)
        x_off += target_w + 20
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_a", type=str, default="outputs/pretrain/epoch_0")
    parser.add_argument("--epoch_b", type=str, default="outputs/pretrain/epoch_1")
    parser.add_argument("--coco_jsonl", type=str, default="data/pretrain/grounding.jsonl")
    parser.add_argument("--image_root", type=str, default="data/coco/val")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/epoch_comparison")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Comparing: {args.epoch_a}  vs  {args.epoch_b}")
    print("=" * 60)

    # Load test samples
    samples = load_coco_samples(args.coco_jsonl, args.image_root, args.num_samples)
    print(f"Loaded {len(samples)} test samples")

    # Run epoch A
    model_a = load_model(args.epoch_a)
    results_a = []
    for s in tqdm(samples, desc=f"Running {Path(args.epoch_a).name}"):
        resp = run_inference(model_a, s["image_path"], s["prompt"], args.max_new_tokens)
        boxes = parse_box_token(resp)
        results_a.append({"response": resp, "boxes": boxes})
    unload_model(model_a)

    # Run epoch B
    model_b = load_model(args.epoch_b)
    results_b = []
    for s in tqdm(samples, desc=f"Running {Path(args.epoch_b).name}"):
        resp = run_inference(model_b, s["image_path"], s["prompt"], args.max_new_tokens)
        boxes = parse_box_token(resp)
        results_b.append({"response": resp, "boxes": boxes})
    unload_model(model_b)

    # Print comparison table
    label_a = Path(args.epoch_a).name
    label_b = Path(args.epoch_b).name

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    stats = {"a_has_box": 0, "b_has_box": 0, "a_has_primitive": 0, "b_has_primitive": 0}

    for i, (s, ra, rb) in enumerate(zip(samples, results_a, results_b)):
        has_a = bool(ra["boxes"])
        has_b = bool(rb["boxes"])
        prim_a = "<|box|>" in ra["response"]
        prim_b = "<|box|>" in rb["response"]

        stats["a_has_box"] += int(has_a)
        stats["b_has_box"] += int(has_b)
        stats["a_has_primitive"] += int(prim_a)
        stats["b_has_primitive"] += int(prim_b)

        print(f"\n--- Sample {i+1}: {Path(s['image_path']).name} ---")
        print(f"Prompt: {s['prompt']}")
        print(f"  [{label_a}] boxes={ra['boxes']} primitive={prim_a}")
        print(f"           resp: {ra['response'][:150]}{'...' if len(ra['response'])>150 else ''}")
        print(f"  [{label_b}] boxes={rb['boxes']} primitive={prim_b}")
        print(f"           resp: {rb['response'][:150]}{'...' if len(rb['response'])>150 else ''}")

        # Generate comparison image
        from PIL import ImageDraw, ImageFont
        make_comparison_image(
            s["image_path"], s["gt_boxes"],
            ra["response"], rb["response"],
            ra["boxes"], rb["boxes"],
            str(out_dir / f"compare_{i:03d}.png"),
            label_a, label_b,
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n = len(samples)
    print(f"  {label_a}: output boxes {stats['a_has_box']}/{n} | primitive format {stats['a_has_primitive']}/{n}")
    print(f"  {label_b}: output boxes {stats['b_has_box']}/{n} | primitive format {stats['b_has_primitive']}/{n}")

    # Save JSON
    json_out = []
    for s, ra, rb in zip(samples, results_a, results_b):
        json_out.append({
            "image": s["image_path"],
            "prompt": s["prompt"],
            "gt_boxes": s["gt_boxes"],
            label_a: ra,
            label_b: rb,
        })
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\nImages saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
