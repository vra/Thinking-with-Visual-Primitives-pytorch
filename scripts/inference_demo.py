"""
Simple inference demo for pretrained / fine-tuned checkpoints.

Usage:
    python scripts/inference_demo.py \
        --model_path outputs/pretrain/epoch_0 \
        --image data/coco/val/images/000000000139.jpg \
        --prompt "Locate the cat in the image."

Or batch inference on a JSONL:
    python scripts/inference_demo.py \
        --model_path outputs/pretrain/epoch_0 \
        --jsonl data/sft/counting/counting_data.jsonl \
        --max_samples 5
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe what you see in the image.")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Run batch inference on a JSONL file (expects 'image' and 'question' fields)")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_conversation(image_path: str, prompt: str):
    """Build Qwen2.5-VL style conversation."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that can understand images and reason with visual primitives."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ]
        },
    ]


def single_inference(model, tokenizer, image_path: str, prompt: str, max_new_tokens: int = 512):
    """Run single-sample inference."""
    conv = build_conversation(image_path, prompt)
    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    from transformers import AutoProcessor
    # For adapter directories, load processor from base model path
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
    return response


def batch_inference(model, tokenizer, jsonl_path: str, image_root: Path, max_samples: int = 5, max_new_tokens: int = 512):
    """Run batch inference on a JSONL file."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line.strip()))

    results = []
    for item in samples:
        image_rel = item.get("image", "")
        img_path = image_root / image_rel
        question = item.get("question", item.get("label", "Describe what you see."))

        if not img_path.exists():
            print(f"[SKIP] Image not found: {img_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Image: {img_path}")
        print(f"Question: {question}")
        response = single_inference(model, tokenizer, str(img_path), question, max_new_tokens)
        print(f"Response: {response}")
        results.append({
            "image": str(img_path),
            "question": question,
            "response": response,
        })

    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.model_path}")
    model = VisualPrimitiveVLM.from_pretrained(
        args.model_path,
        device_map=str(device),
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    tokenizer = model.tokenizer
    print("Model loaded.")

    if args.jsonl:
        image_root = Path(args.image_root) if args.image_root else Path(args.jsonl).parent
        results = batch_inference(
            model, tokenizer, args.jsonl, image_root,
            max_samples=args.max_samples, max_new_tokens=args.max_new_tokens
        )
        out_path = Path(args.model_path) / "inference_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {out_path}")

    elif args.image:
        print(f"\nImage: {args.image}")
        print(f"Prompt: {args.prompt}")
        response = single_inference(model, tokenizer, args.image, args.prompt, args.max_new_tokens)
        print(f"\n{'='*60}")
        print(f"Model Response:\n{response}")
        print(f"{'='*60}")

    else:
        print("Please provide --image or --jsonl")


if __name__ == "__main__":
    main()
