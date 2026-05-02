#!/usr/bin/env python3
"""
Test generalization on arbitrary images (test1.jpg, test2.jpg).
"""
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from model.special_tokens import parse_box_token, denormalize_coordinate


def load_model(model_path: str, device: str = "cuda"):
    print(f"[LOAD] {model_path}")
    model = VisualPrimitiveVLM.from_pretrained(
        model_path, device_map=device, load_in_4bit=False, freeze_vision_tower=True
    )
    model.eval()
    return model


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

    processor_path = getattr(model, "base_model_path", tokenizer.name_or_path)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    # Pre-resize to 336x336 (same as training/eval)
    image_resized = image.resize((336, 336), Image.LANCZOS)
    inputs = processor(text=[text], images=[image_resized], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output_ids = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    resp = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0].strip()
    return resp, orig_w, orig_h


def visualize_boxes(image_path, boxes_norm, save_path, labels=None):
    """Draw bounding boxes on image. boxes_norm are in [0,999]."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFAA00", "#AA00FF", "#00FFFF"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, int(min(w, h) * 0.02)))
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes_norm):
        x1 = denormalize_coordinate(box[0], w)
        y1 = denormalize_coordinate(box[1], h)
        x2 = denormalize_coordinate(box[2], w)
        y2 = denormalize_coordinate(box[3], h)
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) * 0.003)))
        label = labels[i] if labels and i < len(labels) else f"box{i}"
        draw.text((x1, max(0, y1 - font.size)), label, fill=color, font=font)

    img.save(save_path)
    print(f"  Saved visualization: {save_path}")
    return img


def main():
    model_path = "outputs/sft_box/final"
    model = load_model(model_path)

    tests = [
        {
            "image": "test1.jpg",
            "queries": [
                ("Locate the car in the image.", "car"),
                ("Locate the person in the image.", "person"),
                ("Locate the tree in the image.", "tree"),
                ("Locate the motorcycle in the image.", "motorcycle"),
                ("Locate the elephant in the image.", "elephant (should reject)"),
            ],
        },
        {
            "image": "test2.jpg",
            "queries": [
                ("Locate the person in the image.", "person"),
                ("Locate the skateboard in the image.", "skateboard"),
                ("Locate the helmet in the image.", "helmet"),
                ("Locate the knee pad in the image.", "knee pad"),
                ("Locate the bicycle in the image.", "bicycle (should reject)"),
            ],
        },
    ]

    out_dir = Path("outputs/generalization_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_boxes_to_draw = {}  # image_path -> list of (boxes_norm, label)

    for test in tests:
        img_path = test["image"]
        print(f"\n{'='*60}")
        print(f"Image: {img_path}")
        print(f"{'='*60}")
        img_boxes = []
        for prompt, label in test["queries"]:
            print(f"\nQuery: {prompt}")
            resp, orig_w, orig_h = run_inference(model, img_path, prompt)
            print(f"Response: {resp[:300]}...")
            boxes = parse_box_token(resp)
            print(f"Parsed boxes ({len(boxes)}): {boxes}")
            if boxes:
                img_boxes.append((boxes, label))
        all_boxes_to_draw[img_path] = img_boxes

    # Visualize all predictions per image
    for img_path, box_groups in all_boxes_to_draw.items():
        if not box_groups:
            continue
        # Flatten: each box gets its label
        all_boxes = []
        all_labels = []
        for boxes, label in box_groups:
            for b in boxes:
                all_boxes.append(b)
                all_labels.append(label)
        vis_path = out_dir / f"{Path(img_path).stem}_pred.jpg"
        visualize_boxes(img_path, all_boxes, vis_path, all_labels)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
