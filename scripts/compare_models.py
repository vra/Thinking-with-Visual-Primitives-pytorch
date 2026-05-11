"""
Compare model outputs across training stages on the same image.
"""
import os
import sys
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from utils.visualization import draw_boxes, visualize_primitive_output


def parse_response(text: str):
    """Parse <|ref|>label<|/ref|><|box|>...<|/box|> from response."""
    # Remove im_end and other special tokens
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    
    boxes = []
    labels = []
    
    # Pattern: <|ref|>label<|/ref|><|box|>[[...]]<|/box|>
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>.*?<\|box\|>(.*?)<\|/box\|>",
        re.DOTALL
    )
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        box_str = m.group(2).strip()
        # Parse [[x1,y1,x2,y2], ...]
        box_list = re.findall(r"\[(\d+),(\d+),(\d+),(\d+)\]", box_str)
        for b in box_list:
            boxes.append(tuple(int(x) for x in b))
            labels.append(label)
    
    return boxes, labels, text


def run_inference(model_path: str, image_path: str, prompt: str, device="cuda"):
    model = VisualPrimitiveVLM.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=False,
    )
    model.eval()
    tokenizer = model.tokenizer
    
    from transformers import AutoProcessor
    processor_path = getattr(model, "base_model_path", tokenizer.name_or_path)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    with torch.no_grad():
        output_ids = model.vlm.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
    return response


def main():
    image_path = "test_image.jpg"
    prompt = "Locate the three men wearing green vests in the image."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    stages = [
        ("Pretrain", "outputs/pretrain/final", "orange"),
        ("SFT Box", "outputs/sft_box/final", "green"),
        ("SFT Point", "outputs/sft_point/final", "blue"),
        ("OPD", "outputs/opd/final", "red"),
    ]
    
    results = []
    for name, path, color in stages:
        print(f"\n>>> Running {name} ...")
        try:
            response = run_inference(path, image_path, prompt, device)
            boxes, labels, raw = parse_response(response)
            print(f"    Response: {raw[:200]}...")
            print(f"    Boxes: {boxes}, Labels: {labels}")
            results.append((name, boxes, labels, color, raw))
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append((name, [], [], color, f"ERROR: {e}"))
        # Clear GPU cache between models
        torch.cuda.empty_cache()
    
    # Build comparison image: 2x2 grid
    base_img = Image.open(image_path).convert("RGB")
    W, H = base_img.size
    
    grid_w = W * 2
    grid_h = H * 2
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    
    for idx, (name, boxes, labels, color, raw) in enumerate(results):
        img = base_img.copy()
        if boxes:
            img = draw_boxes(img, boxes, labels=labels, color=color, width=3, normalized=True)
        
        # Add title
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((5, 5), f"{name}: {len(boxes)} box(es)", fill=color, font=font)
        
        x = (idx % 2) * W
        y = (idx // 2) * H
        grid.paste(img, (x, y))
    
    output_path = "comparison_4stages.jpg"
    grid.save(output_path, quality=95)
    print(f"\nComparison saved to: {output_path}")
    
    # Also save individual annotated images
    for name, boxes, labels, color, raw in results:
        img = base_img.copy()
        if boxes:
            img = draw_boxes(img, boxes, labels=labels, color=color, width=3, normalized=True)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((5, 5), f"{name}: {len(boxes)} box(es)", fill=color, font=font)
        img.save(f"stage_{name.lower().replace(' ', '_')}.jpg", quality=95)
    print("Individual images saved.")


if __name__ == "__main__":
    main()
