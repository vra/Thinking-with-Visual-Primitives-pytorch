"""
Gradio demo for Thinking with Visual Primitives.

Upload an image and ask the model to locate objects, count items, or reason spatially.

Usage:
    python app.py
    python app.py --model_path outputs/opd/final
    python app.py --model_path yunfengwang/TVP-OPD-Qwen2VL-2B --load_in_4bit
    python app.py --share  # public link
"""

import os
import sys
import re
import argparse
from pathlib import Path

import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import VisualPrimitiveVLM
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/opd/final")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def parse_visual_primitives(text):
    """Extract boxes and points from model output."""
    boxes = []
    labels = []

    box_pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|box\|>\[(.*?)\]<\|/box\|>'
    for match in re.finditer(box_pattern, text):
        label = match.group(1).strip()
        coords_str = match.group(2)
        coord_groups = re.findall(r'\[(\d+),(\d+),(\d+),(\d+)\]', coords_str)
        for c in coord_groups:
            boxes.append(tuple(int(x) for x in c))
            labels.append(label)

    if not boxes:
        box_only = r'<\|box\|>\[(.*?)\]<\|/box\|>'
        for match in re.finditer(box_only, text):
            coords_str = match.group(1)
            coord_groups = re.findall(r'\[?(\d+),(\d+),(\d+),(\d+)\]?', coords_str)
            for c in coord_groups:
                boxes.append(tuple(int(x) for x in c))
                labels.append("object")

    points = []
    point_pattern = r'<\|point\|>\[(.*?)\]<\|/point\|>'
    for match in re.finditer(point_pattern, text):
        coords_str = match.group(1)
        point_groups = re.findall(r'\[(\d+),(\d+)\]', coords_str)
        for p in point_groups:
            points.append((int(p[0]), int(p[1])))

    return boxes, labels, points


def draw_visual_primitives(image, boxes, labels, points):
    """Draw boxes and points on image."""
    img = image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    colors = ["#FF0000", "#00FF00", "#0066FF", "#FFFF00", "#FF00FF", "#00FFFF"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels)):
        px1 = int(x1 * w / 999)
        py1 = int(y1 * h / 999)
        px2 = int(x2 * w / 999)
        py2 = int(y2 * h / 999)
        color = colors[i % len(colors)]
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        text_bbox = draw.textbbox((px1, py1 - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((px1, py1 - 20), label, fill="white", font=font)

    for i, (x, y) in enumerate(points):
        px = int(x * w / 999)
        py = int(y * h / 999)
        r = 5
        color = colors[i % len(colors)]
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)
        if i > 0:
            prev_x, prev_y = points[i - 1]
            ppx = int(prev_x * w / 999)
            ppy = int(prev_y * h / 999)
            draw.line([ppx, ppy, px, py], fill=color, width=2)

    return img


def load_model(model_path, load_in_4bit, device):
    print(f"Loading model from: {model_path}")
    model = VisualPrimitiveVLM.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=load_in_4bit,
    )
    model.eval()
    tokenizer = model.tokenizer
    processor_path = getattr(model, "base_model_path", "Qwen/Qwen2-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print("Model loaded.")
    return model, tokenizer, processor


def predict(image, prompt, max_tokens, do_sample, temperature):
    if image is None:
        return None, "Please upload an image."

    if not prompt.strip():
        prompt = "Describe what you see in the image."

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can understand images and reason with visual primitives."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    gen_kwargs = dict(
        max_new_tokens=int(max_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.vlm.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
    response = response.replace("<|im_end|>", "").strip()

    boxes, labels, points = parse_visual_primitives(response)

    if boxes or points:
        vis_image = draw_visual_primitives(image, boxes, labels, points)
    else:
        vis_image = image

    display_response = response.replace("<|ref|>", "`<ref>`")
    display_response = display_response.replace("<|/ref|>", "`</ref>`")
    display_response = display_response.replace("<|box|>", "`<box>`")
    display_response = display_response.replace("<|/box|>", "`</box>`")
    display_response = display_response.replace("<|point|>", "`<point>`")
    display_response = display_response.replace("<|/point|>", "`</point>`")

    summary_parts = []
    if boxes:
        summary_parts.append(f"{len(boxes)} box(es)")
    if points:
        summary_parts.append(f"{len(points)} point(s)")
    if summary_parts:
        display_response += f"\n\n---\nDetected: {', '.join(summary_parts)}"

    return vis_image, display_response


EXAMPLE_PROMPTS = [
    "Locate the person in the image.",
    "How many people are in the image?",
    "Locate the cat in the image.",
    "Locate the car in the image.",
    "How many dogs are in the image?",
    "Locate the sports ball in the image.",
    "Describe the spatial relationship between objects.",
]


if __name__ == "__main__":
    args = parse_args()

    model, tokenizer, processor = load_model(
        args.model_path, args.load_in_4bit, args.device
    )

    with gr.Blocks(title="Thinking with Visual Primitives", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# Thinking with Visual Primitives

Upload an image and ask the model to **locate**, **count**, or **reason** about objects.
The model outputs structured thinking with bounding boxes `<|box|>` and points `<|point|>` as visual primitives.

[GitHub](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch) |
[Original](https://github.com/mitkox/Thinking-with-Visual-Primitives)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., Locate the person in the image.",
                    value="Locate the person in the image.",
                )
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
                    do_sample = gr.Checkbox(value=False, label="Sampling (uncheck for greedy)")
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                submit_btn = gr.Button("Run", variant="primary")

                gr.Markdown("**Example prompts:**")
                for ep in EXAMPLE_PROMPTS:
                    gr.Button(ep, size="sm").click(
                        fn=lambda p=ep: p, outputs=prompt_input
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Visualization")
                output_text = gr.Markdown(label="Model Response")

        submit_btn.click(
            fn=predict,
            inputs=[input_image, prompt_input, max_tokens, do_sample, temperature],
            outputs=[output_image, output_text],
        )

        gr.Markdown(f"**Model**: `{args.model_path}`")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
