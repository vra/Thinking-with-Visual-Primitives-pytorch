"""
Programmatic path tracing data generation.

Generates images with curved lines that connect start/end icons.
The model must trace the line using visual primitives (points).
"""

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw


def generate_curve(
    start: Tuple[int, int],
    end: Tuple[int, int],
    num_control_points: int = 3,
    curvature: float = 1.0,
) -> List[Tuple[int, int]]:
    """Generate a smooth curved path from start to end using Bezier-like interpolation."""
    points = [start]
    # Random control points
    for _ in range(num_control_points):
        x = random.randint(min(start[0], end[0]), max(start[0], end[0]))
        y = random.randint(min(start[1], end[1]), max(start[1], end[1]))
        # Add some random offset for curvature
        x += int((random.random() - 0.5) * curvature * 100)
        y += int((random.random() - 0.5) * curvature * 100)
        x = max(0, min(999, x))
        y = max(0, min(999, y))
        points.append((x, y))
    points.append(end)

    # Interpolate smooth path
    path = []
    n_segments = 50
    for i in range(n_segments + 1):
        t = i / n_segments
        # Simple spline interpolation through points
        x = int(np.interp(t, np.linspace(0, 1, len(points)), [p[0] for p in points]))
        y = int(np.interp(t, np.linspace(0, 1, len(points)), [p[1] for p in points]))
        path.append((x, y))
    return path


def generate_crossing_lines(
    img_size: int = 800,
    num_lines: int = 3,
) -> Tuple[Image.Image, List[Tuple[int, int]], str, str]:
    """
    Generate an image with multiple curved lines crossing each other.
    Returns:
      image, target_path_points, start_label, end_label
    """
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # Generate background noise
    for _ in range(100):
        x, y = random.randint(0, img_size - 1), random.randint(0, img_size - 1)
        draw.point((x, y), fill=(240, 240, 240))

    lines = []
    labels_pool = ["crown", "octopus", "star", "heart", "diamond", "club", "spade",
                   "moon", "sun", "cloud", "tree", "flower", "fish", "bird"]
    chosen_labels = random.sample(labels_pool, num_lines + 1)

    for i in range(num_lines):
        start = (random.randint(50, img_size - 50), random.randint(50, img_size - 50))
        end = (random.randint(50, img_size - 50), random.randint(50, img_size - 50))
        path = generate_curve(start, end, num_control_points=random.randint(2, 5))
        color = random.choice(["red", "blue", "green", "purple", "orange", "black"])
        lines.append({"path": path, "color": color, "start": chosen_labels[i], "end": chosen_labels[i+1]})

    # Draw all lines
    for line in lines:
        pts = line["path"]
        # Scale to image size
        img_pts = [(int(x / 999 * img_size), int(y / 999 * img_size)) for x, y in pts]
        draw.line(img_pts, fill=line["color"], width=random.randint(2, 4))

    # Pick one line as target
    target = random.choice(lines)
    # Draw start/end icons as simple text markers
    sx, sy = target["path"][0]
    ex, ey = target["path"][-1]
    sx_img = int(sx / 999 * img_size)
    sy_img = int(sy / 999 * img_size)
    ex_img = int(ex / 999 * img_size)
    ey_img = int(ey / 999 * img_size)
    draw.text((sx_img - 10, sy_img - 10), target["start"][:2].upper(), fill="black")
    draw.text((ex_img - 10, ey_img - 10), target["end"][:2].upper(), fill="black")

    return img, target["path"], target["start"], target["end"]


def generate_path_thinking(path: List[Tuple[int, int]], start_label: str, end_label: str) -> str:
    """Generate thinking content with point visual primitives tracing the path."""
    lines = []
    sx, sy = path[0]
    ex, ey = path[-1]
    lines.append(f"I find the starting point you mentioned, it's located here: <|point|>[[{sx},{sy}]]<|/point|>.")
    lines.append("Following this line, the visual path I observe is:")
    # Sample points adaptively: fewer for straight segments, more for curves
    sampled = [path[0]]
    for i in range(1, len(path)):
        prev = sampled[-1]
        curr = path[i]
        dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        # Adaptive sampling: if distance > threshold, add point
        if dist > 20 or i == len(path) - 1:
            sampled.append(curr)

    pt_str = ",".join(f"[{x},{y}]" for x, y in sampled)
    lines.append(f"<|point|>[{pt_str}]<|/point|>")
    lines.append(f"Following this path, it connects to: <|point|>[[{ex},{ey}]]<|/point|>.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sft/path")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--min_lines", type=int, default=2)
    parser.add_argument("--max_lines", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for i in range(args.num_samples):
        num_lines = random.randint(args.min_lines, args.max_lines)
        img, path, start_label, end_label = generate_crossing_lines(num_lines=num_lines)
        img_path = img_dir / f"path_{i:06d}.png"
        img.save(img_path)

        thinking = generate_path_thinking(path, start_label, end_label)
        question = f"Where does the {start_label} icon connect to? Put the destination icon name in \\boxed{{}}."
        answer = f"\\boxed{{{end_label}}}"

        records.append({
            "image": str(img_path.relative_to(out_dir)),
            "question": question,
            "thinking": thinking,
            "start_label": start_label,
            "end_label": end_label,
            "answer": answer,
        })

    with open(out_dir / "path_data.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {args.num_samples} path tracing samples in {out_dir}")


if __name__ == "__main__":
    main()
