"""
Visualization utilities for visual primitives.
"""

from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(
    image: Union[str, Path, Image.Image],
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str] = None,
    color: str = "red",
    width: int = 3,
    normalized: bool = True,
    image_size: int = None,
) -> Image.Image:
    """
    Draw bounding boxes on an image.
    boxes: list of (x1, y1, x2, y2). If normalized=True, coordinates are in [0, 999].
    """
    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    image_size = image_size or max(W, H)

    labels = labels or [None] * len(boxes)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        if normalized:
            x1 = int(x1 / 999 * W)
            y1 = int(y1 / 999 * H)
            x2 = int(x2 / 999 * W)
            y2 = int(y2 / 999 * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if label:
            draw.text((x1, max(0, y1 - 15)), label, fill=color)
    return img


def draw_points(
    image: Union[str, Path, Image.Image],
    points: List[Tuple[int, int]],
    color: str = "blue",
    radius: int = 5,
    normalized: bool = True,
    connect: bool = False,
) -> Image.Image:
    """
    Draw points (and optionally connect them) on an image.
    points: list of (x, y). If normalized=True, coordinates are in [0, 999].
    """
    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    pts = []
    for x, y in points:
        if normalized:
            x = int(x / 999 * W)
            y = int(y / 999 * H)
        pts.append((x, y))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=color,
        )

    if connect and len(pts) > 1:
        draw.line(pts, fill=color, width=2)

    return img


def visualize_primitive_output(
    image: Union[str, Path, Image.Image],
    text: str,
    output_path: str = None,
) -> Image.Image:
    """
    Parse visual primitives from generated text and draw them on the image.
    Returns the annotated image.
    """
    from model.special_tokens import parse_box_token, parse_point_token

    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    boxes = parse_box_token(text)
    points = parse_point_token(text)

    if boxes:
        img = draw_boxes(img, boxes, color="red", normalized=True)
    if points:
        img = draw_points(img, points, color="blue", normalized=True, connect=True)

    if output_path:
        img.save(output_path)
    return img
