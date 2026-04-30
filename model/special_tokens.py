"""
Special tokens for visual primitives.

Paper format:
  Box:    <|ref|>TARGET<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>
  Point:  <|point|>[[x1,y1],[x2,y2],...]<|/point|>

Coordinates are normalized to discrete integers in [0, 999].
"""

from typing import List, Tuple

# Special token strings
REF_START = "<|ref|>"
REF_END = "<|/ref|>"
BOX_START = "<|box|>"
BOX_END = "<|/box|>"
POINT_START = "<|point|>"
POINT_END = "<|/point|>"

SPECIAL_TOKENS = [
    REF_START, REF_END,
    BOX_START, BOX_END,
    POINT_START, POINT_END,
]

# For parsing
import re

BOX_PATTERN = re.compile(
    r"<\|box\|>(.*?)<\|/box\|>", re.DOTALL
)
POINT_PATTERN = re.compile(
    r"<\|point\|>(.*?)<\|/point\|>", re.DOTALL
)
REF_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL
)


def normalize_coordinate(val: float, image_size: int) -> int:
    """Map a pixel coordinate to [0, 999]."""
    return int(round(val / image_size * 999))


def denormalize_coordinate(val: int, image_size: int) -> int:
    """Map a normalized coordinate [0, 999] back to pixel space."""
    return int(round(val / 999 * image_size))


def format_box_token(boxes: List[Tuple[int, int, int, int]]) -> str:
    """
    boxes: list of (x1, y1, x2, y2) in normalized [0, 999] ints.
    Returns: <|box|>[[x1,y1,x2,y2],...]<|/box|>
    """
    if not boxes:
        return f"{BOX_START}[]{BOX_END}"
    inner = ",".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in boxes)
    return f"{BOX_START}[{inner}]{BOX_END}"


def format_point_token(points: List[Tuple[int, int]]) -> str:
    """
    points: list of (x, y) in normalized [0, 999] ints.
    Returns: <|point|>[[x1,y1],[x2,y2],...]<|/point|>
    """
    if not points:
        return f"{POINT_START}[]{POINT_END}"
    inner = ",".join(f"[{x},{y}]" for x, y in points)
    return f"{POINT_START}[{inner}]{POINT_END}"


def parse_box_token(text: str) -> List[Tuple[int, int, int, int]]:
    """Parse all box groups from text. Returns list of (x1,y1,x2,y2)."""
    boxes = []
    for m in BOX_PATTERN.finditer(text):
        content = m.group(1).strip()
        # Match [[x1,y1,x2,y2],[x3,y3,x4,y4],...]
        for bm in re.finditer(r"\[(\d+),(\d+),(\d+),(\d+)\]", content):
            boxes.append(tuple(int(bm.group(i)) for i in range(1, 5)))
    return boxes


def parse_point_token(text: str) -> List[Tuple[int, int]]:
    """Parse all point groups from text. Returns list of (x,y)."""
    points = []
    for m in POINT_PATTERN.finditer(text):
        content = m.group(1).strip()
        for pm in re.finditer(r"\[(\d+),(\d+)\]", content):
            points.append((int(pm.group(1)), int(pm.group(2))))
    return points


def parse_ref_text(text: str) -> List[str]:
    """Parse all reference texts from text."""
    refs = []
    for m in REF_PATTERN.finditer(text):
        refs.append(m.group(1).strip())
    return refs


def add_special_tokens(tokenizer):
    """Add visual primitive special tokens to a tokenizer."""
    special = {"additional_special_tokens": SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special)
    return tokenizer
