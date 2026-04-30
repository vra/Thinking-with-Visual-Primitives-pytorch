"""
Evaluation metrics for visual primitive tasks.
"""

import re
import math
from typing import Dict, List, Tuple

from model.special_tokens import parse_box_token, parse_point_token


def counting_exact_match(pred_text: str, gt_count: int) -> bool:
    """Exact match for counting tasks."""
    numbers = re.findall(r"\b\d+\b", pred_text)
    if not numbers:
        return False
    pred = int(numbers[-1])
    return pred == gt_count


def counting_relative_accuracy(pred_text: str, gt_count: int, threshold: float = 0.1) -> bool:
    """Relative accuracy: |pred - gt| / gt <= threshold."""
    numbers = re.findall(r"\b\d+\b", pred_text)
    if not numbers:
        return False
    pred = int(numbers[-1])
    if gt_count == 0:
        return pred == 0
    return abs(pred - gt_count) / gt_count <= threshold


def vqa_accuracy(pred_text: str, gt_answer: str) -> bool:
    """Simple accuracy for VQA / spatial reasoning."""
    pred = pred_text.strip().lower()
    gt = gt_answer.strip().lower()
    # Exact match or containment
    if gt in pred or pred in gt:
        return True
    # For true/false
    if gt in ["true", "false"]:
        pred_tf = "true" if "true" in pred else "false" if "false" in pred else ""
        return pred_tf == gt
    return False


def maze_answer_correctness(pred_text: str, solvable: bool) -> bool:
    """Check if maze solvability answer is correct."""
    pred = pred_text.strip().lower()
    gt = "true" if solvable else "false"
    pred_tf = "true" if "true" in pred else "false" if "false" in pred else ""
    return pred_tf == gt


def maze_path_validity(pred_text: str, min_points: int = 2) -> bool:
    """Check if the predicted path has valid point sequence."""
    points = parse_point_token(pred_text)
    return len(points) >= min_points


def path_endpoint_accuracy(
    pred_text: str,
    gt_end_point: Tuple[int, int],
    tolerance: int = 50,
) -> bool:
    """Check if predicted endpoint is within tolerance of GT."""
    points = parse_point_token(pred_text)
    if not points:
        return False
    pred_end = points[-1]
    dist = math.hypot(pred_end[0] - gt_end_point[0], pred_end[1] - gt_end_point[1])
    return dist <= tolerance


def path_answer_correctness(pred_text: str, gt_label: str) -> bool:
    """Check if the predicted destination label is correct."""
    pred = pred_text.strip().lower()
    gt = gt_label.strip().lower()
    return gt in pred


def path_trajectory_similarity(
    pred_text: str,
    gt_points: List[Tuple[int, int]],
) -> float:
    """
    Compute trajectory similarity using bidirectional distance.
    Returns score in [0, 1].
    """
    pred_points = parse_point_token(pred_text)
    if not pred_points or not gt_points:
        return 0.0

    def point_to_segments_distance(pt, segments):
        min_d = float("inf")
        for i in range(len(segments) - 1):
            d = _point_to_segment_distance(pt, segments[i], segments[i + 1])
            min_d = min(min_d, d)
        return min_d

    # Forward: pred -> gt
    forward_dists = [point_to_segments_distance(p, gt_points) for p in pred_points]
    forward_score = max(0.0, 1.0 - sum(forward_dists) / len(forward_dists) / 100.0)

    # Reverse: gt -> pred
    reverse_dists = [point_to_segments_distance(g, pred_points) for g in gt_points]
    reverse_score = max(0.0, 1.0 - sum(reverse_dists) / len(reverse_dists) / 100.0)

    return (forward_score + reverse_score) / 2.0


def _point_to_segment_distance(p, a, b):
    """Distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)
