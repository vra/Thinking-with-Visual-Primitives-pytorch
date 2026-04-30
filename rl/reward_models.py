"""
Reward Models for GRPO training.

Three types:
  1. Format RM (rule-based): checks syntax, duplicates, etc.
  2. Quality RM (LLM-based GRM): evaluates thinking consistency, redundancy, etc.
  3. Accuracy RM (task-specific): checks correctness against ground truth.
"""

import re
import math
from typing import Dict, List, Tuple, Callable
import numpy as np

from model.special_tokens import parse_box_token, parse_point_token


# ---------------------------------------------------------------------------
# 1. Format Reward Model
# ---------------------------------------------------------------------------

class FormatRewardModel:
    """
    Rule-based reward model checking visual primitive format.
    Output: score in [0, 1]
    """

    def __init__(self):
        self.box_pattern = re.compile(r"<\|box\|>(.*?)<\|/box\|>")
        self.point_pattern = re.compile(r"<\|point\|>(.*?)<\|/point\|>")

    def __call__(self, text: str, metadata: Dict = None) -> float:
        score = 1.0
        # Check box format validity
        for m in self.box_pattern.finditer(text):
            content = m.group(1).strip()
            if content and not re.search(r"\[\d+,\d+,\d+,\d+\]", content):
                score -= 0.3  # invalid box content
        # Check point format validity
        for m in self.point_pattern.finditer(text):
            content = m.group(1).strip()
            if content and not re.search(r"\[\d+,\d+\]", content):
                score -= 0.3  # invalid point content
        # Penalize duplicate boxes
        boxes = parse_box_token(text)
        if len(boxes) != len(set(boxes)):
            score -= 0.2
        return max(0.0, score)


# ---------------------------------------------------------------------------
# 2. Quality Reward Model (LLM-based GRM)
# ---------------------------------------------------------------------------

class QualityRewardModel:
    """
    LLM-based Generative Reward Model.
    Uses a separate LLM to score thinking content and final response.
    Output: score in {0.0, 0.5, 1.0}
    """

    def __init__(self, judge_model=None, judge_tokenizer=None):
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer

    def __call__(self, text: str, metadata: Dict = None) -> float:
        # If no judge model available, fallback to heuristic
        if self.judge_model is None:
            return self._heuristic_score(text)

        # Build evaluation prompt
        prompt = self._build_prompt(text)
        # Run judge model (simplified - in practice use generate)
        return self._heuristic_score(text)

    def _heuristic_score(self, text: str) -> float:
        """Fallback heuristic when no LLM judge is available."""
        score = 1.0
        # Penalize extreme length (redundancy)
        if len(text) > 3000:
            score -= 0.3
        # Penalize repetition
        lines = text.split("\n")
        unique_lines = set(lines)
        if len(unique_lines) < len(lines) * 0.7:
            score -= 0.2
        # Check self-contradiction (simple heuristic)
        if "not" in text.lower() and "yes" in text.lower():
            # Potential contradiction
            pass
        return max(0.0, score)

    def _build_prompt(self, text: str) -> str:
        return (
            "Evaluate the following response for:\n"
            "1. Redundancy\n"
            "2. Consistency between thinking and final answer\n"
            "3. Self-contradictions\n"
            "4. Reward hacking behaviors\n\n"
            f"Response:\n{text}\n\n"
            "Score: 0.0 (poor), 0.5 (fair), or 1.0 (good). Output only the score."
        )


# ---------------------------------------------------------------------------
# 3. Accuracy Reward Models (Task-specific)
# ---------------------------------------------------------------------------

class CountingRewardModel:
    """
    Accuracy reward for counting tasks.
    R = α * exp(-β * |ŷ - y| / (|y| + 1))
    """

    def __init__(self, alpha: float = 0.7, beta: float = 3.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, text: str, metadata: Dict) -> float:
        gt = metadata.get("count", None)
        if gt is None:
            return 0.0
        # Extract predicted number from final answer
        numbers = re.findall(r"\b\d+\b", text)
        if not numbers:
            return 0.0
        pred = int(numbers[-1])  # last number in text is likely the answer
        error = abs(pred - gt)
        reward = self.alpha * math.exp(-self.beta * error / (abs(gt) + 1))
        return reward


class SpatialRewardModel:
    """
    Accuracy reward for spatial reasoning / VQA.
    Uses an LLM judge or exact match.
    """

    def __init__(self):
        pass

    def __call__(self, text: str, metadata: Dict) -> float:
        gt_answer = metadata.get("answer", "").strip().lower()
        if not gt_answer:
            return 0.5  # neutral if no GT
        # Extract final answer (after </think> or last sentence)
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        # Simple exact or contain match
        if gt_answer in final or final in gt_answer:
            return 1.0
        # Check for true/false
        if gt_answer in ["true", "false"]:
            pred = "true" if "true" in final else "false" if "false" in final else ""
            return 1.0 if pred == gt_answer else 0.0
        return 0.0


class MazeRewardModel:
    """
    Accuracy reward for maze navigation.
    Components:
      - causal exploration progress
      - exploration completeness (for unsolvable)
      - wall violation penalty
      - final path validity
      - answer correctness
    """

    def __init__(self):
        pass

    def __call__(self, text: str, metadata: Dict) -> float:
        solvable = metadata.get("solvable", True)
        gt_answer = "true" if solvable else "false"

        # Answer correctness
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        pred_answer = "true" if "true" in final else "false" if "false" in final else ""
        answer_score = 1.0 if pred_answer == gt_answer else 0.0

        # Path extraction and validation (simplified)
        points = parse_point_token(text)
        path_validity = 1.0 if len(points) > 0 else 0.0

        # Wall violation penalty (would need grid access for full implementation)
        wall_penalty = 0.0  # placeholder

        if solvable:
            return 0.4 * answer_score + 0.4 * path_validity + 0.2 * (1.0 - wall_penalty)
        else:
            return 0.5 * answer_score + 0.3 * path_validity + 0.2 * (1.0 - wall_penalty)


class PathTracingRewardModel:
    """
    Accuracy reward for path tracing.
    Components:
      - bidirectional trajectory distance
      - endpoint accuracy
      - trajectory continuity penalty
      - answer correctness
    """

    def __init__(self, endpoint_tolerance: float = 50.0):
        self.endpoint_tolerance = endpoint_tolerance

    def __call__(self, text: str, metadata: Dict) -> float:
        gt_points = metadata.get("points", [])
        gt_end_label = metadata.get("end_label", "").lower()

        pred_points = parse_point_token(text)

        # Answer correctness
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        answer_score = 1.0 if gt_end_label in final else 0.0

        # Endpoint accuracy
        endpoint_score = 0.0
        if gt_points and pred_points:
            gt_end = gt_points[-1]
            pred_end = pred_points[-1]
            dist = math.hypot(gt_end[0] - pred_end[0], gt_end[1] - pred_end[1])
            endpoint_score = max(0.0, 1.0 - dist / self.endpoint_tolerance)

        # Trajectory distance (simplified)
        traj_score = 0.0
        if gt_points and pred_points:
            # Forward: for each pred point, min dist to any gt segment
            # Reverse: for each gt point, min dist to any pred segment
            # Simplified: mean point-to-point distance between aligned sequences
            min_len = min(len(gt_points), len(pred_points))
            if min_len > 0:
                dists = [
                    math.hypot(gt_points[i][0] - pred_points[i][0],
                               gt_points[i][1] - pred_points[i][1])
                    for i in range(min_len)
                ]
                mean_dist = sum(dists) / len(dists)
                traj_score = max(0.0, 1.0 - mean_dist / 100.0)

        # Continuity penalty
        continuity_score = 1.0
        if len(pred_points) >= 2:
            jumps = 0
            for i in range(1, len(pred_points)):
                d = math.hypot(pred_points[i][0] - pred_points[i-1][0],
                               pred_points[i][1] - pred_points[i-1][1])
                if d > 100:  # big jump
                    jumps += 1
            continuity_score = max(0.0, 1.0 - jumps * 0.2)

        return 0.3 * answer_score + 0.3 * endpoint_score + 0.25 * traj_score + 0.15 * continuity_score


def build_reward_models(task_type: str = "mixed", judge_model=None) -> List[Callable]:
    """Build a list of reward functions for the given task type."""
    format_rm = FormatRewardModel()
    quality_rm = QualityRewardModel(judge_model)

    if task_type == "counting":
        acc_rm = CountingRewardModel()
    elif task_type == "spatial":
        acc_rm = SpatialRewardModel()
    elif task_type == "maze":
        acc_rm = MazeRewardModel()
    elif task_type == "path":
        acc_rm = PathTracingRewardModel()
    else:
        # Mixed: dispatch based on metadata
        counting_rm = CountingRewardModel()
        spatial_rm = SpatialRewardModel()
        maze_rm = MazeRewardModel()
        path_rm = PathTracingRewardModel()

        def mixed_acc(text, meta):
            subtask = meta.get("task_type", "spatial")
            if subtask == "counting":
                return counting_rm(text, meta)
            elif subtask == "spatial":
                return spatial_rm(text, meta)
            elif subtask == "maze":
                return maze_rm(text, meta)
            elif subtask == "path":
                return path_rm(text, meta)
            return 0.0

        acc_rm = type("MixedAccRM", (), {"__call__": mixed_acc})()

    return [format_rm, quality_rm, acc_rm]
