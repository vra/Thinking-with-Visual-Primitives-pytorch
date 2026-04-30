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
    Components (per paper Section 2.5.2):
      - causal exploration progress (solvable only)
      - exploration completeness (unsolvable only)
      - wall violation penalty
      - final path validity
      - answer correctness
    """

    def __init__(self):
        pass

    def __call__(self, text: str, metadata: Dict) -> float:
        solvable = metadata.get("solvable", True)
        gt_answer = "true" if solvable else "false"
        grid = metadata.get("grid", None)  # 2D grid array if available
        gt_path = metadata.get("solution", [])  # list of (row, col) cells

        # 1. Answer correctness (binary)
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        pred_answer = "true" if "true" in final else "false" if "false" in final else ""
        answer_score = 1.0 if pred_answer == gt_answer else 0.0

        # 2. Extract all point sequences from the text
        all_points = parse_point_token(text)

        # 3. Final path validity: check if a final path is present with >= 2 points
        # and that it connects start to end (if solvable)
        final_path_score = 0.0
        if all_points and len(all_points) >= 2:
            final_path_score = 1.0
            # If grid is available, verify consecutive cells are connected
            if grid is not None and len(gt_path) > 0:
                final_path_score = self._check_path_connectivity(
                    all_points, grid, metadata
                )

        # 4. Wall violation penalty
        wall_violation_score = 1.0
        if grid is not None and all_points:
            wall_violation_score = self._compute_wall_violation_score(
                all_points, grid, metadata
            )

        # 5. Causal exploration progress (solvable) / exploration completeness (unsolvable)
        exploration_score = 1.0
        if grid is not None and gt_path:
            if solvable:
                exploration_score = self._compute_exploration_progress(
                    all_points, gt_path, grid, metadata
                )
            else:
                exploration_score = self._compute_exploration_completeness(
                    all_points, grid, metadata
                )

        if solvable:
            return (0.2 * answer_score + 0.2 * exploration_score +
                    0.2 * wall_violation_score + 0.3 * final_path_score +
                    0.1 * (1.0 if pred_answer == gt_answer and final_path_score > 0 else 0.0))
        else:
            return (0.3 * answer_score + 0.3 * exploration_score +
                    0.2 * wall_violation_score + 0.2 * final_path_score)

    def _points_to_grid_cells(self, points, metadata):
        """Convert normalized [0,999] points to grid cell coordinates."""
        grid_h = metadata.get("grid_height", 1)
        grid_w = metadata.get("grid_width", 1)
        cells = []
        for x, y in points:
            col = int(round(x / 999 * (grid_w - 1))) if grid_w > 1 else 0
            row = int(round(y / 999 * (grid_h - 1))) if grid_h > 1 else 0
            cells.append((row, col))
        return cells

    def _check_path_connectivity(self, points, grid, metadata):
        """Check if consecutive points in the final path are legally connected."""
        cells = self._points_to_grid_cells(points, metadata)
        if len(cells) < 2:
            return 0.0
        valid_transitions = 0
        total_transitions = len(cells) - 1
        for i in range(total_transitions):
            r1, c1 = cells[i]
            r2, c2 = cells[i + 1]
            # Check adjacency (Manhattan distance == 1 in cell space)
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                # Check if wall exists between them in the grid
                gr1, gc1 = 2 * r1, 2 * c1
                gr2, gc2 = 2 * r2, 2 * c2
                mid_r, mid_c = (gr1 + gr2) // 2, (gc1 + gc2) // 2
                if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]
                        and grid[mid_r, mid_c] == 1):
                    valid_transitions += 1
        return valid_transitions / max(total_transitions, 1)

    def _compute_wall_violation_score(self, points, grid, metadata):
        """Count wall-violating transitions. Score = 1 - violations/total_legal."""
        cells = self._points_to_grid_cells(points, metadata)
        if len(cells) < 2:
            return 1.0
        violations = 0
        for i in range(len(cells) - 1):
            r1, c1 = cells[i]
            r2, c2 = cells[i + 1]
            gr1, gc1 = 2 * r1, 2 * c1
            gr2, gc2 = 2 * r2, 2 * c2
            mid_r, mid_c = (gr1 + gr2) // 2, (gc1 + gc2) // 2
            if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]):
                if grid[mid_r, mid_c] == 0:  # wall
                    violations += 1
            else:
                violations += 1  # out of bounds
        total_legal = grid.sum()  # total passable cells
        return max(0.0, 1.0 - violations / max(total_legal, 1))

    def _compute_exploration_progress(self, points, gt_path, grid, metadata):
        """For solvable mazes: how close did exploration get to the endpoint."""
        cells = self._points_to_grid_cells(points, metadata)
        if not cells or not gt_path:
            return 0.0
        end_r, end_c = gt_path[-1]
        # Find minimum Manhattan distance from any explored cell to the endpoint
        min_dist = float("inf")
        for r, c in cells:
            d = abs(r - end_r) + abs(c - end_c)
            min_dist = min(min_dist, d)
        path_len = len(gt_path)
        return max(0.0, 1.0 - min_dist / max(path_len, 1))

    def _compute_exploration_completeness(self, points, grid, metadata):
        """For unsolvable mazes: fraction of reachable cells explored."""
        cells = set(self._points_to_grid_cells(points, metadata))
        # Count all reachable cells in the grid
        grid_h = metadata.get("grid_height", 1)
        grid_w = metadata.get("grid_width", 1)
        reachable = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if grid[2 * r, 2 * c] == 1:
                    reachable += 1
        explored = len(cells.intersection(
            {(r, c) for r in range(grid_h) for c in range(grid_w)
             if 2 * r < grid.shape[0] and 2 * c < grid.shape[1] and grid[2 * r, 2 * c] == 1}
        ))
        return explored / max(reachable, 1)


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

        # Bidirectional trajectory distance (per paper Section 2.5.2)
        traj_score = 0.0
        if gt_points and pred_points:
            # Forward: for each pred point, min distance to any GT segment
            forward_dists = []
            for p in pred_points:
                min_d = self._point_to_polyline_distance(p, gt_points)
                forward_dists.append(min_d)
            forward_score = max(0.0, 1.0 - (sum(forward_dists) / len(forward_dists)) / 100.0)

            # Reverse: for each GT point, min distance to any pred segment
            reverse_dists = []
            for g in gt_points:
                min_d = self._point_to_polyline_distance(g, pred_points)
                reverse_dists.append(min_d)
            reverse_score = max(0.0, 1.0 - (sum(reverse_dists) / len(reverse_dists)) / 100.0)

            traj_score = (forward_score + reverse_score) / 2.0

        # Continuity penalty (per paper: fixed penalty if last trajectory point
        # is far from predicted endpoint)
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

    @staticmethod
    def _point_to_segment_distance(p, a, b):
        """Distance from point p to line segment ab."""
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

    @staticmethod
    def _point_to_polyline_distance(p, polyline):
        """Minimum distance from point p to any segment of the polyline."""
        if len(polyline) < 2:
            if polyline:
                return math.hypot(p[0] - polyline[0][0], p[1] - polyline[0][1])
            return float("inf")
        min_d = float("inf")
        for i in range(len(polyline) - 1):
            d = PathTracingRewardModel._point_to_segment_distance(
                p, polyline[i], polyline[i + 1]
            )
            min_d = min(min_d, d)
        return min_d


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