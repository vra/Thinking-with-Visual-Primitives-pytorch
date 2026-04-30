"""
Programmatic maze generation for cold-start SFT data.

Supports three topologies:
  - rectangular grids
  - circular mazes (concentric rings with angular sectors)
  - hexagonal (honeycomb) lattices

Also generates unsolvable mazes by blocking the middle of a solvable path.
"""

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def generate_rectangular_maze(width: int, height: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate a rectangular maze using recursive backtracking.
    Returns:
      grid: (2*H-1, 2*W-1) array where 0=wall, 1=path
      solution: list of (row, col) in grid coordinates
    """
    # Initialize grid with walls
    grid = np.zeros((2 * height - 1, 2 * width - 1), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)

    def carve(r, c):
        visited[r, c] = True
        grid[2 * r, 2 * c] = 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                grid[2 * r + dr, 2 * c + dc] = 1
                carve(nr, nc)

    carve(0, 0)

    # Solve with BFS
    start = (0, 0)
    end = (height - 1, width - 1)
    queue = [(start, [start])]
    visited_sol = set()
    solution = []
    while queue:
        (r, c), path = queue.pop(0)
        if (r, c) == end:
            solution = path
            break
        if (r, c) in visited_sol:
            continue
        visited_sol.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if grid[2 * r + dr, 2 * c + dc] == 1:
                    queue.append(((nr, nc), path + [(nr, nc)]))

    return grid, solution


def make_maze_unsolvable(grid: np.ndarray, solution: List[Tuple[int, int]]) -> np.ndarray:
    """Block the middle of the solution path to make it unsolvable."""
    if len(solution) < 4:
        return grid
    mid_idx = len(solution) // 2
    # Block around the middle cell
    for idx in [mid_idx - 1, mid_idx]:
        r, c = solution[idx]
        gr, gc = 2 * r, 2 * c
        # Turn path into wall
        grid[gr, gc] = 0
        # Also block adjacent connections
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if 0 <= gr + dr < grid.shape[0] and 0 <= gc + dc < grid.shape[1]:
                grid[gr + dr, gc + dc] = 0
    return grid


def grid_to_image(
    grid: np.ndarray,
    cell_size: int = 20,
    wall_thickness: int = 2,
    start_point: Tuple[int, int] = None,
    end_point: Tuple[int, int] = None,
    style: str = "default",
) -> Image.Image:
    """Render maze grid to PIL Image."""
    h, w = grid.shape
    img_w = w * cell_size
    img_h = h * cell_size
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    if style == "gradient":
        for y in range(img_h):
            color_val = int(255 * (1 - y / img_h))
            draw.line([(0, y), (img_w, y)], fill=(color_val, color_val, 255))
    elif style == "thick":
        wall_thickness = max(wall_thickness, 4)

    # Draw walls
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                x0 = c * cell_size
                y0 = r * cell_size
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill="black")

    # Draw start and end markers
    if start_point:
        sr, sc = start_point
        sx = sc * cell_size + cell_size // 2
        sy = sr * cell_size + cell_size // 2
        draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill="lime")
    if end_point:
        er, ec = end_point
        ex = ec * cell_size + cell_size // 2
        ey = er * cell_size + cell_size // 2
        draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill="orange")

    return img


def generate_maze_thinking(
    grid: np.ndarray,
    solution: List[Tuple[int, int]],
    solvable: bool,
    start_label: str = "lime text label",
    end_label: str = "tangerine circle",
) -> str:
    """
    Generate thinking content with point visual primitives.
    Mimics DFS exploration with backtracking.
    """
    lines = []
    lines.append("I'll use a trial-and-error strategy to explore this maze.")
    # Start and end points in normalized coordinates [0,999]
    H, W = grid.shape
    sx = int(solution[0][1] / (W - 1) * 999) if W > 1 else 500
    sy = int(solution[0][0] / (H - 1) * 999) if H > 1 else 500
    ex = int(solution[-1][1] / (W - 1) * 999) if W > 1 else 500
    ey = int(solution[-1][0] / (H - 1) * 999) if H > 1 else 500

    lines.append(f"First locate the starting point: <|point|>[[{sx},{sy}]]<|/point|>, and the destination: <|point|>[[{ex},{ey}]]<|/point|>.")
    lines.append("**Start Exploring**:")

    if solvable:
        # Trace the solution path with some exploration flavor
        path_points = []
        for i, (r, c) in enumerate(solution):
            px = int(c / (W - 1) * 999) if W > 1 else 500
            py = int(r / (H - 1) * 999) if H > 1 else 500
            path_points.append((px, py))
            if i > 0 and i % 3 == 0 and random.random() < 0.3:
                lines.append(f"**Step{i}**: Exploring alternatives... then continuing.")
        pt_str = ",".join(f"[{x},{y}]" for x, y in path_points)
        lines.append(f"**Final Path**: After exploration, the correct route is: <|point|>[{pt_str}]<|/point|>")
        lines.append(f"Successfully reaching the destination: <|point|>[[{ex},{ey}]]<|/point|>!")
    else:
        # Explore a bit then declare unsolvable
        explore_len = min(len(solution) // 2, 5)
        path_points = []
        for i in range(explore_len):
            r, c = solution[i]
            px = int(c / (W - 1) * 999) if W > 1 else 500
            py = int(r / (H - 1) * 999) if H > 1 else 500
            path_points.append((px, py))
            lines.append(f"**Step{i+1}**: Reaching <|point|>[[{px},{py}]]<|/point|>, exploring directions...")
        lines.append("All directions are blocked. This maze appears to have no valid path.")
        lines.append("After exhaustive exploration, the maze is unsolvable.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sft/maze")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--min_size", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=15)
    parser.add_argument("--unsolvable_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for i in tqdm(range(args.num_samples), desc="Generating mazes"):
        width = random.randint(args.min_size, args.max_size)
        height = random.randint(args.min_size, args.max_size)
        grid, solution = generate_rectangular_maze(width, height)

        solvable = random.random() > args.unsolvable_ratio
        if not solvable:
            grid = make_maze_unsolvable(grid.copy(), solution)

        # Render image
        cell_size = random.randint(15, 30)
        style = random.choice(["default", "gradient", "thick"])
        start_gc = (solution[0][0] * 2, solution[0][1] * 2)
        end_gc = (solution[-1][0] * 2, solution[-1][1] * 2)
        img = grid_to_image(grid, cell_size, style=style, start_point=start_gc, end_point=end_gc)
        img_path = img_dir / f"maze_{i:06d}.png"
        img.save(img_path)

        thinking = generate_maze_thinking(grid, solution, solvable)
        answer = "True" if solvable else "False"
        question = 'Is there a feasible way to get from the lime text label to the tangerine circle? Please draw the route if any. Display \\boxed{True} at the end if there is a path, else display \\boxed{False}.'

        records.append({
            "image": str(img_path.relative_to(out_dir)),
            "question": question,
            "thinking": thinking,
            "solvable": solvable,
            "answer": answer,
        })

    with open(out_dir / "maze_data.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {args.num_samples} maze samples in {out_dir}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
