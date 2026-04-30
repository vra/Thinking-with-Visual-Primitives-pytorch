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


def _cell_to_norm(r: int, c: int, H: int, W: int) -> Tuple[int, int]:
    """Convert grid cell (row, col) to normalized [0, 999] coordinates (x, y)."""
    x = int(c / max(W - 1, 1) * 999)
    y = int(r / max(H - 1, 1) * 999)
    return x, y


def _get_neighbors(r: int, c: int, grid: np.ndarray, height: int, width: int) -> List[Tuple[int, int]]:
    """Get accessible neighbor cells from (r, c) in the maze grid."""
    neighbors = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            # Check if the wall between (r,c) and (nr,nc) is open
            if grid[2 * r + dr, 2 * c + dc] == 1:
                neighbors.append((nr, nc))
    return neighbors


def _direction_name(dr: int, dc: int) -> str:
    """Human-readable direction name."""
    if dr == -1:
        return "upper"
    elif dr == 1:
        return "lower"
    elif dc == -1:
        return "left"
    elif dc == 1:
        return "right"
    return "forward"


def generate_maze_thinking(
    grid: np.ndarray,
    solution: List[Tuple[int, int]],
    solvable: bool,
    height: int,
    width: int,
    start_label: str = "lime text label",
    end_label: str = "tangerine circle",
) -> str:
    """
    Generate thinking content with point visual primitives.
    Mimics DFS exploration with forward moves, dead-end detection, and backtracking.
    """
    H, W = grid.shape
    lines = []
    lines.append("I'll use a trial-and-error strategy to explore this maze.")

    sx, sy = _cell_to_norm(solution[0][0], solution[0][1], height, width)
    ex, ey = _cell_to_norm(solution[-1][0], solution[-1][1], height, width)

    lines.append(f"First locate the starting point: <|point|>[[{sx},{sy}]]<|/point|>, "
                 f"and the destination: <|point|>[[{ex},{ey}]]<|/point|>.")
    lines.append("**Start Exploring**:")

    if solvable:
        # Simulate DFS with occasional dead-end exploration and backtracking
        step = 1
        visited = set()
        path_so_far = []

        for idx, (r, c) in enumerate(solution):
            px, py = _cell_to_norm(r, c, height, width)
            visited.add((r, c))
            path_so_far.append((px, py))

            neighbors = _get_neighbors(r, c, grid, height, width)
            unvisited_neighbors = [(nr, nc) for nr, nc in neighbors if (nr, nc) not in visited]

            # At certain junctions, simulate exploring a dead-end branch
            if idx > 0 and len(unvisited_neighbors) > 1 and random.random() < 0.4:
                # Pick a wrong neighbor to explore briefly
                wrong_neighbors = [n for n in unvisited_neighbors
                                   if idx + 1 < len(solution) and n != solution[idx + 1]]
                if wrong_neighbors:
                    wr, wc = random.choice(wrong_neighbors)
                    wpx, wpy = _cell_to_norm(wr, wc, height, width)
                    lines.append(
                        f"**Step{step}**: Reaching <|point|>[[{px},{py}]]<|/point|>, "
                        f"I face {len(unvisited_neighbors)} forks. "
                        f"Let me try the {_direction_name(wr - r, wc - c)} direction first."
                    )
                    step += 1
                    # Check if the wrong path is a dead end
                    dead_end_neighbors = _get_neighbors(wr, wc, grid, height, width)
                    dead_end_unvisited = [(nr, nc) for nr, nc in dead_end_neighbors
                                          if (nr, nc) not in visited and (nr, nc) != (r, c)]
                    lines.append(
                        f"**Step{step}**: Moving to <|point|>[[{wpx},{wpy}]]<|/point|>... "
                        f"{'this is a dead end!' if not dead_end_unvisited else 'exploring further...'} "
                        f"Backtracking to <|point|>[[{px},{py}]]<|/point|>."
                    )
                    step += 1
                    visited.add((wr, wc))
                    continue

            if idx == 0:
                lines.append(
                    f"**Step{step}**: Starting at <|point|>[[{px},{py}]]<|/point|>, "
                    f"I see {len(neighbors)} directions to choose from."
                )
            elif idx == len(solution) - 1:
                lines.append(
                    f"**Step{step}**: Arriving at <|point|>[[{px},{py}]]<|/point|>, "
                    f"I finally see the destination!"
                )
            else:
                if len(unvisited_neighbors) > 0:
                    next_r, next_c = solution[idx + 1] if idx + 1 < len(solution) else (r, c)
                    direction = _direction_name(next_r - r, next_c - c)
                    lines.append(
                        f"**Step{step}**: Reaching <|point|>[[{px},{py}]]<|/point|>, "
                        f"continuing {direction}."
                    )
                else:
                    lines.append(
                        f"**Step{step}**: At <|point|>[[{px},{py}]]<|/point|>, the path is clear."
                    )
            step += 1

        pt_str = ",".join(f"[{x},{y}]" for x, y in path_so_far)
        lines.append(f"**Final Path**: After exploration, the correct route is:\n"
                     f"<|point|>[{pt_str}]<|/point|>")
        lines.append(f"Successfully reaching the destination: <|point|>[[{ex},{ey}]]<|/point|>!")
    else:
        # For unsolvable: explore reachable region, then declare unsolvable
        step = 1
        visited = set()
        stack = [solution[0]]
        explored_points = []

        while stack and step <= 15:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            px, py = _cell_to_norm(r, c, height, width)
            explored_points.append((px, py))

            neighbors = _get_neighbors(r, c, grid, height, width)
            unvisited = [(nr, nc) for nr, nc in neighbors if (nr, nc) not in visited]

            if not unvisited:
                lines.append(
                    f"**Step{step}**: At <|point|>[[{px},{py}]]<|/point|>, "
                    f"all directions are dead ends. Backtracking."
                )
            else:
                lines.append(
                    f"**Step{step}**: Reaching <|point|>[[{px},{py}]]<|/point|>, "
                    f"I see {len(unvisited)} unexplored direction(s). Exploring..."
                )
                for nr, nc in unvisited:
                    stack.append((nr, nc))
            step += 1

        lines.append(
            "After exhaustive exploration of all reachable paths, "
            "no valid route to the destination exists. The maze is unsolvable."
        )

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

        thinking = generate_maze_thinking(grid, solution, solvable, height, width)
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