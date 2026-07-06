"""
一站式数据准备脚本。

自动完成：
  1. 下载 COCO 2017 检测数据（val 子集，约 5K 图，~1GB）
  2. 生成预训练 grounding 数据（JSONL）
  3. 生成 Counting 冷启动数据（基于 COCO）
  4. 生成 Spatial Reasoning 数据（CLEVR 风格，纯程序生成）
  5. 调用 maze / path 生成脚本

用法:
  python scripts/prepare_all_data.py --output_dir data --coco_split val
"""

import os
import sys
import json
import argparse
import random
import math
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont, ImageColor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


IRREGULAR_PLURALS = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "knife": "knives",
    "child": "children",
}


def pluralize(word: str) -> str:
    """Simple English pluralization for COCO category names."""
    low = word.lower()
    if low in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[low]
    if " " in word:
        parts = word.rsplit(" ", 1)
        return parts[0] + " " + pluralize(parts[1])
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"

from model.special_tokens import normalize_coordinate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--coco_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--coco_subset", type=int, default=5000, help="最多使用多少张 COCO 图片")
    parser.add_argument("--num_counting", type=int, default=2000, help="生成 counting 样本数")
    parser.add_argument("--num_spatial", type=int, default=2000, help="生成 spatial 样本数")
    parser.add_argument("--num_maze", type=int, default=5000, help="生成 maze 样本数")
    parser.add_argument("--num_path", type=int, default=3000, help="生成 path tracing 样本数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. COCO 下载 & 导出
# ---------------------------------------------------------------------------

def download_coco(output_dir: Path, split: str = "val", max_images: int = 5000):
    """使用 datasets 库下载 COCO，导出为图片+标注文件。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装 datasets: pip install datasets")
        sys.exit(1)

    print(f"正在下载 COCO 2017 {split} ...")
    ds = load_dataset("detection-datasets/coco", split=split, streaming=True)

    img_dir = output_dir / "coco" / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = output_dir / "coco" / split / "annotations.json"

    annotations = {"images": [], "annotations": [], "categories": []}
    category_map = {}
    cat_counter = 1

    count = 0
    for sample in tqdm(ds, desc="COCO download"):
        if count >= max_images:
            break
        # sample keys: image, image_id, width, height, objects
        img = sample["image"]
        img_id = sample.get("image_id", count)
        w, h = sample.get("width", img.width), sample.get("height", img.height)

        img_path = img_dir / f"{img_id:012d}.jpg"
        img.save(img_path)

        annotations["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        objects = sample.get("objects", {})
        bboxes = objects.get("bbox", [])
        labels = objects.get("category", [])
        for bbox, label in zip(bboxes, labels):
            if label not in category_map:
                category_map[label] = cat_counter
                annotations["categories"].append({
                    "id": cat_counter,
                    "name": str(label),
                })
                cat_counter += 1
            annotations["annotations"].append({
                "id": len(annotations["annotations"]) + 1,
                "image_id": img_id,
                "category_id": category_map[label],
                "bbox": bbox,  # [x1, y1, x2, y2]
            })
        count += 1

    with open(ann_path, "w") as f:
        json.dump(annotations, f)
    print(f"COCO 导出完成: {img_dir} ({count} 张图), 标注: {ann_path}")
    return img_dir, ann_path, annotations


# ---------------------------------------------------------------------------
# 2. 预训练 Grounding 数据
# ---------------------------------------------------------------------------

def generate_pretrain_data(annotations: dict, output_path: Path):
    """从 COCO 标注生成 grounding JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 按图分组
    img_anns = defaultdict(list)
    for ann in annotations["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    cats = {c["id"]: c["name"] for c in annotations["categories"]}

    records = []
    for img_info in annotations["images"]:
        img_id = img_info["id"]
        W, H = img_info["width"], img_info["height"]
        anns = img_anns.get(img_id, [])
        if not anns:
            continue
        # 按类别分组
        by_cat = defaultdict(list)
        for ann in anns:
            cat_name = cats[ann["category_id"]]
            x1, y1, x2, y2 = ann["bbox"]  # HuggingFace COCO: [x1,y1,x2,y2]
            x1 = max(0.0, min(x1, W))
            y1 = max(0.0, min(y1, H))
            x2 = max(0.0, min(x2, W))
            y2 = max(0.0, min(y2, H))
            box = (
                normalize_coordinate(x1, W),
                normalize_coordinate(y1, H),
                normalize_coordinate(x2, W),
                normalize_coordinate(y2, H),
            )
            by_cat[cat_name].append(box)

        for cat_name, boxes in by_cat.items():
            records.append({
                "image": str(Path("images") / img_info["file_name"]),
                "label": cat_name,
                "boxes": boxes,
                "points": [],
                "normalized": True,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"预训练 grounding 数据: {len(records)} 条 -> {output_path}")


# ---------------------------------------------------------------------------
# 3. Counting 冷启动数据（基于 COCO）
# ---------------------------------------------------------------------------

def generate_counting_thinking(category: str, boxes: List[Tuple[int, int, int, int]], count: int) -> str:
    """程序生成 Counting 的 thinking 内容（统一模板：与 grounding 格式一致）。"""
    lines = []
    lines.append("1. **Analyzing the request**")
    lines.append(f"The user asks me to count the {category} in this image.")
    lines.append("2. **Object grounding**")
    box_strs = []
    for x1, y1, x2, y2 in boxes:
        box_strs.append(f"[{x1},{y1},{x2},{y2}]")
    lines.append(f"I see {count} instance(s) of <|ref|>{category}<|/ref|><|box|>[{','.join(box_strs)}]<|/box|>.")
    lines.append("3. **Conclusion**")
    lines.append(f"There are {count} {category} in this image.")
    return "\n".join(lines)


def generate_counting_data(annotations: dict, output_path: Path, num_samples: int):
    """从 COCO 生成 counting 数据。"""
    from utils.coco_categories import COCO_CATS, get_category_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_anns = defaultdict(list)
    for ann in annotations["annotations"]:
        img_anns[ann["image_id"]].append(ann)
    cats = {c["id"]: c["name"] for c in annotations["categories"]}

    # 筛选出实例数 >=2 的图
    candidates = []
    for img_info in annotations["images"]:
        img_id = img_info["id"]
        W, H = img_info["width"], img_info["height"]
        anns = img_anns.get(img_id, [])
        by_cat = defaultdict(list)
        for ann in anns:
            raw_name = cats[ann["category_id"]]
            cat_name = get_category_name(raw_name)
            x1, y1, x2, y2 = ann["bbox"]  # HuggingFace COCO: [x1,y1,x2,y2]
            x1 = max(0.0, min(x1, W))
            y1 = max(0.0, min(y1, H))
            x2 = max(0.0, min(x2, W))
            y2 = max(0.0, min(y2, H))
            box = (
                normalize_coordinate(x1, W),
                normalize_coordinate(y1, H),
                normalize_coordinate(x2, W),
                normalize_coordinate(y2, H),
            )
            by_cat[cat_name].append(box)
        for cat_name, boxes in by_cat.items():
            if len(boxes) >= 2:
                candidates.append((img_info, cat_name, boxes))

    random.shuffle(candidates)
    candidates = candidates[:num_samples]

    records = []
    templates = [
        "How many {category} are in this image?",
        "How many {category} are in the image?",
        "How many {plural} are in this image?",
        "How many {plural} are in the image?",
        "Count the number of {category}.",
        "Count the number of {plural}.",
        "Count the {plural} in the image.",
        "Count all {plural} in this image.",
        "What is the total count of {category}?",
        "What is the total count of {plural}?",
        "How many {plural} can you see?",
        "How many {plural} are there in the image?",
    ]
    for img_info, cat_name, boxes in candidates:
        plural_name = pluralize(cat_name)
        question = random.choice(templates).format(category=cat_name, plural=plural_name)
        thinking = generate_counting_thinking(cat_name, boxes, len(boxes))
        records.append({
            "image": str(Path("images") / img_info["file_name"]),
            "question": question,
            "thinking": thinking,
            "count": len(boxes),
            "boxes": boxes,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Counting 数据: {len(records)} 条 -> {output_path}")


# ---------------------------------------------------------------------------
# 4. Spatial Reasoning 数据（CLEVR 风格，程序生成）
# ---------------------------------------------------------------------------

def mix_color(color: str, target: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
    """把命名颜色按比例混合到目标 RGB 颜色上，用来模拟变亮/变暗。"""
    base = ImageColor.getrgb(color)
    return tuple(int(base[i] * (1 - ratio) + target[i] * ratio) for i in range(3))


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    x: int,
    y: int,
    obj_w: int,
    obj_h: int,
    fill,
    outline,
    width: int = 2,
):
    """按形状画主体。三角形需要手动画边，其余形状 PIL 可以直接画边框。"""
    bbox = [x, y, x + obj_w, y + obj_h]
    if shape == "circle":
        draw.ellipse(bbox, fill=fill, outline=outline, width=width)
    elif shape == "rectangle":
        draw.rectangle(bbox, fill=fill, outline=outline, width=width)
    else:
        points = [(x + obj_w // 2, y), (x, y + obj_h), (x + obj_w, y + obj_h)]
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=width)


def draw_object(
    draw: ImageDraw.ImageDraw,
    shape: str,
    color: str,
    material: str,
    x: int,
    y: int,
    obj_w: int,
    obj_h: int,
):
    """画一个 CLEVR 风格物体，并把 material 显式编码进视觉样式。"""
    if material == "metal":
        # metal: 先画一个轻微偏移的灰色阴影，再画浅色主体，最后叠白色高光。
        draw_shape(draw, shape, x + 3, y + 3, obj_w, obj_h, fill=(95, 95, 95), outline=(95, 95, 95), width=1)
        draw_shape(
            draw,
            shape,
            x,
            y,
            obj_w,
            obj_h,
            fill=mix_color(color, (255, 255, 255), 0.35),
            outline=mix_color(color, (0, 0, 0), 0.35),
            width=2,
        )

        # 三角形的左上角不一定在物体内部，所以高光放到中轴附近，避免越过斜边。
        if shape == "triangle":
            hx1 = x + obj_w // 2
            hy1 = y + max(5, obj_h // 5)
            hx2 = x + obj_w // 2 + max(4, obj_w // 10)
            hy2 = y + max(8, obj_h // 3)
        else:
            hx1 = x + max(4, obj_w // 6)
            hy1 = y + max(4, obj_h // 6)
            hx2 = x + max(8, obj_w // 2)
            hy2 = y + max(6, obj_h // 4)
        draw.line([(hx1, hy1), (hx2, hy2)], fill="white", width=3)
        draw.ellipse([hx1 - 2, hy1 - 2, hx1 + 4, hy1 + 4], fill="white")
    else:
        # rubber: 不加高光，只用略暗的哑光填充和粗黑边，和 metal 区分开。
        draw_shape(
            draw,
            shape,
            x,
            y,
            obj_w,
            obj_h,
            fill=mix_color(color, (0, 0, 0), 0.05),
            outline="black",
            width=4,
        )


def generate_clevr_image(size: int = 400) -> Image.Image:
    """生成一张 CLEVR 风格的简单几何体图片。"""
    img = Image.new("RGB", (size, size), "lightgray")
    draw = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "yellow", "purple", "cyan", "brown", "gray"]
    shapes = ["circle", "rectangle", "triangle"]
    materials = ["metal", "rubber"]
    objects = []

    num_objs = random.randint(3, 6)
    for _ in range(num_objs):
        obj_w = random.randint(30, 80)
        obj_h = random.randint(30, 80)
        x = random.randint(10, size - obj_w - 10)
        y = random.randint(10, size - obj_h - 10)
        color = random.choice(colors)
        shape = random.choice(shapes)
        material = random.choice(materials)

        draw_object(draw, shape, color, material, x, y, obj_w, obj_h)

        # normalized bbox
        nx1 = normalize_coordinate(x, size)
        ny1 = normalize_coordinate(y, size)
        nx2 = normalize_coordinate(x + obj_w, size)
        ny2 = normalize_coordinate(y + obj_h, size)
        objects.append({
            "shape": shape,
            "color": color,
            "material": material,
            "bbox": [nx1, ny1, nx2, ny2],
        })
    return img, objects


def generate_spatial_question(objects: List[dict], img_path: Path) -> dict:
    """基于生成物体生成空间推理问题和答案。"""
    if len(objects) < 2:
        return None

    # 简化: 只使用 attribute 问题类型
    q_type = "attribute"

    # 选一个目标物体
    target = random.choice(objects)
    target_color = target["color"]
    target_shape = target["shape"]
    target_mat = target["material"]

    question = f"Is there a {target_color} {target_mat} {target_shape}?"
    answer = "Yes"

    # 构建 thinking
    lines = []
    lines.append("1. **Analyzing the request**")
    lines.append(f"The user asks if there is a {target_color} {target_mat} {target_shape}.")
    lines.append("2. **Object grounding**")
    for obj in objects:
        c = obj["color"]
        s = obj["shape"]
        m = obj["material"]
        b = obj["bbox"]
        lines.append(f"I see a <|ref|>{c} {m} {s}<|/ref|><|box|>[[{b[0]},{b[1]},{b[2]},{b[3]}]]<|/box|>.")
    lines.append("3. **Conclusion**")
    lines.append(f"Since there is a {target_color} {target_mat} {target_shape}, the answer is Yes.")
    thinking = "\n".join(lines)

    return {
        "image": str(img_path),
        "question": question,
        "thinking": thinking,
        "answer": answer,
        "boxes": [target["bbox"]],
        "points": [],
    }


def generate_spatial_data(output_dir: Path, num_samples: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for i in range(num_samples):
        img, objects = generate_clevr_image(size=400)
        img_path = img_dir / f"spatial_{i:06d}.png"
        img.save(img_path)

        rec = generate_spatial_question(objects, img_path.relative_to(output_dir))
        if rec:
            records.append(rec)

    with open(output_dir / "spatial_data.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Spatial 数据: {len(records)} 条 -> {output_dir}")


# ---------------------------------------------------------------------------
# 5. 调用 maze / path 生成
# ---------------------------------------------------------------------------

def call_maze_generation(output_dir: Path, num_samples: int):
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_maze_data.py"),
        "--output_dir", str(output_dir),
        "--num_samples", str(num_samples),
    ]
    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def call_path_generation(output_dir: Path, num_samples: int):
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_path_data.py"),
        "--output_dir", str(output_dir),
        "--num_samples", str(num_samples),
    ]
    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. COCO
    coco_img_dir, coco_ann_path, annotations = download_coco(
        output_dir / "coco" / args.coco_split,
        split=args.coco_split,
        max_images=args.coco_subset,
    )

    # 2. 预训练 grounding
    generate_pretrain_data(
        annotations,
        output_dir / "pretrain" / "grounding.jsonl",
    )
    # 创建符号链接或复制 images
    pretrain_img_dir = output_dir / "pretrain" / "images"
    pretrain_img_dir.mkdir(parents=True, exist_ok=True)
    # 这里直接写入相对路径，训练时 image_root 指向 coco/val/images

    # 3. Counting
    generate_counting_data(
        annotations,
        output_dir / "sft" / "counting" / "counting_data.jsonl",
        num_samples=args.num_counting,
    )

    # 4. Spatial
    generate_spatial_data(
        output_dir / "sft" / "spatial",
        num_samples=args.num_spatial,
    )

    # 5. Maze
    call_maze_generation(output_dir / "sft" / "maze", args.num_maze)

    # 6. Path
    call_path_generation(output_dir / "sft" / "path", args.num_path)

    print("\n========================================")
    print("所有数据准备完成！")
    print(f"预训练数据: {output_dir / 'pretrain' / 'grounding.jsonl'}")
    print(f"Counting:   {output_dir / 'sft' / 'counting'}")
    print(f"Spatial:    {output_dir / 'sft' / 'spatial'}")
    print(f"Maze:       {output_dir / 'sft' / 'maze'}")
    print(f"Path:       {output_dir / 'sft' / 'path'}")
    print("========================================")


if __name__ == "__main__":
    main()
