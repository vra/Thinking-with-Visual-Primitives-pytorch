# Thinking with Visual Primitives — PyTorch Implementation

<p align="center">
    <a href="https://huggingface.co/yunfengwang"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face"></a>
    <a href="https://huggingface.co/datasets/yunfengwang/TVP-Training-Data"><img src="https://img.shields.io/badge/%F0%9F%93%A6%20Dataset-HF-orange" alt="Dataset"></a>
    <a href="https://arxiv.org/abs/2506.00000"><img src="https://img.shields.io/badge/%F0%9F%93%91%20Paper-arXiv-b31b1b" alt="Paper"></a>
    <a href="README_ZH.md"><img src="https://img.shields.io/badge/%F0%9F%87%A8%F0%9F%87%B3%20%E4%B8%AD%E6%96%87-README-blue" alt="中文文档"></a>
</p>

<p align="center">
    <a href="https://huggingface.co/yunfengwang/TVP-OPD-Qwen2VL-2B">OPD Model</a> |
    <a href="https://huggingface.co/yunfengwang/TVP-SFTBox-Qwen2VL-2B">SFT Box Expert</a> |
    <a href="https://huggingface.co/yunfengwang/TVP-SFTPoint-Qwen2VL-2B">SFT Point Expert</a> |
    <a href="https://huggingface.co/yunfengwang/TVP-Pretrain-Qwen2VL-2B">Pretrain</a> |
    <a href="https://huggingface.co/datasets/yunfengwang/TVP-Training-Data">Dataset</a>
</p>

> Unofficial PyTorch reproduction of [*Thinking with Visual Primitives*](https://arxiv.org/abs/2506.00000).

This project implements a multi-stage training pipeline that teaches multimodal LLMs to reason with **bounding boxes** and **points** as first-class "thought units" — interleaving spatial coordinates within chain-of-thought to close the **Reference Gap** in visual reasoning.

## Overview

```
Stage 1: Pretraining        — Learn to output visual primitive format
Stage 2: Specialized SFT    — Expert fine-tuning (Box expert + Point expert)
Stage 3: On-Policy Distill  — Distill both experts into a unified model
```

The model outputs structured thinking with embedded coordinates:

```
1. Analyzing the request
The user asks me to locate the cat in this image.
2. Object grounding
I see a <|ref|>cat<|/ref|><|box|>[[370,334,408,497]]<|/box|>.
3. Conclusion
The cat is located at the specified coordinates.
```

## Models

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| Pretrain | [yunfengwang/TVP-Pretrain-Qwen2VL-2B](https://huggingface.co/yunfengwang/TVP-Pretrain-Qwen2VL-2B) | Base model with visual primitive format |
| SFT Box Expert | [yunfengwang/TVP-SFTBox-Qwen2VL-2B](https://huggingface.co/yunfengwang/TVP-SFTBox-Qwen2VL-2B) | Grounding specialist (counting, spatial) |
| SFT Point Expert | [yunfengwang/TVP-SFTPoint-Qwen2VL-2B](https://huggingface.co/yunfengwang/TVP-SFTPoint-Qwen2VL-2B) | Pointing specialist (maze, path tracing) |
| OPD Unified | [yunfengwang/TVP-OPD-Qwen2VL-2B](https://huggingface.co/yunfengwang/TVP-OPD-Qwen2VL-2B) | Final distilled model (both capabilities) |

## Example Results

We compare the model outputs across three training stages on the same images, showing complete model responses.

### Grounding: "Locate the person in the image."

<table>
<tr><th>Stage</th><th>Complete Output</th><th>Visualization</th></tr>
<tr>
<td><b>Pretrain</b></td>
<td>

```
<|ref|>Person<|/ref|><|box|>[[480,201,999,999]]<|/box|>
```
No structured thinking. Box is oversized (covers most of the image).
</td>
<td><img src="test-images/results/pretrain_img_001.jpg" width="250"></td>
</tr>
<tr>
<td><b>SFT Box</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to locate the person in this image.
2. **Object grounding**
I see a <|ref|>person<|/ref|><|box|>[[511,208,738,963]]<|/box|>.
3. **Conclusion**
The person is located at the specified coordinates.
The person is located at [[511,208,738,963]].
```
Structured thinking with accurate box on the silhouette.
</td>
<td><img src="test-images/results/sft_box_img_001.jpg" width="250"></td>
</tr>
<tr>
<td><b>OPD</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to locate the person in this image.
2. **Object grounding**
After carefully scanning the entire image,
I do not see any person present.
3. **Conclusion**
There is no person in this image.
```
Known limitation: OPD loses silhouette detection after multi-task distillation (SFT Box handles it correctly).
</td>
<td><img src="test-images/results/opd_img_001.jpg" width="250"></td>
</tr>
</table>

### Grounding: "Locate the sports ball in the image."

<table>
<tr><th>Stage</th><th>Complete Output</th><th>Visualization</th></tr>
<tr>
<td><b>Pretrain</b></td>
<td>

```
<|ref|>[[278,267,729,759]]<|/ref|>
```
No label, wrong tag usage, oversized box.
</td>
<td><img src="test-images/results/pretrain_img_003.jpg" width="250"></td>
</tr>
<tr>
<td><b>SFT Box</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to locate the sports ball in this image.
2. **Object grounding**
I see a <|ref|>sports ball<|/ref|><|box|>[[277,244,479,510]]<|/box|>.
3. **Conclusion**
The sports ball is located at the specified coordinates.
The sports ball is located at [[277,244,479,510]].
```
Correct label, tight box, full structured thinking.
</td>
<td><img src="test-images/results/sft_box_img_003.jpg" width="250"></td>
</tr>
<tr>
<td><b>OPD</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to locate the sports ball in this image.
2. **Object grounding**
I see a <|ref|>sports ball<|/ref|><|box|>[[277,244,479,510]]<|/box|>.
3. **Conclusion**
The sports ball is located at the specified coordinates.
The sports ball is located at [[277,244,479,510]].
```
Identical quality to SFT Box (KL=0.35).
</td>
<td><img src="test-images/results/opd_img_003.jpg" width="250"></td>
</tr>
</table>

### Counting: "How many people are in the image?"

<table>
<tr><th>Stage</th><th>Complete Output</th><th>Visualization</th></tr>
<tr>
<td><b>Pretrain</b></td>
<td>

```
There are two people in the image.
```
Correct count but no boxes or structured thinking.
</td>
<td><img src="test-images/results/pretrain_img_000.jpg" width="250"></td>
</tr>
<tr>
<td><b>SFT Box</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to count the person in this image.
2. **Object grounding**
I see 2 instance(s) of <|ref|>person<|/ref|>
<|box|>[[338,121,630,923],[634,154,888,945]]<|/box|>.
3. **Conclusion**
There are 2 person in this image.
```
Structured thinking with count + bounding boxes for each person.
</td>
<td><img src="test-images/results/sft_box_img_000.jpg" width="250"></td>
</tr>
<tr>
<td><b>OPD</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to count the person in this image.
2. **Object grounding**
I see 2 instance(s) of <|ref|>person<|/ref|>
<|box|>[[339,130,632,923],[642,157,887,940]]<|/box|>.
3. **Conclusion**
There are 2 person in this image.
```
Preserves SFT Box quality — structured counting with boxes.
</td>
<td><img src="test-images/results/opd_img_000.jpg" width="250"></td>
</tr>
</table>

### Counting: "How many sports balls are in the image?"

<table>
<tr><th>Stage</th><th>Complete Output</th><th>Visualization</th></tr>
<tr>
<td><b>Pretrain</b></td>
<td>

```
There are four sports balls in the image.
```
Correct count, no boxes.
</td>
<td><img src="test-images/results/pretrain_img_002.jpg" width="250"></td>
</tr>
<tr>
<td><b>SFT Box</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to count out sports balls in this image.
2. **Object grounding**
I see 4 instance(s) of <|ref|>sports ball<|/ref|>
<|box|>[[386,265,653,587],[87,497,445,752],
[448,587,682,878],[642,456,904,695]]<|/box|>.
3. **Conclusion**
There are 4 sports ball in this image.
```
Structured thinking with 4 bounding boxes — fixed by prompt template diversification.
</td>
<td><img src="test-images/results/sft_box_img_002.jpg" width="250"></td>
</tr>
<tr>
<td><b>OPD</b></td>
<td>

```
1. **Analyzing the request**
The user asks me to count out sports balls in this image.
2. **Object grounding**
I see 4 instance(s) of <|ref|>sports ball<|/ref|>
<|box|>[[386,263,653,587],[87,497,445,752],
[510,571,675,878],[643,453,904,695]]<|/box|>.
3. **Conclusion**
There are 4 sports ball in this image.
```
Preserves counting ability with accurate boxes.
</td>
<td><img src="test-images/results/opd_img_002.jpg" width="250"></td>
</tr>
</table>

**Key observations:**
- **Pretrain** learns the visual primitive token format but produces oversized boxes and no structured thinking
- **SFT Box** adds structured thinking (Analyzing → Grounding → Conclusion) with accurate bounding boxes; counting now produces boxes via prompt template diversification (3→12 templates with plural forms)
- **OPD** preserves SFT Box counting/grounding quality with nearly identical box coordinates, while also supporting point-based tasks (maze, path tracing)
- **Prompt diversity fix**: expanding counting templates from 3 to 12 (with singular/plural, "the"/"this", varied verbs) fixed the issue where "How many sports balls are in the image?" produced plain text instead of structured thinking with boxes
- **Distillation tuning**: lr=5e-7 + temperature=1.0 + positive-only grounding data prevents catastrophic forgetting during multi-task distillation
- **neg_ratio tuning**: reducing negative sample ratio from 0.30 to 0.15 in SFT fixed over-rejection

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Thinking-with-Visual-Primitives-pytorch.git
cd Thinking-with-Visual-Primitives-pytorch

conda create -n vprim python=3.10 -y
conda activate vprim
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, CUDA ≥ 11.8, GPU with 12GB+ VRAM (tested on RTX 4070 Ti 12GB).

### Gradio Demo

```bash
# Interactive web demo with visualization
python app.py --model_path outputs/opd/final --load_in_4bit

# Or use HuggingFace model directly
python app.py --model_path yunfengwang/TVP-OPD-Qwen2VL-2B --load_in_4bit

# Public shareable link
python app.py --model_path outputs/opd/final --load_in_4bit --share
```

Upload an image, type a prompt (e.g., "Locate the cat"), and see the model's structured reasoning with bounding boxes drawn on the image.

### Inference (CLI)

```bash
# Single image inference
python scripts/inference_demo.py \
    --model_path outputs/opd/final \
    --image your_image.jpg \
    --prompt "Locate the person in the image."

# With 4-bit quantization (saves VRAM)
python scripts/inference_demo.py \
    --model_path outputs/opd/final \
    --image your_image.jpg \
    --prompt "Locate the person in the image." \
    --load_in_4bit

# Batch inference on JSONL
python scripts/inference_demo.py \
    --model_path outputs/opd/final \
    --jsonl data/sft/counting/counting_data.jsonl \
    --image_root data/coco/val \
    --max_samples 10
```

### Inference from Python

```python
import torch
from PIL import Image
from model import VisualPrimitiveVLM
from transformers import AutoProcessor

model = VisualPrimitiveVLM.from_pretrained("yunfengwang/TVP-OPD-Qwen2VL-2B", device_map="cuda")
model.eval()
tokenizer = model.tokenizer
processor = AutoProcessor.from_pretrained(
    model.base_model_path, trust_remote_code=True
)

image = Image.open("your_image.jpg").convert("RGB")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image", "image": "your_image.jpg"},
        {"type": "text", "text": "Locate the cat in the image."},
    ]},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

with torch.no_grad():
    output_ids = model.vlm.generate(**inputs, max_new_tokens=256, do_sample=False)

new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
response = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
print(response)
```

## Full Reproduction

### Step 1: Prepare Data

```bash
python scripts/prepare_all_data.py \
    --output_dir data \
    --coco_split val \
    --coco_subset 5000 \
    --num_counting 2000 \
    --num_spatial 2000 \
    --num_maze 5000 \
    --num_path 3000
```

This downloads COCO 2017 val (~1GB) and generates all training data:

```
data/
├── coco/val/images/                    # COCO images
├── pretrain/grounding.jsonl            # Pretrain grounding data (~14K)
├── sft/
│   ├── counting/counting_data.jsonl    # Counting with boxes (2K)
│   ├── spatial/                        # CLEVR-style spatial reasoning (2K)
│   ├── maze/                           # Procedural mazes (5K)
│   ├── path/                           # Path tracing (3K)
│   └── grounding/sft_grounding.jsonl   # Grounding with negatives (10K)
```

Then generate the SFT grounding data with negative samples:

```bash
python scripts/generate_sft_grounding_data.py \
    --coco_jsonl data/pretrain/grounding.jsonl \
    --image_root data/coco/val \
    --output data/sft/grounding/sft_grounding.jsonl \
    --neg_ratio 0.30
```

### Step 2: Pretraining

Teaches the base model to output visual primitive tokens (`<|box|>`, `<|point|>`, `<|ref|>`).

```bash
python pretraining/train_pretrain.py \
    --config configs/pretrain_12g.yaml \
    --output_dir outputs/pretrain
```

| Config | Value |
|--------|-------|
| Base model | Qwen/Qwen2-VL-2B-Instruct |
| LoRA | r=16, alpha=32 |
| Epochs | 3 |
| Effective batch | 2 × 8 = 16 |
| ~Time (12GB GPU) | ~1 hour |

### Step 3: Specialized SFT

Train two expert models from the pretrained checkpoint:

```bash
# Box expert (grounding, counting, spatial)
python sft/train_sft_box.py \
    --config configs/sft_box_12g.yaml \
    --output_dir outputs/sft_box

# Point expert (maze navigation, path tracing)
python sft/train_sft_point.py \
    --config configs/sft_point_12g.yaml \
    --output_dir outputs/sft_point
```

| Config | Box Expert | Point Expert |
|--------|-----------|-------------|
| Data | 10K grounding (7K pos + 3K neg) | Maze + Path |
| LoRA | r=64, alpha=128 | r=64, alpha=128 |
| Epochs | 5 | 5 |
| ~Time | ~2.5 hours | ~2.5 hours |

### Step 4: On-Policy Distillation (OPD)

Distill both experts into a single unified model using forward KL divergence with task-adaptive teacher routing:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python unified/train_opd.py \
    --config configs/opd_12g.yaml \
    --output_dir outputs/opd
```

| Config | Value |
|--------|-------|
| Student | outputs/pretrain/final |
| Teachers | sft_box/final + sft_point/final |
| Loss | Forward KL + CE (ce_coeff=0.5) |
| Temperature | 1.5 |
| Routing | Task-adaptive (box tasks→box teacher, point tasks→point teacher) |
| Epochs | 3 |
| ~Time | ~1.5 hours |

### Step 5: Evaluate

```bash
# Evaluate on counting
python evaluation/run_eval.py \
    --model_path outputs/opd/final \
    --task counting \
    --data_path data/sft/counting/counting_data.jsonl \
    --image_root data/coco/val \
    --output outputs/opd/eval_counting.json

# Evaluate on maze
python evaluation/run_eval.py \
    --model_path outputs/opd/final \
    --task maze \
    --data_path data/sft/maze/maze_data.jsonl \
    --output outputs/opd/eval_maze.json

# Visual comparison across all stages
python scripts/compare_models.py
```

Supported tasks: `counting`, `spatial`, `maze`, `path`, `all`

## Visual Primitives Format

Coordinates are normalized integers in `[0, 999]`:

```
# Bounding box
<|ref|>cat<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>

# Multiple boxes
<|ref|>person<|/ref|><|box|>[[130,50,400,800],[500,60,750,790]]<|/box|>

# Point
<|point|>[[x,y]]<|/point|>

# Point sequence (path/maze)
<|point|>[[100,200],[150,250],[200,300]]<|/point|>
```

## Project Structure

```
├── configs/                    # Training configs (*_12g.yaml for 12GB GPUs)
├── model/
│   ├── vl_model.py             # VisualPrimitiveVLM wrapper (PEFT, quantization)
│   ├── special_tokens.py       # Visual primitive token definitions
│   ├── spatial_compression.py  # 3×3 spatial compression module
│   └── vision_projector.py     # Vision-language projector
├── data/
│   ├── datasets_pretrain.py    # Pretrain dataset
│   ├── datasets_sft.py         # SFT dataset (JSONL-based)
│   ├── collators.py            # Conversation collator with assistant-only masking
│   └── transforms.py           # Image transforms
├── pretraining/
│   └── train_pretrain.py
├── sft/
│   ├── train_sft_box.py        # Box expert SFT
│   └── train_sft_point.py      # Point expert SFT
├── rl/
│   ├── grpo_trainer.py         # GRPO trainer
│   ├── reward_models.py        # Format/Quality/Accuracy reward models
│   ├── train_rl_box.py         # Box expert RL (optional)
│   └── train_rl_point.py       # Point expert RL (optional)
├── unified/
│   ├── train_opd.py            # On-Policy Distillation
│   ├── train_rft.py            # Rejection Fine-Tuning (optional)
│   └── generate_rft_data.py
├── evaluation/
│   ├── run_eval.py             # Unified evaluation entry point
│   └── metrics.py              # Task-specific metrics
├── scripts/
│   ├── prepare_all_data.py     # One-command data preparation
│   ├── compare_models.py       # Visual comparison across stages
│   ├── inference_demo.py       # Inference demo
│   ├── generate_maze_data.py   # Procedural maze generation
│   ├── generate_path_data.py   # Path tracing data generation
│   └── regenerate_data.py      # Regenerate data with correct normalization
└── utils/
    ├── visualization.py        # Draw boxes/points on images
    ├── coco_categories.py      # COCO category ID→name mapping
    ├── checkpoint.py           # Save/load checkpoints
    └── logging.py
```

## VRAM Guide

| GPU VRAM | Config suffix | Key settings |
|----------|---------------|-------------|
| 24GB | `*.yaml` | batch=2, LoRA r=128, image_size=448 |
| 16GB | — | batch=1, LoRA r=64, image_size=384 |
| **12GB** | `*_12g.yaml` | batch=1, LoRA r=64, image_size=336, max_length=1024 |

For 12GB GPUs, the collator pre-resizes images to 336×336 before the VL processor, capping visual tokens at ~576 (vs ~1500 at native resolution). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

For OPD (3 models loaded simultaneously), teachers use 4-bit quantization automatically.

## Training Pipeline Details

### Pretraining
- **Data**: COCO detection annotations → `<|ref|>label<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>`
- **Training**: LoRA on LLM, freeze ViT
- **Loss**: Standard cross-entropy (next-token prediction)

### Specialized SFT
Two experts with structured thinking templates:

| Expert | Tasks | Primitive | Thinking Template |
|--------|-------|-----------|-------------------|
| Box (FTwG) | Counting, Spatial, Grounding | `<\|box\|>` | Intent → Grounding → Conclusion |
| Point (FTwP) | Maze, Path Tracing | `<\|point\|>` | DFS exploration / waypoint sequence |

### On-Policy Distillation
- **Forward KL** with temperature scaling: `D_KL(teacher ‖ student)`
- **Task-adaptive routing**: each sample goes to its relevant expert teacher only
- **CE regularization**: prevents catastrophic forgetting (ce_coeff=0.5)

### Optional: RL with GRPO
The RL stage uses Group Relative Policy Optimization with three reward models:

| Reward Model | Type | Description |
|-------------|------|-------------|
| Format RM | Rule-based | Validates `<\|box\|>`/`<\|point\|>` syntax |
| Quality RM | Heuristic/LLM | Checks redundancy, self-contradiction |
| Accuracy RM | Task-specific | Counting: exponential error; Maze: multi-component; Path: bidirectional trajectory distance |

## Citation

If you find this repository useful, please cite our implementation and the original paper:

```bibtex
@software{wang2026tvp_pytorch,
  title={Thinking with Visual Primitives — PyTorch Implementation},
  author={Wang, Weishan},
  url={https://github.com/vra/Thinking-with-Visual-Primitives-pytorch},
  year={2026}
}

@article{lu2026thinking,
  title={Thinking with Visual Primitives},
  author={Lu, Ruijie and Ma, Yiyang and Chen, Xiaokang and others},
  journal={arXiv preprint arXiv:2506.00000},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
