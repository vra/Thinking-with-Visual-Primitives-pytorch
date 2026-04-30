# Thinking with Visual Primitives — 复现训练框架

> 基于论文 [*Thinking with Visual Primitives*](https://arxiv.org/abs/2506.00000) 的训练代码复现。

本项目完整复现了论文中的**五阶段训练管线**：

```
Pretraining → Specialized SFT → Specialized RL (GRPO) → Unified RFT → On-Policy Distillation
```

核心创新点——将 **Bounding Box** 和 **Point** 提升为推理的"最小思想单元"（Visual Primitives），通过在 CoT 中交错空间坐标来解决 MLLM 的 **Reference Gap** 问题。

---

## 1. 项目特性

| 特性 | 说明 |
|------|------|
| **五阶段训练** | 完整复现 Pretrain → SFT → RL → RFT → OPD 全流程 |
| **Visual Primitives** | 支持 `<\|box\|>` 和 `<\|point\|>` 特殊 Token，坐标归一化到 `[0, 999]` |
| **GRPO 强化学习** | Group Relative Policy Optimization，无需价值网络，组内相对优势估计 |
| **多维度奖励模型** | Format RM (规则) + Quality RM (LLM/启发式) + Accuracy RM (任务-specific) |
| **程序生成数据** | Maze（矩形/圆形/六边形）、Path Tracing、CLEVR 风格 Spatial Reasoning 全部程序生成 |
| **24G 显存优化** | bf16 + LoRA(r=128) + Gradient Accumulation，单卡 24G 可训练 7B 模型 |
| **4-bit 量化支持** | 可选 `bitsandbytes` 4-bit 加载，适配 12G/16G 显存 |

---

## 2. 环境要求

- **GPU**: 24GB VRAM (RTX 3090/4090, A10, etc.)
- **CUDA**: ≥ 11.8
- **Python**: ≥ 3.9
- **OS**: Linux (推荐) / macOS (仅推理/调试)

```bash
# 1. 创建环境
conda create -n vprim python=3.10 -y
conda activate vprim

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

> **12G/16G 显存用户**: 在配置文件中开启 `load_in_4bit: true` 即可。

---

## 3. 快速开始

### 3.1 一键准备数据

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

执行后会生成：

```
data/
├── coco/val/images/              # COCO 2017 val 图片 (~1GB)
├── pretrain/grounding.jsonl      # 预训练 grounding 数据 (~40K 条)
├── sft/
│   ├── counting/counting_data.jsonl
│   ├── spatial/spatial_data.jsonl + images/
│   ├── maze/maze_data.jsonl + images/
│   └── path/path_data.jsonl + images/
```

> **关于数据规模**：
> - 预训练使用 COCO val 5K 图生成的 grounding 数据（论文用 40M+，复现用 ~40K）
> - SFT 冷启动数据全部程序生成，无需额外下载
> - 如需扩大规模，修改 `--coco_subset` 和 `--num_*` 参数

### 3.2 五阶段训练

```bash
# 阶段 1: 预训练 (学习输出 visual primitives)
python pretraining/train_pretrain.py --config configs/pretrain.yaml

# 阶段 2: 专家 SFT — Box 分支 (Counting + Spatial)
python sft/train_sft_box.py --config configs/sft_box.yaml

# 阶段 2: 专家 SFT — Point 分支 (Maze + Path)
python sft/train_sft_point.py --config configs/sft_point.yaml

# 阶段 3: 专家 RL — Box 分支 (GRPO)
python rl/train_rl_box.py --config configs/rl_box.yaml

# 阶段 3: 专家 RL — Point 分支 (GRPO)
python rl/train_rl_point.py --config configs/rl_point.yaml

# 阶段 4: 统一 RFT
python unified/generate_rft_data.py
python unified/train_rft.py --config configs/rft.yaml

# 阶段 5: On-Policy Distillation
python unified/train_opd.py --config configs/opd.yaml
```

### 3.3 评估

```bash
python evaluation/run_eval.py \
    --model_path outputs/opd/final \
    --task counting \
    --data_path data/sft/counting/counting_data.jsonl \
    --output results_counting.json
```

支持的任务: `counting`, `spatial`, `maze`, `path`, `all`

---

## 4. 训练流程详解

### 阶段 1: 预训练 (Pretraining)

**目标**: 让模型学会输出 visual primitives 的基本格式和定位能力。

- **数据**: COCO 检测数据，格式为 `<\|ref\|>cat<\|/ref\|><\|box\|>[[x1,y1,x2,y2]]<\|/box\|>`
- **训练**: 只训练 Vision Projector + LoRA adapter，冻结 ViT 和 LLM backbone
- **Loss**: 标准 next-token prediction (CrossEntropy)

### 阶段 2: 专家 SFT (Specialized SFT)

将模型分为两个专家：

| 专家 | 数据 | Visual Primitive |
|------|------|------------------|
| **FTwG** (Thinking with Grounding) | Counting + Spatial Reasoning | Box |
| **FTwP** (Thinking with Pointing) | Maze Navigation + Path Tracing | Point |

每个任务的 thinking content 遵循结构化模板：
- **Counting**: Intent Analysis → Batch Grounding → Statistical Summation
- **Spatial**: Intent Analysis → Object Grounding → Relational Inference
- **Maze**: DFS exploration trace with forward / backtracking
- **Path**: Start → Waypoints → End

### 阶段 3: 专家 RL (Specialized RL)

使用 **GRPO** (Group Relative Policy Optimization) 优化专家模型：

1. 对每个 prompt 采样 `group_size` 个 rollout
2. 用多维度 Reward Model 评分
3. 计算组内相对优势: `A_i = (R_i - mean(R)) / std(R)`
4. 策略更新 + KL 惩罚 against SFT reference model

**Reward Models**:

| RM | 类型 | 说明 |
|----|------|------|
| **Format RM** | 规则 | 检查 visual primitive 语法、重复 box |
| **Quality RM** | LLM/启发式 | 检查冗余、一致性、自相矛盾、reward hacking |
| **Accuracy RM** | 任务-specific | Counting: 平滑指数奖励; Maze: 多组件加权; Path: 双向轨迹距离 |

### 阶段 4: 统一 RFT (Unified RFT)

- 用 ETwG 和 ETwP 对数据池做 rollout
- 按难度分级：Easy (全对) / Normal (部分对) / Hard (全错)
- **只保留 Normal-Level + 5% Easy-Level**
- 从 **Base Pretrained Model** 初始化，重新 SFT 训练统一模型

### 阶段 5: 蒸馏 (On-Policy Distillation)

- **Loss**: `L_OPD = Σ w_i * D_KL(π_student || π_expert_i)`
- 全词表 logit 级蒸馏
- 教师: ETwG + ETwP
- 可选保留少量 CE loss 防止灾难性遗忘

---

## 5. 显存配置指南

### 24G VRAM (推荐，默认配置)

```yaml
batch_size: 2
gradient_accumulation_steps: 8  # 有效 batch = 16
load_in_4bit: false
torch_dtype: "bfloat16"
lora_r: 128
```

### 16G VRAM

```yaml
batch_size: 1
gradient_accumulation_steps: 16
load_in_4bit: false
torch_dtype: "bfloat16"
lora_r: 64
image_size: 384
```

### 12G VRAM

```yaml
batch_size: 1
gradient_accumulation_steps: 16
load_in_4bit: true          # 开启 4-bit 量化
lora_r: 64
image_size: 384
max_length: 1536
```

> 在 OPD 阶段（需同时加载 student + 2 teachers），12G 用户可能需要将 teachers 也量化为 4-bit，或分两阶段蒸馏。

---

## 6. 目录结构

```
thinking_with_visual_primitives/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── configs/                           # 训练配置
│   ├── pretrain.yaml
│   ├── sft_box.yaml
│   ├── sft_point.yaml
│   ├── rl_box.yaml
│   ├── rl_point.yaml
│   ├── rft.yaml
│   └── opd.yaml
├── model/                             # 模型架构
│   ├── vl_model.py                    # VLM 封装 (支持 4bit/8bit)
│   ├── special_tokens.py              # Visual Primitive 特殊 Token
│   ├── spatial_compression.py         # 3x3 空间压缩
│   └── vision_projector.py            # 投影层
├── data/                              # 数据管道
│   ├── formats.py                     # 统一数据格式定义
│   ├── datasets_pretrain.py           # 预训练数据集
│   ├── datasets_sft.py                # SFT 数据集
│   ├── collators.py                   # Batch 拼接
│   └── transforms.py                  # 图像预处理
├── pretraining/
│   └── train_pretrain.py              # 预训练脚本
├── sft/
│   ├── train_sft_box.py               # Box 专家 SFT
│   └── train_sft_point.py             # Point 专家 SFT
├── rl/
│   ├── grpo_trainer.py                # GRPO 训练器
│   ├── reward_models.py               # 多维度奖励模型
│   ├── train_rl_box.py                # Box 专家 RL
│   └── train_rl_point.py              # Point 专家 RL
├── unified/
│   ├── generate_rft_data.py           # RFT 数据生成
│   ├── train_rft.py                   # 统一 RFT 训练
│   └── train_opd.py                   # OPD 蒸馏
├── evaluation/
│   ├── metrics.py                     # 评估指标
│   └── run_eval.py                    # 统一评估入口
├── scripts/
│   ├── prepare_all_data.py            # 一键数据准备
│   ├── generate_maze_data.py          # 迷宫程序生成
│   └── generate_path_data.py          # 路径追踪程序生成
└── utils/
    ├── logging.py
    ├── checkpoint.py
    └── visualization.py               # 可视化 visual primitives
```

---

## 7. 核心模块说明

### Visual Primitives 格式

论文定义的标准格式：

```
<|ref|>TARGET<|/ref|><|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4],...]<|/box|>
<|point|>[[x1,y1],[x2,y2],...]<|/point|>
```

坐标归一化为 `[0, 999]` 的整数。

### Spatial Compression

```python
from model.spatial_compression import SpatialCompression
compress = SpatialCompression(in_channels=768, out_channels=768, kernel_size=3)
# Input:  (B, H, W, C)  e.g. (1, 54, 54, 768)
# Output: (B, H//3, W//3, C)  e.g. (1, 18, 18, 768)
```

### GRPO 训练器

```python
from rl.grpo_trainer import GRPOTrainer

trainer = GRPOTrainer(
    model=policy,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fns=[format_rm, quality_rm, accuracy_rm],
    group_size=4,
    kl_coeff=0.04,
)
stats = trainer.train_epoch(dataloader)
```

---

## 8. 训练监控

推荐使用 **WandB** 监控训练：

```bash
wandb login
# 在训练脚本中会自动记录 loss, lr, reward 等指标
```

或在配置中关闭：

```yaml
# 在 configs/*.yaml 中
report_to: "none"
```

---

## 9. 常见问题

**Q: COCO 数据下载失败？**
> `prepare_all_data.py` 使用 `datasets` 库的 streaming 模式下载，需要良好的网络连接。如失败可手动下载 COCO 2017 val 并放置到 `data/coco/val/images/`，然后修改脚本中的路径。

**Q: 能否用其他 VLM 骨干？**
> 可以。修改 `configs/*.yaml` 中的 `model_name_or_path` 即可。已测试兼容:
> - `Qwen/Qwen2.5-VL-7B-Instruct` (推荐)
> - `Qwen/Qwen2-VL-7B-Instruct`
> - `llava-hf/llava-1.5-7b-hf`

**Q: 训练时 CUDA OOM？**
> 1. 减小 `batch_size` 到 1
> 2. 增大 `gradient_accumulation_steps`
> 3. 减小 `image_size` 到 384
> 4. 减小 `max_length`
> 5. 开启 `load_in_4bit: true`

**Q: 如何只训练某个阶段？**
> 每个阶段独立，可从任意 checkpoint 恢复。修改配置中的 `model_name_or_path` 指向上阶段的输出目录即可。

---

## 10. 引用

如果本项目对你的研究有帮助，请引用原论文及本代码仓库：

```bibtex
@article{lu2026thinking,
  title={Thinking with Visual Primitives},
  author={Lu, Ruijie and Ma, Yiyang and Chen, Xiaokang and others},
  journal={arXiv preprint arXiv:2506.00000},
  year={2026}
}
```

同时也请引用本 PyTorch 复现仓库：

```bibtex
@misc{thinking2026pytorch,
  title={Thinking with Visual Primitives — PyTorch Implementation},
  author={Wang, Yunfeng},
  howpublished={\url{https://github.com/vra/Thinking-with-Visual-Primitives-pytorch}},
  year={2026}
}
```

---

## 11. License

本项目代码遵循 MIT License。原论文版权归原作者所有。
