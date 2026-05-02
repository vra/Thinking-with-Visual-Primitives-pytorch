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
| **12G 显存支持** | 预 resize 图片 + max_length 控制，RTX 4070 Ti 12G 可全精度训练 2B 模型 |
| **4-bit 量化支持** | 可选 `bitsandbytes` 4-bit 加载（预训练可用，SFT 建议全精度） |

---

## 2. 环境要求

- **GPU**: 12GB+ VRAM (已验证 RTX 4070 Ti 12G)
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

> **12G 显存用户**: 本项目已针对 12G 显存优化，无需 4-bit 量化即可全精度训练。关键优化点：
> - Collator 中预 resize 图片到 336×336（避免 Qwen2-VL processor 保留原始尺寸产生 1500+ patches）
> - max_length 控制在 1024，限制每样本最多 8 个 box
> - 详见 `configs/sft_box_12g.yaml`

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
│   ├── path/path_data.jsonl + images/
│   └── grounding/sft_grounding.jsonl   # 带负样本的 grounding SFT 数据
```

> **关于数据规模**：
> - 预训练使用 COCO val 5K 图生成的 grounding 数据（论文用 40M+，复现用 ~40K）
> - SFT grounding 数据包含 10K 条（7K 正样本 + 3K 负样本），用于提升定位精度和拒绝能力
> - 如需扩大规模，修改 `--coco_subset` 和 `--num_*` 参数

#### 生成 SFT Grounding 数据（含负样本）

```bash
python scripts/generate_sft_grounding_data.py \
    --coco_jsonl data/pretrain/grounding.jsonl \
    --image_root data/coco/val \
    --output data/sft/grounding/sft_grounding.jsonl \
    --neg_ratio 0.30 \
    --max_samples 10000
```

### 3.2 五阶段训练

```bash
# 阶段 1: 预训练 (学习输出 visual primitives)
python pretraining/train_pretrain.py --config configs/pretrain_12g.yaml

# 阶段 2: 专家 SFT — Grounding Box 分支 (从预训练 final 继续，带负样本)
python sft/train_sft_box.py --config configs/sft_box_12g.yaml --output_dir outputs/sft_box

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

#### 训练监控

```bash
# 预训练监控（自动检测崩溃并恢复）
bash scripts/monitor_pretrain.sh

# SFT 监控
bash scripts/monitor_sft.sh
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

#### 分层评估 (Tiered Evaluation)

```bash
# Easy / Medium / Hard / Negative 四级评估
python scripts/eval_tiered.py \
    --model_path outputs/pretrain/final \
    --coco_jsonl data/pretrain/grounding.jsonl \
    --image_root data/coco/val \
    --output_dir outputs/tiered_eval

# 三 Epoch 对比 (epoch_0 vs epoch_1 vs final)
python scripts/compare_three_epochs.py \
    --num_samples 5 \
    --output_dir outputs/three_epoch_comparison

# 两 Epoch 对比
python scripts/compare_epoch_progress.py \
    --epoch_a outputs/pretrain/epoch_0 \
    --epoch_b outputs/pretrain/epoch_1 \
    --num_samples 5 \
    --output_dir outputs/epoch_comparison
```

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

### 12G VRAM (已验证 RTX 4070 Ti)

```yaml
batch_size: 1
gradient_accumulation_steps: 16
load_in_4bit: false        # 全精度 bfloat16，无需 4-bit
lora_r: 64
lora_alpha: 128
image_size: 336             # 预 resize 到 336x336，控制 visual tokens ≈ 576
max_length: 1024            # 限制序列长度
torch_dtype: "bfloat16"
```

**关键优化原理**：
1. **预 resize 图片**：Qwen2-VL processor 默认 `max_pixels=12845056`，会保留原始图片尺寸。640×480 图片产生 **1564 patches**。Collator 中预先 resize 到 336×336 后仅 **576 patches**，节省 ~3GB VRAM。
2. **限制 box 数量**：每样本最多 8 个 box，避免 thinking 模板过长。
3. **expandable_segments**：设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少内存碎片。

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
│   ├── generate_sft_grounding_data.py # SFT grounding 数据生成（含负样本）
│   ├── generate_maze_data.py          # 迷宫程序生成
│   ├── generate_path_data.py          # 路径追踪程序生成
│   ├── eval_tiered.py                 # 四级评估 (Easy/Medium/Hard/Negative)
│   ├── compare_three_epochs.py        # 三 Epoch 对比
│   ├── compare_epoch_progress.py      # 两 Epoch 对比
│   ├── monitor_pretrain.sh            # 预训练监控（自动恢复）
│   └── monitor_sft.sh                 # SFT 监控
├── utils/
│   ├── logging.py
│   ├── checkpoint.py
│   ├── visualization.py               # 可视化 visual primitives
│   └── coco_categories.py             # COCO 类别映射
    ├── logging.py
    ├── checkpoint.py
    └── visualization.py               # 可视化 visual primitives
```

---

## 7. 训练记录

### 阶段 1: 预训练 (Pretraining)

| 配置 | 值 |
|------|-----|
| 模型 | Qwen/Qwen2-VL-2B-Instruct |
| 数据 | COCO val 5K 生成的 ~40K grounding 样本 |
| LoRA | r=16, alpha=32, target=q/k/v/o/gate/up/down_proj |
| 精度 | 4-bit NF4 + bfloat16 compute |
| Epochs | 3 |
| Batch | 1 × 16 grad_accum = 16 effective |
| Loss | 0.827 → 0.688 → 0.654 |
| GPU | RTX 4070 Ti 12G |

**Tiered 评估结果** (60 样本，15/级):

| Tier | Avg IoU | 说明 |
|------|---------|------|
| Easy | 0.347 | 大单物体定位较好 (bear/cat IoU >0.97) |
| Medium | 0.275 | 多物体中等难度 |
| Hard | 0.125 | 小物体/密集场景 |
| Negative | 1/15 | 几乎不会拒绝，因预训练无负样本 |

**结论**: 3 epoch 足够，final (epoch_2) 最佳。继续预训练收益递减。

### 阶段 2: SFT Grounding (进行中)

| 配置 | 值 |
|------|-----|
| 基模型 | outputs/pretrain/final (adapter) |
| 数据 | 10K 样本 (7K 正样本 + 3K 负样本) |
| LoRA | r=64, alpha=128 |
| 精度 | bfloat16 全精度 |
| Epochs | 5 |
| Batch | 1 × 16 grad_accum = 16 effective |
| 优化 | 图片预 resize 336×336, max_length=1024, 每样本≤8 box |
| GPU | RTX 4070 Ti 12G (~8.7GB / 12GB) |

**目标**: 注入负样本学习拒绝能力，提升定位精度。

---

## 8. 核心模块说明

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

## 9. 训练监控

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

## 10. 常见问题

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
> 3. 减小 `image_size` 到 336（collator 会预 resize 图片）
> 4. 减小 `max_length` 到 1024
> 5. 限制每样本最大 box 数量（`scripts/generate_sft_grounding_data.py` 中 `MAX_BOXES_PER_SAMPLE`）
> 6. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
> 7. 最后手段：开启 `load_in_4bit: true`（注意：4-bit 下验证可能不稳定）

**Q: 如何只训练某个阶段？**
> 每个阶段独立，可从任意 checkpoint 恢复。修改配置中的 `model_name_or_path` 指向上阶段的输出目录即可。

---

## 11. 引用

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

## 12. License

本项目代码遵循 MIT License。原论文版权归原作者所有。
