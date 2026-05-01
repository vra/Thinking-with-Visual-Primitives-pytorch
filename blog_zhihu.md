# 在 12GB 显存上复现 "Thinking with Visual Primitives"：从训练到推理的完整踩坑记录

> 本文记录了我基于 PyTorch 从零复现 [Thinking with Visual Primitives](https://github.com/Thinking-with-Visual-Primitives) 的全过程。使用 **Qwen2.5-VL-7B-Instruct** 作为基座模型，在一张 **RTX 4070 Ti (12GB VRAM)** 上完成预训练、SFT、RL 全流程。文中会详细记录架构设计、数据构造、训练调参，以及 Qwen2.5-VL 在 4-bit 量化 + LoRA 场景下的各种深坑和解决方案。

---

## 一、项目概述

### 1.1 什么是 Visual Primitives？

大型视觉语言模型（VLM）在视觉推理任务上表现出色，但传统的 Chain-of-Thought (CoT) 方法让模型用**自然语言**描述视觉信息，这存在两个根本问题：

1. **信息损失**：用 "左上角有一只猫" 来描述一个 bounding box，远不如直接输出坐标 `[x1, y1, x2, y2]` 精确。
2. **推理效率低**：自然语言描述冗长，模型需要生成大量 token 才能表达一个简单的空间关系。

**Visual Primitives** 的核心思想是：让 VLM 在推理过程中直接输出结构化的视觉基元（bounding box、point、mask），而不是用自然语言描述它们。具体来说，模型会输出类似这样的序列：

```
The image contains <|box_start|><|box_100|><|box_250|><|box_400|><|box_350|><|box_end|>.
```

其中 `<|box_start|>`、`<|box_end|>` 是特殊 token，中间的 `<|box_xxx|>` 是坐标 token（类似"视觉词表"），每个 token 对应一个离散化的坐标值。

### 1.2 训练流程

原论文采用四阶段训练策略：

```
Stage 1: Pretraining      →  大规模 grounding 数据，学习输出 primitives 的格式
Stage 2: SFT              →  任务特定数据，学习具体任务的输入输出模式
Stage 3: RL (GRPO)        →  强化学习，优化可验证的奖励信号
Stage 4: RFT + OPD        →  拒绝采样微调 + 在线偏好蒸馏
```

我目前完成了 **Stage 1 Pretraining**（3 epochs），正在推进后续阶段。

---

## 二、硬件与软件环境

| 配置项 | 详情 |
|--------|------|
| GPU | RTX 4070 Ti (12GB VRAM) |
| CUDA | 12.1 |
| Python | 3.9 |
| PyTorch | 2.3.1+cu121 |
| Transformers | 4.57.6 |
| PEFT | 0.12.0 |
| bitsandbytes | 0.48.2 |

**关键约束**：12GB 显存意味着无法直接加载 Qwen2.5-VL-7B（BF16 下约 16GB）。必须采用 **4-bit 量化 + LoRA** 才能在单卡上训练和推理。

---

## 三、模型架构设计

### 3.1 整体架构

```
VisualPrimitiveVLM
├── Tokenizer (Qwen2.5-VL)
│   └── 新增 6 个特殊 token: <|box_start|>, <|box_end|>, <|point_start|>, <|point_end|>, <|mask_start|>, <|mask_end|>
├── Vision Encoder (Qwen2.5-VL 自带)
│   └── 冻结 (freeze_vision_tower=True)
├── LLM (Qwen2.5-VL-7B)
│   ├── 4-bit 量化 (bnb_4bit)
│   ├── 词表扩展至 152064 (添加 special tokens)
│   └── LoRA 微调 (r=16, target_modules=全部 linear)
└── 可选: SpatialCompression 模块
```

### 3.2 核心代码：VisualPrimitiveVLM

```python
class VisualPrimitiveVLM(nn.Module):
    def __init__(self, model_name_or_path, ..., load_in_4bit=False):
        super().__init__()
        # 1. 加载 tokenizer 并添加特殊 token
        self.tokenizer = AutoTokenizer.from_pretrained(...)
        self.tokenizer = add_special_tokens(self.tokenizer)  # +6 tokens
        
        # 2. 配置 4-bit 量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # 3. 加载模型
        self.vlm = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 4. 关键：扩展词表！
        self.vlm.resize_token_embeddings(len(self.tokenizer))
```

**注意**：`resize_token_embeddings` 必须在 `from_pretrained` 之后立即调用。如果顺序错了，保存的 adapter 权重和加载时的模型维度会对不上。

### 3.3 LoRA 配置

由于 12GB 显存极其紧张，我采用了非常激进的配置：

```yaml
use_lora: true
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

训练参数量：**47.6M / 8.34B = 0.57%**，这是能在 12GB 上跑起来的极限。

---

## 四、数据构造

### 4.1 预训练数据 (Stage 1)

预训练的目标是让模型学会"如何输出 visual primitives 的格式"。数据格式为 JSONL：

```json
{
  "image": "COCO_val2014_000000000139.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nLocate the cat in the image."},
    {"from": "gpt", "value": "The cat is at <|box_start|><|box_100|><|box_200|><|box_300|><|box_250|><|box_end|>."}
  ]
}
```

数据来源：
- **COCO val2014 5K 图像**：提供真实场景图片
- **Grounding 标注**：转换为上述对话格式

总计：**14,631 条样本**

### 4.2 合成数据 (Stage 2+)

为了支持后续阶段的 counting、spatial reasoning、maze solving、path finding 任务，我构造了合成数据：

| 任务 | 样本数 | 说明 |
|------|--------|------|
| Counting | 2,000 | 图像中物体的计数 |
| Spatial | 2,000 | 空间关系推理（上下左右） |
| Maze | 5,000 | 迷宫求解，输出路径点 |
| Path | 3,000 | 路径规划，输出关键点 |

合成数据的构造脚本在 `scripts/generate_*_data.py` 中，使用 Pillow 动态生成图像。

### 4.3 Collator 的设计

Qwen2.5-VL 的 processor 非常特殊：
- `text` 参数接收**纯文本字符串**（不是 tokenized input_ids）
- `images` 参数接收 PIL Image 列表
- 返回值包含 `pixel_values`、`input_ids`、`attention_mask`，以及关键的 **`image_grid_thw`**

```python
class ConversationCollator:
    def __call__(self, batch_samples):
        texts = []
        images = []
        for sample in batch_samples:
            # 将对话格式化为 Qwen2.5-VL 的 chat template
            text = self.tokenizer.apply_chat_template(
                sample["conversations"], 
                tokenize=False, 
                add_generation_prompt=True
            )
            texts.append(text)
            images.append(Image.open(sample["image"]).convert("RGB"))
        
        # 使用 Qwen2.5-VL processor
        inputs = self.processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        return inputs  # 包含 pixel_values, input_ids, attention_mask, image_grid_thw
```

---

## 五、训练过程与踩坑记录

### 5.1 显存优化的极限配置

12GB VRAM 下，我进行了大量实验才找到稳定的配置：

```yaml
# configs/pretrain_12g.yaml
batch_size: 1                    # 只能为 1
gradient_accumulation_steps: 16   # 模拟 batch_size=16
image_size: 224                   # 默认 448 会 OOM
max_length: 512                   # 默认 2048 会 OOM
load_in_4bit: true
lora_r: 16
```

显存占用峰值：**~10.8GB / 12GB**

### 5.2 Bug #1：bitsandbytes 版本问题

**现象**：
```
RuntimeError: bnb_4bit_quantize failed: no kernel image is available
```

**原因**：`bitsandbytes 0.42.0` 不支持 CUDA 12.1 的某些操作，且 `.to(device)` 在 4-bit 参数上会报错。

**解决**：升级到 `bitsandbytes >= 0.43.2`（最终使用 0.48.2）。

```bash
pip install -U bitsandbytes
```

### 5.3 Bug #2：Gradient Checkpointing 与 4-bit 不兼容

**现象**：
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**原因**：Qwen2.5-VL 的 vision encoder 输出 `pixel_values` 不需要梯度。当开启 `gradient_checkpointing` 时，4-bit 模型要求所有输入都有 `requires_grad=True`。

**解决**：在 `forward()` 中强制设置 `pixel_values.requires_grad_(True)`：

```python
def forward(self, pixel_values=None, input_ids=None, ...):
    if self.training and pixel_values is not None and not pixel_values.requires_grad:
        pixel_values = pixel_values.requires_grad_(True)
    return self.vlm(pixel_values=pixel_values, ...)
```

或者更简单的方案：**禁用 gradient checkpointing**（4-bit 量化本身已经大幅降低了激活内存）。

### 5.4 Bug #3：Collator truncation 导致 image token 数量不匹配

**现象**：
```
ValueError: Image tokens do not match the number of images
```

**原因**：当开启 `truncation=True` 时，processor 可能会截断文本，导致 `<|image_pad|>` token 数量与实际图像 patch 数量不匹配。

**解决**：捕获异常后回退到无 truncation 模式：

```python
try:
    inputs = self.processor(..., truncation=True, max_length=self.max_length)
except ValueError:
    inputs = self.processor(..., padding=True)  # 无 truncation
```

### 5.5 Bug #4：Vocab Size 不匹配

**现象**：加载 checkpoint 时报错：
```
RuntimeError: Error(s) in loading state_dict for PeftModel:
size mismatch for base_model.model.model.embed_tokens.weight: 
copying a param with shape torch.Size([151671, 4096]) 
from checkpoint, the shape in current model is torch.Size([152064, 4096]).
```

**原因分析**：
- Qwen2.5-VL 原始词表：151,665
- `add_special_tokens` 后：151,671（+6 个特殊 token）
- 但 `len(tokenizer)` 返回 **152,064**，因为 tokenizer 内部有 padding 到 64 的倍数
- 保存 adapter 时：embeddings 维度是 151,671（当时 resize 的大小）
- 加载时：新模型 resize 到 152,064，与保存的 151,671 不匹配

**解决**：确保 `resize_token_embeddings` 始终使用 `len(self.tokenizer)`，并且在保存和加载时使用一致的路径：

```python
# 保存时
self.vlm.resize_token_embeddings(len(self.tokenizer))  # 152064

# 加载时（from_pretrained）
base_model.resize_token_embeddings(len(self.tokenizer))  # 同样 152064
self.vlm = PeftModel.from_pretrained(base_model, adapter_path)
```

### 5.6 Bug #5：推理时 image_grid_thw 未传递

**现象**：
```
TypeError: forward() got an unexpected keyword argument 'image_grid_thw'
```
或
```
RuntimeError: image_grid_thw is None
```

**原因**：Qwen2.5-VL 的 `generate()` 方法需要 `image_grid_thw` 来计算旋转位置编码（rotary position embedding for vision patches）。Processor 会返回这个值，但如果直接调用 `model.generate(**inputs)` 时 `inputs` 字典中没有包含它，就会报错。

**解决**：确保 processor 返回的所有 key 都传递给 model：

```python
inputs = processor(text=[text], images=[img], return_tensors="pt")
inputs = {k: v.to(model.vlm.device) for k, v in inputs.items()}
output_ids = model.vlm.generate(**inputs, max_new_tokens=512)
```

**注意**：不要用 `model.generate()`（自定义的包装方法），而要直接用 `model.vlm.generate()`，因为 Qwen2.5-VL 的 `generate` 内部会处理 `image_grid_thw`。

### 5.7 Bug #6：PEFT Adapter 恢复训练

**现象**：`--resume outputs/pretrain/epoch_0` 报错：
```
IsADirectoryError: [Errno 21] Is a directory
```

**原因**：`save_pretrained` 保存的是 PEFT adapter 目录，但 `load_checkpoint` 期望 `.pt` 文件。

**解决**：修改 resume 逻辑：

```python
if resume_path.is_dir() and (resume_path / "adapter_config.json").exists():
    # 从 PEFT adapter 目录恢复
    base_model = model.vlm.model if isinstance(model.vlm, PeftModel) else model.vlm
    model.vlm = PeftModel.from_pretrained(base_model, str(resume_path))
    start_epoch = int(resume_path.name.split("_")[-1]) + 1
```

---

## 六、训练结果

### 6.1 Stage 1 Pretraining

| Epoch | Avg Loss | 说明 |
|-------|----------|------|
| 0 | 15.6350 | 初始训练，模型尚未学会格式 |
| 1 | ~12.5 | 训练中（已恢复） |
| 2 | TBD | 预计进一步下降 |

训练速度：约 **1.03 it/s**，每个 epoch 约 **4 小时**。

### 6.2 推理 Demo

使用 epoch_0 的 checkpoint 进行推理测试：

```python
from model import VisualPrimitiveVLM
from transformers import AutoProcessor
from PIL import Image

model = VisualPrimitiveVLM.from_pretrained(
    "outputs/pretrain/epoch_0", 
    device_map="cuda", 
    load_in_4bit=True
)
model.eval()

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    trust_remote_code=True
)

img = Image.open("data/coco/val/images/000000000139.jpg")
conv = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image", "image": "test.jpg"},
        {"type": "text", "text": "Locate the cat."}
    ]},
]
text = model.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[img], return_tensors="pt")
inputs = {k: v.to(model.vlm.device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.vlm.generate(**inputs, max_new_tokens=64)
    response = model.tokenizer.decode(out[0], skip_special_tokens=False)
```

**当前输出**：epoch_0 的模型输出仍是随机文本，预计 3 个 epoch 后能输出正确的 box token 序列。

---

## 七、关键代码文件索引

| 文件 | 说明 |
|------|------|
| `model/vl_model.py` | VisualPrimitiveVLM 主模型 |
| `model/special_tokens.py` | 6 个特殊 token 的定义与添加 |
| `data/collators.py` | Qwen2.5-VL processor 的 batch collator |
| `data/datasets_pretrain.py` | JSONL / COCO grounding 数据集 |
| `pretraining/train_pretrain.py` | Stage 1 预训练脚本 |
| `sft/train_sft_*.py` | Stage 2 SFT 脚本 (box/point) |
| `rl/train_rl_*.py` | Stage 3 GRPO 强化学习 |
| `scripts/inference_demo.py` | 推理测试脚本 |
| `configs/*_12g.yaml` | 12GB VRAM 配置文件 |

---

## 八、经验总结

### 8.1 12GB VRAM 训练大模型的关键

1. **4-bit 量化是必须的**：NF4 + double quantization 可以将 7B 模型压缩到 ~4GB。
2. **LoRA r=16 是极限**：再大就会 OOM。如果预算允许，r=32 或 r=64 效果会更好。
3. **image_size=224**：Qwen2.5-VL 默认 448，但 224 在绝大多数任务上足够。
4. **max_length=512**：预训练时不需要太长的序列。
5. **batch_size=1 + grad_accum=16**：模拟大 batch，同时控制显存。

### 8.2 Qwen2.5-VL 的特殊性

相比 LLaVA、InternVL 等模型，Qwen2.5-VL 有几个需要特别注意的地方：

1. **Processor 返回 `image_grid_thw`**：这是计算 vision RoPE 的必需输入，推理时必须传递。
2. **`pixel_values` 是扁平化的**：不是标准的 `[B, C, H, W]`，而是 `[total_patches, C, patch_size, patch_size]`，内部通过 `image_grid_thw` reshape。
3. **`trust_remote_code=True`**：很多类不在 transformers 主仓库中，必须从 HuggingFace 加载。
4. **Vocab size 是 64 的倍数**：`resize_token_embeddings` 会自动对齐到 64 的倍数。

### 8.3 PEFT + 量化模型的最佳实践

1. **先 resize，再 wrap PEFT**：顺序不能错。
2. **保存用 `save_pretrained`**：会保存 adapter + tokenizer + 配置。
3. **恢复时 unwrap 再 reload**：不能直接 `load_state_dict`。
4. **bitsandbytes >= 0.43.2**：旧版本有各种 CUDA kernel 问题。

---

## 九、后续计划

- [ ] 完成 Stage 1 Pretraining（剩余 2 个 epoch）
- [ ] Stage 2 SFT：Counting / Spatial / Maze / Path 任务
- [ ] Stage 3 RL：GRPO 训练，设计 reward model
- [ ] Stage 4 RFT + OPD：拒绝采样与在线偏好蒸馏
- [ ] 完整 evaluation：在 COCO、RefCOCO 等 benchmark 上测试

---

## 十、参考链接

- 原论文：[Thinking with Visual Primitives](https://www.k-a.in/Thinking_with_Visual_Primitives.pdf) (DeepSeek-AI, PKU, THU)
- 基座模型：[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- PEFT：[huggingface/peft](https://github.com/huggingface/peft)
- bitsandbytes：[TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

> **项目地址**：https://github.com/vra/Thinking-with-Visual-Primitives-pytorch
> 
> 如果你也在做 VLM 的 visual reasoning 或者 4-bit 量化训练，欢迎在评论区交流！
