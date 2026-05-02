"""Collators for batching conversation data."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch
from transformers import AutoProcessor


class ConversationCollator:
    """
    Collates conversation samples for training.
    Supports Qwen2.5-VL style image + text conversations.
    """

    def __init__(self, tokenizer, image_transform=None, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        # Try to load processor for the model
        self.processor = None
        processor_candidates = []
        if hasattr(tokenizer, "base_model_path"):
            processor_candidates.append(tokenizer.base_model_path)
        if getattr(tokenizer, "name_or_path", None):
            processor_candidates.append(tokenizer.name_or_path)
        for model_name in processor_candidates:
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                break
            except Exception:
                continue

    def _build_conversation(self, sample: dict) -> List[dict]:
        """Build OpenAI-style conversation from sample."""
        messages = []
        # System prompt
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant that can understand images and reason with visual primitives."
        })

        image_path = sample.get("image_path")
        question = sample.get("question", "")
        thinking = sample.get("thinking", "")
        answer = sample.get("answer", "")
        label = sample.get("label", "")
        boxes = sample.get("boxes", [])
        points = sample.get("points", [])

        # Build user content
        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": str(image_path)})

        # Use question if available, otherwise build from label/boxes for pretrain
        if question:
            user_content.append({"type": "text", "text": question})
        elif label:
            user_content.append({"type": "text", "text": f"Locate the {label} in the image."})
        else:
            user_content.append({"type": "text", "text": "Describe what you see."})

        messages.append({"role": "user", "content": user_content})

        # Build assistant content
        assistant_text = ""
        if thinking:
            assistant_text += thinking + "\n"
        if answer:
            assistant_text += answer
        if not assistant_text and boxes:
            from model.special_tokens import format_box_token
            assistant_text = format_box_token(boxes)
        if not assistant_text and points:
            from model.special_tokens import format_point_token
            assistant_text = format_point_token(points)
        if not assistant_text:
            assistant_text = "I see the object."

        messages.append({"role": "assistant", "content": assistant_text.strip()})
        return messages

    def __call__(self, batch_samples: List[dict]) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            return self._call_with_processor(batch_samples)
        return self._call_simple(batch_samples)

    def _call_with_processor(self, batch_samples: List[dict]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        metadata_list = []

        for sample in batch_samples:
            conv = self._build_conversation(sample)
            try:
                text = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback for processors without apply_chat_template
                text = self._fallback_format(conv)
            texts.append(text)
            metadata_list.append(sample.get("metadata", {}))

            img_path = sample.get("image_path")
            if img_path and Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
                # Resize to fixed size BEFORE processor to control visual token count.
                # Qwen2-VL processor uses max_pixels=12845056 by default, which keeps
                # original image size and can produce 1500+ patches for 640x480 images.
                # Pre-resizing to 336x336 caps patches at 576, saving ~3GB VRAM.
                img = img.resize((336, 336), Image.LANCZOS)
                images.append(img)
            else:
                images.append(Image.new("RGB", (336, 336), (128, 128, 128)))

        # Processor handles images and text together
        try:
            inputs = self.processor(
                text=texts,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        except ValueError as e:
            if "Mismatch" in str(e) and "token" in str(e):
                # Fallback: no truncation to avoid breaking image tokens
                inputs = self.processor(
                    text=texts,
                    images=images if images else None,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                raise

        # Build labels: mask user/system tokens, keep only assistant tokens for loss
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Qwen2.5-VL uses <|im_start|> and <|im_end|> to delimit turns.
        # Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        # We mask everything except the assistant turn content.
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Encode "assistant" to find the role token(s)
        assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)

        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            seq_len = len(seq)
            # Default: mask everything
            mask = [True] * seq_len  # True = masked (set to -100)

            # Find all assistant turns and unmask them
            i = 0
            while i < seq_len:
                # Look for <|im_start|>
                if seq[i] == im_start_id:
                    # Check if this is an assistant turn
                    # The role tokens follow immediately after <|im_start|>
                    role_end = min(i + 1 + len(assistant_token_ids), seq_len)
                    role_tokens = seq[i + 1:role_end]
                    if role_tokens == assistant_token_ids:
                        # Find the newline after "assistant\n" — content starts after it
                        content_start = role_end
                        # Skip past the newline
                        while content_start < seq_len and seq[content_start] in [
                            self.tokenizer.convert_tokens_to_ids("\n"),
                            10,  # common newline token id
                        ]:
                            content_start += 1
                        # Find the matching <|im_end|>
                        content_end = content_start
                        while content_end < seq_len and seq[content_end] != im_end_id:
                            content_end += 1
                        # Unmask assistant content (including <|im_end|> for learning to stop)
                        for j in range(content_start, min(content_end + 1, seq_len)):
                            mask[j] = False
                i += 1

            # Apply mask
            for j in range(seq_len):
                if mask[j]:
                    labels[b, j] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "metadata": metadata_list,
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result

    def _call_simple(self, batch_samples: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        metadata_list = []

        for sample in batch_samples:
            text = sample.get("text", "")
            if not text:
                question = sample.get("question", "")
                thinking = sample.get("thinking", "")
                answer = sample.get("answer", "")
                text = f"User: {question}\nAssistant: {thinking}\n{answer}"

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])
            labels_list.append(encoded["input_ids"].copy())
            metadata_list.append(sample.get("metadata", {}))

            img_path = sample.get("image_path")
            if img_path and Path(img_path).exists() and self.image_transform:
                img = Image.open(img_path).convert("RGB")
                pixel_values_list.append(self.image_transform(img))
            else:
                pixel_values_list.append(torch.zeros(3, 448, 448))

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for ids, mask, labs in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(labs + [-100] * pad_len)

        result = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "metadata": metadata_list,
        }
        if pixel_values_list:
            result["pixel_values"] = torch.stack(pixel_values_list)
        return result

    def _fallback_format(self, conv: List[dict]) -> str:
        parts = []
        for msg in conv:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [c["text"] for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)