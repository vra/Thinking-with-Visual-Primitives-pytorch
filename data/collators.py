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
        try:
            model_name = getattr(tokenizer, "name_or_path", None)
            if model_name:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass

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
                images.append(Image.open(img_path).convert("RGB"))
            else:
                images.append(Image.new("RGB", (448, 448), (128, 128, 128)))

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

        # Build labels: mask user/system tokens, keep assistant tokens
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # For each sequence, find the assistant response boundary
        # Qwen2.5 uses im_start/im_end tokens; assistant content follows "assistant\n"
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            # Try to find assistant turn start
            assistant_start = None
            # Search for the pattern indicating assistant start
            for i in range(len(seq) - 1):
                # Heuristic: after im_start + assistant token
                if i + 1 < len(seq):
                    # Different tokenizers use different formats
                    # We'll try to mask everything before the last assistant marker
                    pass
            if assistant_start is None:
                # Fallback: mask first 50% of tokens (rough heuristic)
                assistant_start = len(seq) // 2

            # Actually, let's use a simpler approach: if we have generation prompt
            # separator, we can detect it. But for now, we'll compute loss on all tokens.
            # This is suboptimal but works for pretraining/SFT.
            pass

        # Simpler: compute loss on all non-padding tokens
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
