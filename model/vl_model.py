"""
Unified Vision-Language Model for Thinking with Visual Primitives.

This model wraps an existing VLM backbone (e.g., Qwen2.5-VL, LLaVA) and:
1. Adds special tokens for visual primitives.
2. Optionally inserts spatial compression after the ViT.
3. Supports 4-bit quantization for low-VRAM training (optional).
4. Provides training and generation interfaces.

Architecture flow:
  Image -> ViT -> (SpatialCompression) -> Projector -> LLM -> Text + Visual Primitives
"""

from typing import Optional, Dict, List, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoConfig,
    BitsAndBytesConfig,
)

from .special_tokens import add_special_tokens, SPECIAL_TOKENS
from .spatial_compression import SpatialCompression
from .vision_projector import MLPProjector


def _maybe_bnb_config(torch_dtype, load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Build BitsAndBytesConfig if requested."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class VisualPrimitiveVLM(nn.Module):
    """
    A wrapper around a pretrained VLM that supports visual primitive tokens.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_spatial_compression: bool = False,
        compression_kernel: int = 3,
        freeze_vision_tower: bool = True,
        freeze_llm: bool = False,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.use_spatial_compression = use_spatial_compression

        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.tokenizer = add_special_tokens(self.tokenizer)

        # Load pretrained VLM
        torch_dtype = torch_dtype or torch.bfloat16
        bnb_config = _maybe_bnb_config(torch_dtype, load_in_4bit, load_in_8bit)

        load_kwargs = dict(
            pretrained_model_name_or_path=model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.vlm = AutoModelForVision2Seq.from_pretrained(**load_kwargs)

        # Resize token embeddings for new special tokens
        self.vlm.resize_token_embeddings(len(self.tokenizer))

        # Optionally freeze components
        if freeze_vision_tower:
            self._freeze_vision_tower()
        if freeze_llm:
            self._freeze_llm()

        # Optional spatial compression after ViT features
        self.spatial_compression = None
        if use_spatial_compression:
            vision_cfg = self._get_vision_config()
            in_ch = getattr(vision_cfg, "hidden_size", getattr(vision_cfg, "embed_dim", 768))
            self.spatial_compression = SpatialCompression(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=compression_kernel,
                mode="project",
            )

        # Enable gradient checkpointing for memory saving
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable()

    def _get_vision_config(self):
        config = self.vlm.config
        if hasattr(config, "vision_config"):
            return config.vision_config
        return config

    def _freeze_vision_tower(self):
        """Freeze the vision encoder parameters."""
        if hasattr(self.vlm, "vision_tower"):
            for p in self.vlm.vision_tower.parameters():
                p.requires_grad = False
        elif hasattr(self.vlm, "visual"):
            for p in self.vlm.visual.parameters():
                p.requires_grad = False

    def _freeze_llm(self):
        """Freeze the LLM backbone parameters."""
        if hasattr(self.vlm, "language_model"):
            for p in self.vlm.language_model.parameters():
                p.requires_grad = False
        elif hasattr(self.vlm, "model"):
            for p in self.vlm.model.parameters():
                p.requires_grad = False

    @property
    def config(self):
        return self.vlm.config

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass.
        For most pretrained VLMs, pixel_values and input_ids are handled internally.
        """
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """Generate text responses (including visual primitives)."""
        outputs = self.vlm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        # Decode, skipping the prompt tokens
        generated = outputs[:, input_ids.shape[1]:] if input_ids is not None else outputs
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
        return texts

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.vlm.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model. Special tokens are re-added automatically."""
        return cls(model_name_or_path=model_name_or_path, **kwargs)
