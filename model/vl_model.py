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

from pathlib import Path
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
            device_map=device_map,
            trust_remote_code=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        # Check if loading a PEFT adapter directory
        adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
        if adapter_config_path.exists():
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_path = peft_config.base_model_name_or_path
            # Load base model, resize embeddings, then load adapter
            self.vlm = AutoModelForVision2Seq.from_pretrained(base_model_path, **load_kwargs)
            self.vlm.resize_token_embeddings(len(self.tokenizer))
            # is_trainable=True ensures the loaded adapter weights are unfrozen
            # so downstream SFT can continue updating them instead of adding a
            # second LoRA that starts from zero and overwrites the pretrained
            # knowledge.
            self.vlm.load_adapter(model_name_or_path, is_trainable=True)
            self.base_model_path = base_model_path
            self.tokenizer.base_model_path = base_model_path
        else:
            load_kwargs["pretrained_model_name_or_path"] = model_name_or_path
            self.vlm = AutoModelForVision2Seq.from_pretrained(**load_kwargs)
            # Resize token embeddings for new special tokens
            self.vlm.resize_token_embeddings(len(self.tokenizer))
            self.base_model_path = model_name_or_path
            self.tokenizer.base_model_path = model_name_or_path

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
        if hasattr(self.vlm, "enable_input_require_grads"):
            self.vlm.enable_input_require_grads()

        # Monkey-patch Qwen2.5-VL patch_embed to use reshape instead of view
        # (fixes 4-bit quantized inference where tensors may be non-contiguous)
        self._patch_qwen25vl_patch_embed()

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

    def _patch_qwen25vl_patch_embed(self):
        """Replace .view() with .reshape() in Qwen2.5-VL / Qwen2-VL patch_embed for 4-bit compatibility."""
        def _make_patched_forward(original_forward):
            def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                target_dtype = self.proj.weight.dtype
                hidden_states = hidden_states.reshape(
                    -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
                )
                hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).reshape(-1, self.embed_dim)
                return hidden_states
            return patched_forward

        # Try Qwen2.5-VL first
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed
            Qwen2_5_VisionPatchEmbed.forward = _make_patched_forward(Qwen2_5_VisionPatchEmbed.forward)
        except Exception:
            pass

        # Try Qwen2-VL (2B/7B)
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed as Qwen2VisionPatchEmbed
            Qwen2VisionPatchEmbed.forward = _make_patched_forward(Qwen2VisionPatchEmbed.forward)
        except Exception:
            pass

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
        # Ensure at least one float input has requires_grad for gradient checkpointing to work
        if pixel_values is not None and pixel_values.is_floating_point() and not pixel_values.requires_grad:
            pixel_values = pixel_values.requires_grad_(True)
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
