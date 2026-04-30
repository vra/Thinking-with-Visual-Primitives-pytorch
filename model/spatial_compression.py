"""
Spatial token compression module.

Paper describes:
  At the ViT output, apply a 3x3 spatial compression
  (compress every 9 adjacent patch tokens into a single token along the channel dimension).

Example:
  756x756 image -> 14x14 patch -> 54x54 = 2916 patch tokens
  After 3x3 compression: 18x18 = 324 tokens
  Then CSA further compresses by 4x -> 81 KV entries.

In this reproduction, we implement the 3x3 spatial compression.
CSA (Compressed Sparse Attention) is proprietary, so we skip it or use standard attention.
"""

import torch
import torch.nn as nn


class SpatialCompression(nn.Module):
    """
    Compress a 2D grid of tokens using a non-overlapping 3x3 kernel.

    Input:  (B, H, W, C) where H and W are spatial dimensions of patch tokens.
    Output: (B, H//3, W//3, C*9) if concat mode, or (B, H//3, W//3, C_out) if projection mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = 3,
        mode: str = "project",  # "concat" or "project"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        if mode == "project":
            out_channels = out_channels or in_channels
            # Each 3x3 block has kernel_size^2 tokens; we project from in_channels * kernel_size^2
            self.proj = nn.Linear(
                in_channels * (kernel_size ** 2),
                out_channels,
            )
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels * (kernel_size ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) tensor
        Returns:
            (B, H//k, W//k, C_out) tensor
        """
        B, H, W, C = x.shape
        k = self.kernel_size
        assert H % k == 0 and W % k == 0, f"Spatial dims ({H},{W}) must be divisible by {k}"

        # (B, H, W, C) -> (B, H//k, k, W//k, k, C)
        x = x.reshape(B, H // k, k, W // k, k, C)
        # -> (B, H//k, W//k, k, k, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # -> (B, H//k, W//k, k*k*C)
        x = x.reshape(B, H // k, W // k, k * k * C)

        if self.mode == "project":
            x = self.proj(x)
        return x


class PatchEmbedAnyResolution(nn.Module):
    """
    Patch embedding that supports arbitrary-resolution images.
    Partitions the image using patch_size x patch_size patches.
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            (B, H//p, W//p, D) where p = patch_size
        """
        # (B, D, H//p, W//p)
        x = self.proj(pixel_values)
        # (B, H//p, W//p, D)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
