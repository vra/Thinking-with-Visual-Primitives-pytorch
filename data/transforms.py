"""Image transforms for VL models."""

from pathlib import Path
from typing import Union
from PIL import Image
import torch


class ImageTransform:
    """
    Simple image transform that either uses a processor (if provided)
    or does basic resize + to_tensor + normalize.
    """

    def __init__(self, image_size: int = 448, processor=None):
        self.image_size = image_size
        self.processor = processor

    def __call__(self, image: Union[Image.Image, str, Path]):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        if self.processor is not None:
            # Let the VL processor handle everything
            return image

        # Fallback: basic resize + to tensor + normalize
        image = image.resize((self.image_size, self.image_size))
        arr = torch.tensor([[[c / 255.0 for c in image.getpixel((x, y))]
                             for x in range(self.image_size)]
                            for y in range(self.image_size)], dtype=torch.float32)
        arr = arr.permute(2, 0, 1)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        arr = (arr - mean) / std
        return arr
