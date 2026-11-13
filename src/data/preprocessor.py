"""Medical image preprocessing pipeline for Chest Radiographs.

Includes:
- Quality validation & corruption detection
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Multi-channel color harmonization (1-channel / 4-channel -> 3-channel RGB)
- Clinically-sound data augmentation pipelines
- Normalization (ImageNet stats or custom chest radiograph statistics)
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MedicalImagePreprocessor:
    """Production preprocessor for chest radiographs."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
        std: Tuple[float, float, float] = tuple(IMAGENET_STD),
    ):
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.mean = list(mean)
        self.std = list(std)

        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )

    def validate_image(self, image: Union[Image.Image, np.ndarray, str]) -> bool:
        """Validates image integrity and dimension sanity.

        Args:
            image: Image object, numpy array, or file path.

        Returns:
            True if image is valid, raises ValueError otherwise.
        """
        if isinstance(image, str):
            try:
                with Image.open(image) as img:
                    img.verify()
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Corrupted or invalid image at {image}: {e}")

        if isinstance(image, Image.Image):
            arr = np.array(image)
        elif isinstance(image, np.ndarray):
            arr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if arr.size == 0:
            raise ValueError("Empty image array encountered.")

        if arr.ndim < 2:
            raise ValueError(f"Invalid dimensions for 2D medical radiograph: {arr.shape}")

        if np.all(arr == 0):
            logger.warning("Image consists entirely of black pixels (all zeros).")

        return True

    def apply_clahe(self, img_np: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization.

        Enhances subtle pulmonary parenchymal details and nodular opacities.
        """
        if not self.use_clahe:
            return img_np

        if img_np.ndim == 2:
            # Single channel grayscale
            return self.clahe.apply(img_np.astype(np.uint8))
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            # Convert to LAB color space, apply CLAHE to L-channel, convert back
            lab = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            merged = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            return img_np

    def harmonize_channels(self, img: Image.Image) -> Image.Image:
        """Ensures the image is standard 3-channel RGB.

        Chest X-rays are typically grayscale (L), but pretrained CNN backbones
        expect 3-channel RGB. We replicate the single channel 3 times.
        """
        if img.mode == "RGB":
            return img
        elif img.mode in ("L", "I;16", "I"):
            # Convert 16-bit or 8-bit grayscale to 8-bit RGB
            img_gray = img.convert("L")
            return Image.merge("RGB", (img_gray, img_gray, img_gray))
        elif img.mode == "RGBA":
            # Remove alpha channel with white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        else:
            return img.convert("RGB")

    def get_train_transforms(self) -> transforms.Compose:
        """Data augmentation pipeline for training.

        Notice: Vertical flips and extreme rotations are strictly forbidden in
        chest radiology as they alter anatomical orientation.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04)
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_val_transforms(self) -> transforms.Compose:
        """Deterministic transform pipeline for validation and testing."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess_for_inference(
        self,
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocesses a single image for model inference and XAI visualization.

        Returns:
            tensor: Shape (1, 3, H, W) normalized PyTorch tensor ready for model.
            display_img: (H, W, 3) float32 in range [0, 1] RGB for Grad-CAM overlay.
        """
        if isinstance(image_input, str):
            self.validate_image(image_input)
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            self.validate_image(image_input)
            img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            self.validate_image(image_input)
            img = image_input
        else:
            raise TypeError(f"Invalid input type: {type(image_input)}")

        img = self.harmonize_channels(img)
        img_np = np.array(img)

        if self.use_clahe:
            img_np = self.apply_clahe(img_np)
            img = Image.fromarray(img_np)

        # Resize for display
        display_img = np.array(img.resize(self.image_size)).astype(np.float32) / 255.0

        val_tf = self.get_val_transforms()
        tensor = val_tf(img).unsqueeze(0)  # Add batch dim: (1, 3, H, W)

        return tensor, display_img
