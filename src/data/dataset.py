"""PyTorch Dataset implementation for Multi-Label Chest Radiograph Classification."""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.data.preprocessor import MedicalImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for multi-label chest radiograph classification.

    Parses pipe-delimited finding labels into multi-hot binary vectors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        classes: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        preprocessor: Optional[MedicalImagePreprocessor] = None,
    ):
        """
        Args:
            df: DataFrame containing at least ['Image Index', 'Finding Labels'].
            images_dir: Base directory where radiograph images reside.
            classes: List of 14 target pathology categories.
            transform: Torchvision transform pipeline.
            preprocessor: MedicalImagePreprocessor instance.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.classes = classes or DEFAULT_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.preprocessor = preprocessor or MedicalImagePreprocessor()
        self.transform = transform

        # Precompute target labels matrix
        self.targets = self._encode_labels()

    def _encode_labels(self) -> np.ndarray:
        """Encodes pipe-separated labels into a multi-hot binary matrix of shape (N, 14)."""
        num_samples = len(self.df)
        labels_mat = np.zeros((num_samples, self.num_classes), dtype=np.float32)

        for i, finding_str in enumerate(self.df["Finding Labels"]):
            if pd.isna(finding_str) or finding_str.strip() in ("", "No Finding"):
                continue
            individual_findings = [f.strip() for f in str(finding_str).split("|")]
            for finding in individual_findings:
                # Handle space vs underscore discrepancy (e.g. Pleural Thickening vs Pleural_Thickening)
                normalized_name = finding.replace(" ", "_")
                if finding in self.class_to_idx:
                    labels_mat[i, self.class_to_idx[finding]] = 1.0
                elif normalized_name in self.class_to_idx:
                    labels_mat[i, self.class_to_idx[normalized_name]] = 1.0

        return labels_mat

    def compute_pos_weights(self) -> torch.Tensor:
        """Computes positive class weights for weighted BCE loss.

        Formula: pos_weight = (N_neg) / (N_pos + epsilon)
        Prevents dominant negative class from overwhelming gradients.
        """
        pos_counts = self.targets.sum(axis=0)
        total_samples = len(self.targets)
        neg_counts = total_samples - pos_counts

        # Guard against zero positive cases in small subsets
        pos_counts = np.maximum(pos_counts, 1.0)
        weights = neg_counts / pos_counts
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        image_name = row["Image Index"]

        # Check if direct image_path is provided in row
        if "image_path" in row and os.path.exists(str(row["image_path"])):
            img_path = str(row["image_path"])
        else:
            img_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        img = Image.open(img_path)
        img = self.preprocessor.harmonize_channels(img)

        # Apply CLAHE if enabled
        if self.preprocessor.use_clahe:
            img_np = np.array(img)
            img_np = self.preprocessor.apply_clahe(img_np)
            img = Image.fromarray(img_np)

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = self.preprocessor.get_val_transforms()(img)

        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

        meta = {
            "image_name": image_name,
            "patient_id": str(row.get("Patient ID", "Unknown")),
            "view_position": str(row.get("View Position", "PA")),
            "raw_findings": str(row.get("Finding Labels", "No Finding"))
        }

        return image_tensor, target_tensor, meta


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    classes: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Factory creating Train, Validation, and Test DataLoaders along with loss class weights."""
    preprocessor = MedicalImagePreprocessor()

    train_ds = ChestXrayDataset(
        df=train_df,
        images_dir=images_dir,
        classes=classes,
        transform=preprocessor.get_train_transforms(),
        preprocessor=preprocessor
    )

    val_ds = ChestXrayDataset(
        df=val_df,
        images_dir=images_dir,
        classes=classes,
        transform=preprocessor.get_val_transforms(),
        preprocessor=preprocessor
    )

    test_ds = ChestXrayDataset(
        df=test_df,
        images_dir=images_dir,
        classes=classes,
        transform=preprocessor.get_val_transforms(),
        preprocessor=preprocessor
    )

    pos_weights = train_ds.compute_pos_weights()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, pos_weights
