"""Synthetic chest radiograph and metadata generator for automated testing and quickstart demos.

Generates anatomically plausible synthetic chest radiograph phantoms with
controllable radiological pathology patterns (Cardiomegaly, Effusion, Pneumonia, Nodule, Normal).
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src.utils.logger import get_logger

logger = get_logger(__name__)

NIH_CLASSES = [
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


def generate_synthetic_chest_xray(
    size: Tuple[int, int] = (224, 224),
    findings: List[str] = None
) -> Image.Image:
    """Generates an anatomically structured synthetic chest radiograph phantom.

    Args:
        size: Target image dimensions (width, height).
        findings: List of disease findings to simulate (e.g., ['Cardiomegaly', 'Effusion']).

    Returns:
        PIL Image in grayscale ('L' mode).
    """
    if findings is None:
        findings = []

    w, h = size
    # Dark background (mediastinum & tissue background)
    base = np.full((h, w), 35, dtype=np.float32)

    # 1. Bilateral Lung Fields (radiolucent = darker air pockets)
    # Right Lung (appears on left in standard PA view)
    y, x = np.ogrid[:h, :w]
    right_lung_center = (w * 0.32, h * 0.48)
    left_lung_center = (w * 0.68, h * 0.48)

    right_lung = (
        ((x - right_lung_center[0]) / (w * 0.18)) ** 2 +
        ((y - right_lung_center[1]) / (h * 0.30)) ** 2
    ) <= 1.0

    left_lung = (
        ((x - left_lung_center[0]) / (w * 0.17)) ** 2 +
        ((y - left_lung_center[1]) / (h * 0.28)) ** 2
    ) <= 1.0

    # Lungs are darker in X-ray radiograph (air absorbs fewer X-rays)
    base[right_lung] = 160
    base[left_lung] = 155

    # 2. Cardiac Silhouette (radiopaque = bright heart in central mediastinum)
    heart_scale = 1.45 if "Cardiomegaly" in findings else 1.0
    heart_center = (w * 0.54, h * 0.58)
    heart_mask = (
        ((x - heart_center[0]) / (w * 0.15 * heart_scale)) ** 2 +
        ((y - heart_center[1]) / (h * 0.16 * heart_scale)) ** 2
    ) <= 1.0
    base[heart_mask] = 60

    # 3. Rib silhouettes (horizontal arched radiopaque lines)
    for rib_y in range(int(h * 0.25), int(h * 0.8), int(h * 0.1)):
        rib_mask = (np.abs(y - rib_y - 0.05 * (x - w/2)**2 / w) < 2.5) & (right_lung | left_lung)
        base[rib_mask] -= 25

    # 4. Pathology injection
    if "Effusion" in findings:
        # Costophrenic sulcus blunting / meniscus sign at lung base
        effusion_mask = (y > h * 0.70) & (x < w * 0.42)
        base[effusion_mask] = 50

    if "Pneumonia" in findings or "Consolidation" in findings or "Infiltration" in findings:
        # Ill-defined fluffy alveolar opacity in right mid/lower zone
        pneu_center = (w * 0.30, h * 0.55)
        pneu_mask = (
            ((x - pneu_center[0]) / (w * 0.10)) ** 2 +
            ((y - pneu_center[1]) / (h * 0.10)) ** 2
        ) <= 1.0
        base[pneu_mask] -= 65

    if "Nodule" in findings or "Mass" in findings:
        # Well-circumscribed focal opacity
        nodule_radius = w * 0.06 if "Mass" in findings else w * 0.03
        nodule_center = (w * 0.72, h * 0.40)
        nodule_mask = (
            ((x - nodule_center[0]) / nodule_radius) ** 2 +
            ((y - nodule_center[1]) / nodule_radius) ** 2
        ) <= 1.0
        base[nodule_mask] -= 80

    if "Pneumothorax" in findings:
        # Hyperlucency (absence of lung markings) in left upper zone
        ptx_mask = (y < h * 0.40) & (x > w * 0.65) & left_lung
        base[ptx_mask] = 220

    # Add realistic Gaussian noise and blur to simulate scatter & detector noise
    noise = np.random.normal(0, 5, base.shape).astype(np.float32)
    final_arr = np.clip(base + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(final_arr, mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def create_mock_nih_dataset(
    output_dir: str = "data",
    num_samples: int = 40,
    num_patients: int = 15
) -> pd.DataFrame:
    """Generates a complete self-contained mock NIH ChestX-ray14 dataset on disk.

    Creates both the PNG images in `output_dir/sample_images` and the `sample_metadata.csv`.
    """
    img_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    # Seed reproducible findings
    preset_cases = [
        ["No Finding"],
        ["Cardiomegaly"],
        ["Effusion"],
        ["Pneumonia"],
        ["Infiltration"],
        ["Atelectasis"],
        ["Effusion", "Cardiomegaly"],
        ["Pneumonia", "Consolidation"],
        ["Nodule"],
        ["Pneumothorax"],
        ["Mass"],
        ["Emphysema"],
        ["Edema"],
        ["Pleural_Thickening"],
        ["Fibrosis"],
        ["Hernia"]
    ]

    patient_ids = [f"PATIENT_{i+1:04d}" for i in range(num_patients)]

    for idx in range(num_samples):
        filename = f"0000{idx+1:04d}_001.png"
        img_path = os.path.join(img_dir, filename)

        findings = preset_cases[idx % len(preset_cases)]
        finding_str = "|".join(findings)

        patient_id = patient_ids[idx % num_patients]
        age = np.random.randint(22, 85)
        gender = "M" if np.random.rand() > 0.5 else "F"
        view = "PA" if np.random.rand() > 0.3 else "AP"

        img = generate_synthetic_chest_xray(size=(224, 224), findings=findings)
        img.save(img_path)

        records.append({
            "Image Index": filename,
            "Finding Labels": finding_str,
            "Follow-up #": idx % 3,
            "Patient ID": patient_id,
            "Patient Age": age,
            "Patient Gender": gender,
            "View Position": view,
            "OriginalImage[Width": 1024,
            "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143,
            "image_path": img_path
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sample_metadata.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated {num_samples} mock chest X-rays across {num_patients} patients at {img_dir}")
    logger.info(f"Saved metadata to {csv_path}")
    return df
