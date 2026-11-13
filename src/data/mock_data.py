"""Synthetic chest radiograph and metadata generator for training and testing.

Generates anatomically structured synthetic chest radiograph phantoms with
clear, clinically grounded radiological pathology patterns:
- No Finding (Normal): Clear bilateral lung fields, normal cardiothoracic ratio (CTR < 0.45),
  sharp costophrenic angles, intact diaphragmatic contour.
- Cardiomegaly: Markedly enlarged cardiac silhouette (CTR > 0.60), expanded transverse cardiac diameter.
- Effusion: Fluid accumulation obliterating the costophrenic sulcus with classic meniscus sign.
- Pneumonia: Ill-defined fluffy alveolar consolidative opacities in mid/lower lung zones.
- Nodule / Mass: Discrete, well-circumscribed dense parenchymal coin lesions.
- Pneumothorax: Visceral pleural line with peripheral hyperlucency and absent lung markings.
- Atelectasis: Linear horizontal plate-like discoid collapse bands at lung bases.
- Edema: Diffuse perihilar bat-wing vascular haziness.
- Consolidation: Dense homogeneous opacification obscuring vascular markings.
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

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
        findings: List of disease findings to simulate. Empty or ['No Finding'] creates a clean normal radiograph.

    Returns:
        PIL Image in grayscale ('L').
    """
    if findings is None or "No Finding" in findings:
        findings = []

    w, h = size
    # Radiopaque tissue / background soft tissue base
    base = np.full((h, w), 35.0, dtype=np.float32)

    y, x = np.ogrid[:h, :w]

    # 1. Bilateral Lung Fields (Air-filled = radiolucent, appears darker in X-ray)
    # Right hemithorax (appears on left in PA view)
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

    # Lungs air pockets: standard radiolucency level
    base[right_lung] = 165.0
    base[left_lung] = 160.0

    # 2. Cardiac Silhouette & Central Mediastinum
    is_cardiomegaly = "Cardiomegaly" in findings
    heart_scale_x = 1.65 if is_cardiomegaly else 1.0
    heart_scale_y = 1.35 if is_cardiomegaly else 1.0

    heart_center = (w * 0.54, h * 0.58)
    heart_mask = (
        ((x - heart_center[0]) / (w * 0.14 * heart_scale_x)) ** 2 +
        ((y - heart_center[1]) / (h * 0.15 * heart_scale_y)) ** 2
    ) <= 1.0
    # Heart is soft-tissue dense / radio-opaque = whiter (lower value in inverted display)
    base[heart_mask] = 55.0

    # Mediastinal column (spine and trachea)
    mediastinum_mask = (np.abs(x - w * 0.50) < w * 0.04) & (y < h * 0.75)
    base[mediastinum_mask] = np.minimum(base[mediastinum_mask], 70.0)

    # 3. Rib silhouettes & thoracic cage (arched posterior ribs)
    for rib_y in range(int(h * 0.22), int(h * 0.78), int(h * 0.08)):
        arch = 0.04 * (x - w / 2.0) ** 2 / w
        rib_mask = (np.abs(y - rib_y - arch) < 2.0) & (right_lung | left_lung)
        base[rib_mask] -= 22.0

    # Clavicles (upper horizontal bands)
    clavicle_r = (np.abs(y - (h * 0.18 + 0.05 * x)) < 3.0) & (x < w * 0.45) & (x > w * 0.15)
    clavicle_l = (np.abs(y - (h * 0.18 + 0.05 * (w - x))) < 3.0) & (x > w * 0.55) & (x < w * 0.85)
    base[clavicle_r | clavicle_l] = 50.0

    # 4. PATHOLOGY INJECTION (Only applied if findings are active)
    # Pleural Effusion: Blunting of costophrenic angles (fluid gravity accumulation at lung bases)
    if "Effusion" in findings:
        effusion_mask = (y > h * 0.68) & (x < w * 0.42) & right_lung
        # Dense radio-opaque fluid
        base[effusion_mask] = 50.0
        # Meniscus curvature
        meniscus = (y > (h * 0.64 + 0.15 * (x - w * 0.20))) & (x < w * 0.35)
        base[meniscus & right_lung] = 55.0

    # Pneumonia / Consolidation / Infiltration: Alveolar consolidation with patchy opacities
    if "Pneumonia" in findings or "Consolidation" in findings or "Infiltration" in findings:
        pneu_center = (w * 0.30, h * 0.54)
        pneu_mask = (
            ((x - pneu_center[0]) / (w * 0.12)) ** 2 +
            ((y - pneu_center[1]) / (h * 0.11)) ** 2
        ) <= 1.0
        # Fluffy dense alveolar infiltrate
        base[pneu_mask & right_lung] -= 75.0

    # Nodule / Mass: Well-circumscribed focal coin lesion
    if "Nodule" in findings or "Mass" in findings:
        radius = w * 0.06 if "Mass" in findings else w * 0.03
        nodule_center = (w * 0.72, h * 0.38)
        nodule_mask = (
            ((x - nodule_center[0]) / radius) ** 2 +
            ((y - nodule_center[1]) / radius) ** 2
        ) <= 1.0
        base[nodule_mask & left_lung] = 40.0

    # Pneumothorax: Apical hyperlucency with absence of lung markings
    if "Pneumothorax" in findings:
        ptx_mask = (y < h * 0.42) & (x > w * 0.62) & left_lung
        base[ptx_mask] = 230.0  # Pure air hyperlucency

    # Atelectasis: Linear horizontal plate-like discoid collapse band at base
    if "Atelectasis" in findings:
        atelectasis_mask = (np.abs(y - h * 0.65) < 3.5) & (x > w * 0.18) & (x < w * 0.40) & right_lung
        base[atelectasis_mask] = 45.0

    # Edema: Bat-wing perihilar vascular engorgement
    if "Edema" in findings:
        edema_mask = (np.abs(x - w * 0.50) < w * 0.22) & (np.abs(y - h * 0.52) < h * 0.16) & (right_lung | left_lung)
        base[edema_mask] -= 45.0

    # Add realistic Gaussian noise to simulate X-ray detector scatter
    noise = np.random.normal(0, 4.0, base.shape).astype(np.float32)
    final_arr = np.clip(base + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(final_arr, mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img


def create_mock_nih_dataset(
    output_dir: str = "data",
    num_samples: int = 300,
    num_patients: int = 60
) -> pd.DataFrame:
    """Generates a structured, balanced mock dataset with sufficient Normal and Pathology cases."""
    img_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    # 35% Normal cases to ensure model strongly learns healthy baseline
    # 65% Pathology cases across the key disease types
    disease_classes = [
        ["No Finding"],
        ["No Finding"],
        ["No Finding"],
        ["Cardiomegaly"],
        ["Effusion"],
        ["Pneumonia"],
        ["Consolidation"],
        ["Infiltration"],
        ["Nodule"],
        ["Mass"],
        ["Pneumothorax"],
        ["Atelectasis"],
        ["Edema"],
        ["Cardiomegaly", "Effusion"],
        ["Pneumonia", "Consolidation"],
        ["No Finding"],
        ["No Finding"]
    ]

    patient_ids = [f"PATIENT_{i+1:04d}" for i in range(num_patients)]

    for idx in range(num_samples):
        filename = f"0000{idx+1:04d}_001.png"
        img_path = os.path.join(img_dir, filename)

        findings = disease_classes[idx % len(disease_classes)]
        finding_str = "|".join(findings)

        patient_id = patient_ids[idx % num_patients]
        age = int(np.random.randint(20, 82))
        gender = "M" if (idx % 2 == 0) else "F"
        view = "PA" if (idx % 3 != 0) else "AP"

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
