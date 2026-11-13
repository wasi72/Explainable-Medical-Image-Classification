"""Visualization utilities for medical explainability heatmaps, overlays, and clinical reports."""

import io
import base64
from typing import Dict, List, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def create_cam_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> np.ndarray:
    """Overlays a 2D explainability heatmap onto a medical radiograph.

    Args:
        image_np: (H, W, 3) in range [0, 1] (or [0, 255]).
        heatmap: (H, W) float in range [0, 1].
        alpha: Blending weight for heatmap (0.0 = only image, 1.0 = only heatmap).
        colormap: OpenCV colormap ('jet', 'viridis', 'inferno', 'turbo').

    Returns:
        RGB numpy array (H, W, 3) in uint8 [0, 255].
    """
    # Ensure image_np is uint8 [0, 255]
    if image_np.max() <= 1.0:
        base_img = (image_np * 255).astype(np.uint8)
    else:
        base_img = image_np.astype(np.uint8)

    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    elif base_img.shape[2] == 1:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    # Ensure heatmap is uint8 [0, 255]
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

    # Select OpenCV colormap
    cv_maps = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO
    }
    cv_cmap = cv_maps.get(colormap.lower(), cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv_cmap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(heatmap_colored_rgb, alpha, base_img, 1.0 - alpha, 0)
    return overlay


def create_clinical_comparison_panel(
    original_img: np.ndarray,
    gradcam_overlay: np.ndarray,
    gradcam_pp_overlay: np.ndarray,
    saliency_map: np.ndarray,
    pathology_name: str,
    probability: float,
    threshold: float = 0.5
) -> Image.Image:
    """Generates a 4-panel publication-grade diagnostic comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a24")

    titles = [
        "Original Chest Radiograph",
        f"Grad-CAM ({pathology_name})",
        f"Grad-CAM++ ({pathology_name})",
        "Pixel Saliency Map"
    ]

    images = [
        original_img,
        gradcam_overlay,
        gradcam_pp_overlay,
        saliency_map
    ]

    cmaps = [None, None, None, "inferno"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.set_facecolor("#1a1a24")
        if cmap is not None:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, pad=10, fontweight="bold")
        ax.axis("off")

    status = "POSITIVE SUSPICION" if probability >= threshold else "NEGATIVE / SUB-THRESHOLD"
    status_color = "#ff4d4d" if probability >= threshold else "#00e676"

    fig.suptitle(
        f"Clinical Decision Support: {pathology_name} | Predicted Probability: {probability:.1%} | Assessment: {status}",
        color=status_color,
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def image_to_base64(img_np: np.ndarray) -> str:
    """Converts a numpy image array to base64 PNG data string."""
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
