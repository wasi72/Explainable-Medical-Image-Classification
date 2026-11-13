"""Quantitative evaluation metrics for Explainable AI localization on chest radiographs."""

from typing import Tuple, List
import numpy as np


def pointing_game_hit(heatmap: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """Evaluates Pointing Game hit metric (Zhang et al., ECCV 2018).

    A hit occurs if the maximum activation pixel in the CAM heatmap lies within
    the ground truth clinical pathology bounding box.

    Args:
        heatmap: 2D numpy array (H, W).
        bbox: (x, y, width, height) in pixel coordinates matching heatmap dimensions.

    Returns:
        True if peak coordinate is inside bbox, False otherwise.
    """
    y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x, y, w, h = bbox
    return bool((x <= x_max <= x + w) and (y <= y_max <= y + h))


def cam_energy_in_bbox(heatmap: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Calculates proportion of explainability heatmap energy inside the pathology bounding box.

    Energy = sum(CAM inside bbox) / sum(Total CAM).
    Values closer to 1.0 indicate high localization precision with minimal spurious activations.
    """
    total_energy = np.sum(heatmap)
    if total_energy <= 1e-8:
        return 0.0

    h_img, w_img = heatmap.shape
    x, y, w, h = bbox

    # Clamp coordinates to image boundaries
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))

    bbox_energy = np.sum(heatmap[y1:y2, x1:x2])
    return float(bbox_energy / total_energy)


def cam_iou_with_bbox(
    heatmap: np.ndarray,
    bbox: Tuple[int, int, int, int],
    cam_threshold: float = 0.5
) -> float:
    """Computes Intersection over Union (IoU) between thresholded CAM and pathology bounding box."""
    binary_cam = (heatmap >= cam_threshold).astype(bool)

    h_img, w_img = heatmap.shape
    bbox_mask = np.zeros((h_img, w_img), dtype=bool)
    x, y, w, h = bbox

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))
    bbox_mask[y1:y2, x1:x2] = True

    intersection = np.logical_and(binary_cam, bbox_mask).sum()
    union = np.logical_or(binary_cam, bbox_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)
