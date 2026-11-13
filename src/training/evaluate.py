"""Comprehensive clinical evaluation metrics for multi-label chest radiograph classification."""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes the optimal classification threshold using Youden's J statistic (Sensitivity + Specificity - 1)."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    return np.clip(best_thresh, 0.05, 0.95)


def evaluate_multilabel_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    classes: List[str]
) -> Dict[str, Any]:
    """Evaluates multi-label model on test/val set across ROC-AUC, PR-AUC, and clinical operating points.

    Args:
        model: PyTorch model.
        dataloader: Test or validation DataLoader.
        device: torch.device.
        classes: List of 14 pathology names.

    Returns:
        Structured metrics dictionary.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images, targets, _ = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.vstack(all_preds)      # Shape: (N, 14)
    targets = np.vstack(all_targets)  # Shape: (N, 14)

    per_class_roc_auc = {}
    per_class_pr_auc = {}
    optimal_thresholds = {}
    per_class_sensitivity = {}
    per_class_specificity = {}
    per_class_f1 = {}

    valid_auc_scores = []
    valid_pr_scores = []

    for i, cls_name in enumerate(classes):
        y_true = targets[:, i]
        y_prob = preds[:, i]

        # Check if positive samples exist
        if len(np.unique(y_true)) >= 2:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
            opt_thresh = find_optimal_threshold(y_true, y_prob)

            binary_pred = (y_prob >= opt_thresh).astype(int)
            tn = int(np.sum((binary_pred == 0) & (y_true == 0)))
            fp = int(np.sum((binary_pred == 1) & (y_true == 0)))
            fn = int(np.sum((binary_pred == 0) & (y_true == 1)))
            tp = int(np.sum((binary_pred == 1) & (y_true == 1)))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, binary_pred, average="binary", zero_division=0
            )

            per_class_roc_auc[cls_name] = roc_auc
            per_class_pr_auc[cls_name] = pr_auc
            optimal_thresholds[cls_name] = opt_thresh
            per_class_sensitivity[cls_name] = float(sensitivity)
            per_class_specificity[cls_name] = float(specificity)
            per_class_f1[cls_name] = float(f1)

            valid_auc_scores.append(roc_auc)
            valid_pr_scores.append(pr_auc)
        else:
            # Fallback for unobserved classes in small test subsets
            per_class_roc_auc[cls_name] = 0.50
            per_class_pr_auc[cls_name] = 0.0
            optimal_thresholds[cls_name] = 0.50
            per_class_sensitivity[cls_name] = 0.0
            per_class_specificity[cls_name] = 1.0
            per_class_f1[cls_name] = 0.0

    mean_roc_auc = float(np.mean(valid_auc_scores)) if valid_auc_scores else 0.5
    mean_pr_auc = float(np.mean(valid_pr_scores)) if valid_pr_scores else 0.0

    return {
        "mean_roc_auc": mean_roc_auc,
        "mean_pr_auc": mean_pr_auc,
        "per_class_roc_auc": per_class_roc_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "optimal_thresholds": optimal_thresholds,
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "per_class_f1": per_class_f1
    }


def benchmark_model_latency(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int] = (224, 224),
    num_runs: int = 50,
    warmup: int = 10
) -> Dict[str, float]:
    """Measures single-image inference latency and throughput (FPS)."""
    model.eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Synchronize if GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time

    return {
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "device": str(device)
    }
