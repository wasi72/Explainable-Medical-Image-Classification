"""Input Saliency and SmoothGrad Pixel-Attribution Maps."""

import numpy as np
import torch
import torch.nn as nn


class VanillaSaliency:
    """Computes gradient of target pathology score with respect to input radiograph pixels."""

    def __init__(self, model: nn.Module):
        self.model = model

    def generate_saliency(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int
    ) -> np.ndarray:
        """Computes pixel saliency map.

        Args:
            input_tensor: (1, 3, H, W).
            target_class_idx: Class index.

        Returns:
            Normalized 2D numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        tensor_copy = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(tensor_copy)
        score = logits[0, target_class_idx]
        score.backward()

        # Take maximum gradient across RGB color channels
        # Shape: (1, 3, H, W) -> (H, W)
        grad = tensor_copy.grad.squeeze(0)  # (3, H, W)
        saliency, _ = torch.max(torch.abs(grad), dim=0)

        sal_np = saliency.cpu().numpy()
        s_min, s_max = np.min(sal_np), np.max(sal_np)
        if s_max - s_min > 1e-8:
            sal_np = (sal_np - s_min) / (s_max - s_min)
        else:
            sal_np = np.zeros_like(sal_np)

        return sal_np


class SmoothGrad:
    """Averages saliency across Gaussian-perturbed input copies to reduce visual noise."""

    def __init__(
        self,
        model: nn.Module,
        stdev_spread: float = 0.15,
        n_samples: int = 15
    ):
        self.model = model
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples

    def generate_saliency(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int
    ) -> np.ndarray:
        self.model.eval()
        stdev = self.stdev_spread * (input_tensor.max() - input_tensor.min()).item()
        total_gradients = torch.zeros_like(input_tensor)

        for _ in range(self.n_samples):
            noise = torch.normal(0, stdev, size=input_tensor.shape).to(input_tensor.device)
            noisy_input = (input_tensor + noise).detach().requires_grad_(True)

            logits = self.model(noisy_input)
            score = logits[0, target_class_idx]
            self.model.zero_grad()
            score.backward()

            total_gradients += noisy_input.grad

        avg_grad = (total_gradients / self.n_samples).squeeze(0)
        saliency, _ = torch.max(torch.abs(avg_grad), dim=0)
        sal_np = saliency.cpu().numpy()

        s_min, s_max = np.min(sal_np), np.max(sal_np)
        if s_max - s_min > 1e-8:
            sal_np = (sal_np - s_min) / (s_max - s_min)
        else:
            sal_np = np.zeros_like(sal_np)

        return sal_np
