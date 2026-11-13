"""Explainable AI (XAI) Engine: Grad-CAM and Grad-CAM++ Implementations.

Grad-CAM: Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017)
Grad-CAM++: Generalized Grad-CAM for Multiple Instances (Chattopadhyay et al., WACV 2018)
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Standard Grad-CAM implementation for convolutional neural networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model in eval mode.
            target_layer: Convolutional layer to hook activations & gradients from.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is gradient with respect to layer output
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int,
        counterfactual: bool = False
    ) -> np.ndarray:
        """Computes Grad-CAM heatmap for a given class index.

        Args:
            input_tensor: Shape (1, 3, H, W).
            target_class_idx: Index of pathology class (0 to 13).
            counterfactual: If True, computes negative gradient (what evidence contradicts class).

        Returns:
            Normalized 2D numpy array (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]

        if counterfactual:
            score = -score

        # Backward pass
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture gradients or activations.")

        # Global average pooling of gradients across spatial dimensions (H, W)
        # gradients shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU: only positive contributions to class score are retained
        cam = F.relu(cam)

        # Resize to input spatial resolution
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(input_h, input_w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


class GradCAMPlusPlus:
    """Grad-CAM++ implementation for fine-grained multi-lesion localization."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int
    ) -> np.ndarray:
        """Computes Grad-CAM++ heatmap using higher-order gradient weighting."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture gradients or activations.")

        grads = self.gradients[0]  # (C, H_feat, W_feat)
        acts = self.activations[0]  # (C, H_feat, W_feat)

        # Higher order derivatives
        grad_2 = grads ** 2
        grad_3 = grads ** 3

        # Alpha weights denominator: 2*grad^2 + sum(acts * grad^3)
        denom = 2.0 * grad_2 + torch.sum(acts * grad_3, dim=(1, 2), keepdim=True)
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = grad_2 / denom
        weights = torch.sum(alpha * F.relu(grads), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * acts, dim=0, keepdim=True)
        cam = F.relu(cam).unsqueeze(0)

        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(input_h, input_w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np
