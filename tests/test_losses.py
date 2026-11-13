"""Tests for multi-label loss functions."""

import pytest
import torch
from src.training.losses import (
    WeightedBCEWithLogitsLoss,
    MultiLabelFocalLoss,
    AsymmetricLoss,
    get_loss_function
)


def test_weighted_bce_loss():
    pos_weights = torch.tensor([5.0, 10.0, 2.0])
    loss_fn = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)

    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_focal_loss():
    loss_fn = MultiLabelFocalLoss(gamma=2.0)
    logits = torch.randn(4, 14, requires_grad=True)
    targets = torch.randint(0, 2, (4, 14)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_asymmetric_loss():
    loss_fn = AsymmetricLoss()
    logits = torch.randn(4, 14, requires_grad=True)
    targets = torch.randint(0, 2, (4, 14)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
