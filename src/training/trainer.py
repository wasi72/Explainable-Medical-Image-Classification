"""Production Multi-Label Trainer with Early Stopping, LR Scheduling, and Checkpointing."""

import os
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.evaluate import evaluate_multilabel_performance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiLabelTrainer:
    """Orchestrates multi-label model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 20,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "weights",
        model_name: str = "densenet121",
        classes: Optional[list] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.classes = classes or []

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "lr": []
        }

    def train_one_epoch(self, epoch: int) -> float:
        """Executes a single training epoch."""
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        for step, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)
            loss.backward()

            # Gradient clipping to stabilize deep backbones
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

            self.optimizer.step()
            running_loss += loss.item()

            if (step + 1) % max(1, total_batches // 4) == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}] Step [{step+1}/{total_batches}] "
                    f"Batch Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / max(1, total_batches)
        return epoch_loss

    def validate(self) -> (float, float):
        """Computes validation loss and mean validation AUROC."""
        self.model.eval()
        running_loss = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                running_loss += loss.item()

        val_loss = running_loss / max(1, total_batches)

        # Compute validation AUROC if classes provided
        if self.classes:
            metrics = evaluate_multilabel_performance(
                self.model, self.val_loader, self.device, self.classes
            )
            val_auc = metrics["mean_roc_auc"]
        else:
            val_auc = 0.0

        return val_loss, val_auc

    def fit(self) -> Dict[str, Any]:
        """Runs the complete training loop with early stopping and checkpointing."""
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}"
            )

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpointing best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_name": self.model_name,
                    "state_dict": self.model.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "optimizer_state": self.optimizer.state_dict(),
                    "classes": self.classes
                }, self.best_checkpoint_path)
                logger.info(f"Saved new best model checkpoint to {self.best_checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping counter: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered. Halting training.")
                    break

        return self.history
