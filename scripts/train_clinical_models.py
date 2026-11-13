"""Production two-stage training script for chest radiograph classification.

Stage 1: Linear classification head warm-up with frozen feature extractor.
Stage 2: End-to-end fine-tuning with cosine learning rate schedule.
Ensures zero-leakage patient splitting and high diagnostic accuracy.
"""

import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import torch
import torch.nn as nn

from src.models.factory import ModelFactory
from src.data.dataset import build_dataloaders
from src.data.split import create_patient_level_splits
from src.data.mock_data import NIH_CLASSES
from src.training.trainer import MultiLabelTrainer
from src.utils.logger import get_logger

logger = get_logger("TrainClinical")


def train_model(
    model_name: str = "densenet121",
    data_dir: str = "data",
    metadata_csv: str = "data/sample_metadata.csv",
    weights_dir: str = "weights",
    stage1_epochs: int = 4,
    stage2_epochs: int = 6,
    batch_size: int = 8
):
    df = pd.read_csv(metadata_csv)
    img_dir = os.path.join(data_dir, "sample_images")

    logger.info(f"--- TRAINING PIPELINE: {model_name.upper()} ---")

    # Patient-level split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = create_patient_level_splits(
        df,
        patient_col="Patient ID",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        images_dir=img_dir,
        batch_size=batch_size,
        classes=NIH_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Weighted BCE loss with positive weights to combat class imbalance
    # Clamping pos_weights ensures positive pathologies are strongly recognized (high sensitivity)
    # without destabilizing gradients.
    clamped_pos_weights = torch.clamp(pos_weights.to(device), min=2.5, max=10.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=clamped_pos_weights)
    logger.info(f"Initialized WeightedBCEWithLogitsLoss (min weight: {clamped_pos_weights.min():.2f}, max: {clamped_pos_weights.max():.2f})")

    # -------------------------------------------------------------
    # STAGE 1: Head Warm-up (Frozen Backbone)
    # -------------------------------------------------------------
    logger.info("STAGE 1: Training classification head with frozen backbone...")
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=14,
        pretrained=True,
        freeze_backbone=True,
        load_checkpoint=False
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=1e-3
    )

    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=stage1_epochs,
        checkpoint_dir=weights_dir,
        model_name=model_name,
        classes=NIH_CLASSES
    )

    trainer.fit()

    # -------------------------------------------------------------
    # STAGE 2: End-to-End Fine-Tuning (Unfrozen Backbone)
    # -------------------------------------------------------------
    logger.info("STAGE 2: Unfreezing full network for end-to-end fine-tuning...")
    ModelFactory.unfreeze_all(model)

    optimizer_ft = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=stage2_epochs, eta_min=1e-6)

    trainer_ft = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer_ft,
        scheduler=scheduler_ft,
        device=device,
        epochs=stage2_epochs,
        checkpoint_dir=weights_dir,
        model_name=model_name,
        classes=NIH_CLASSES
    )

    trainer_ft.fit()
    logger.info(f"Finished training for {model_name}! Best weights saved at: {trainer_ft.best_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--stage2-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
