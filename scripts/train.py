"""CLI script to train chest radiograph multi-label models."""

import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import torch

from src.models.factory import ModelFactory
from src.data.dataset import build_dataloaders
from src.data.split import create_patient_level_splits
from src.data.mock_data import create_mock_nih_dataset, NIH_CLASSES
from src.training.losses import get_loss_function
from src.training.trainer import MultiLabelTrainer
from src.utils.logger import get_logger

logger = get_logger("TrainCLI")


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label chest X-ray models")
    parser.add_argument("--model", type=str, default="densenet121", choices=["densenet121", "resnet50", "efficientnet_b0", "baseline_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="weighted_bce", choices=["bce", "weighted_bce", "focal", "asymmetric"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-csv", type=str, default="data/sample_metadata.csv")
    parser.add_argument("--weights-dir", type=str, default="weights")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting training run: model={args.model}, epochs={args.epochs}, loss={args.loss}")

    # Ensure dataset exists; if not, create mock dataset
    if not os.path.exists(args.metadata_csv):
        logger.info(f"Metadata CSV not found at {args.metadata_csv}. Generating self-contained mock dataset...")
        df = create_mock_nih_dataset(output_dir=args.data_dir, num_samples=30, num_patients=10)
    else:
        df = pd.read_csv(args.metadata_csv)

    img_dir = os.path.join(args.data_dir, "sample_images")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_dir, "images")

    # Patient-level split
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
        batch_size=args.batch_size,
        classes=NIH_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Model instantiation
    model = ModelFactory.create_model(
        model_name=args.model,
        num_classes=14,
        pretrained=True
    ).to(device)

    # Loss function
    loss_fn = get_loss_function(
        loss_type=args.loss,
        pos_weights=pos_weights.to(device) if args.loss == "weighted_bce" else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
        checkpoint_dir=args.weights_dir,
        model_name=args.model,
        classes=NIH_CLASSES
    )

    history = trainer.fit()
    logger.info("Training successfully completed!")
    logger.info(f"Best checkpoint saved at {trainer.best_checkpoint_path}")


if __name__ == "__main__":
    main()
