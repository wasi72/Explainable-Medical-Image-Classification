"""CLI script to benchmark and compare all 4 architectures on held-out test data."""

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
from src.training.evaluate import evaluate_multilabel_performance, benchmark_model_latency
from src.utils.logger import get_logger

logger = get_logger("EvalBenchmark")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model architectures")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-csv", type=str, default="data/sample_metadata.csv")
    args = parser.parse_args()

    if not os.path.exists(args.metadata_csv):
        logger.info("Generating mock dataset for benchmarking...")
        df = create_mock_nih_dataset(output_dir=args.data_dir, num_samples=30, num_patients=10)
    else:
        df = pd.read_csv(args.metadata_csv)

    img_dir = os.path.join(args.data_dir, "sample_images")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_dir, "images")

    _, _, test_df = create_patient_level_splits(df, patient_col="Patient ID", train_size=0.7, val_size=0.15, test_size=0.15)
    _, _, test_loader, _ = build_dataloaders(
        train_df=test_df, val_df=test_df, test_df=test_df, images_dir=img_dir, batch_size=4, classes=NIH_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_test = ["baseline_cnn", "resnet50", "densenet121", "efficientnet_b0"]

    results = []
    print("\n" + "="*85)
    print("           COMPREHENSIVE MULTI-MODEL RESEARCH BENCHMARK REPORT")
    print("="*85)

    for m_name in models_to_test:
        logger.info(f"Evaluating {m_name}...")
        model = ModelFactory.create_model(m_name, num_classes=14, pretrained=True).to(device)
        eval_metrics = evaluate_multilabel_performance(model, test_loader, device, NIH_CLASSES)
        speed = benchmark_model_latency(model, device, num_runs=20, warmup=5)

        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        results.append({
            "Architecture": m_name,
            "Parameters (M)": f"{num_params:.2f}M",
            "Mean ROC-AUC": f"{eval_metrics['mean_roc_auc']:.4f}",
            "Mean PR-AUC": f"{eval_metrics['mean_pr_auc']:.4f}",
            "Latency (ms)": f"{speed['avg_latency_ms']:.1f} ms",
            "Throughput (FPS)": f"{speed['fps']:.1f}"
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    print("="*85 + "\n")

    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/model_benchmark_comparison.csv", index=False)
    logger.info("Saved benchmark report to reports/model_benchmark_comparison.csv")


if __name__ == "__main__":
    main()
