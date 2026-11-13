"""CLI utility to generate high-fidelity synthetic chest radiograph phantoms and metadata."""

import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.data.mock_data import create_mock_nih_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic chest radiograph dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of radiograph images")
    parser.add_argument("--num-patients", type=int, default=20, help="Number of distinct patient IDs")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} synthetic radiographs across {args.num_patients} patients...")
    create_mock_nih_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_patients=args.num_patients
    )
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
