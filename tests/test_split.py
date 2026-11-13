"""Tests for patient-level data splitting to verify zero patient leakage."""

import pytest
import pandas as pd
from src.data.split import create_patient_level_splits


def test_patient_level_splits_leakage_free():
    # 100 images from 20 unique patients
    records = []
    for i in range(100):
        patient_id = f"PATIENT_{i % 20:03d}"
        records.append({
            "Image Index": f"img_{i:04d}.png",
            "Patient ID": patient_id,
            "Finding Labels": "Cardiomegaly" if i % 2 == 0 else "No Finding"
        })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = create_patient_level_splits(
        df,
        patient_col="Patient ID",
        train_size=0.70,
        val_size=0.10,
        test_size=0.20,
        random_state=42
    )

    train_pts = set(train_df["Patient ID"])
    val_pts = set(val_df["Patient ID"])
    test_pts = set(test_df["Patient ID"])

    # Core assertion: zero patient leakage
    assert train_pts.isdisjoint(val_pts)
    assert train_pts.isdisjoint(test_pts)
    assert val_pts.isdisjoint(test_pts)

    # Total patient count must equal 20
    assert len(train_pts) + len(val_pts) + len(test_pts) == 20
    # Total images must sum to 100
    assert len(train_df) + len(val_df) + len(test_df) == 100
