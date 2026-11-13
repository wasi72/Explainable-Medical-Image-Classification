"""Patient-level data splitting to guarantee zero data leakage between train/val/test splits."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_patient_level_splits(
    df: pd.DataFrame,
    patient_col: str = "Patient ID",
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    official_test_patients: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a chest radiograph dataframe by unique Patient ID to eliminate data leakage.

    Args:
        df: Metadata DataFrame containing image references and patient IDs.
        patient_col: Name of column containing unique patient identifiers.
        train_size: Proportion of patients for training (e.g. 0.70).
        val_size: Proportion of patients for validation (e.g. 0.10).
        test_size: Proportion of patients for testing (e.g. 0.20).
        random_state: Random seed for reproducibility.
        official_test_patients: Optional predefined list of patient IDs for official test set.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    total_ratio = train_size + val_size + test_size
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got: {total_ratio}")

    all_patients = [str(p) for p in df[patient_col].unique()]
    num_total_patients = len(all_patients)
    logger.info(f"Total images: {len(df)} | Unique patients: {num_total_patients}")

    if official_test_patients is not None:
        # Use predefined official test set
        test_patient_set = set(official_test_patients)
        train_val_patients = [p for p in all_patients if p not in test_patient_set]
        test_patients = [p for p in all_patients if p in test_patient_set]

        # Split remaining into train & val
        val_relative_size = val_size / (train_size + val_size)
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_relative_size,
            random_state=random_state
        )
    else:
        # 3-way patient-level split
        val_plus_test = val_size + test_size
        train_patients, val_test_patients = train_test_split(
            all_patients,
            test_size=val_plus_test,
            random_state=random_state
        )

        test_relative_size = test_size / val_plus_test
        val_patients, test_patients = train_test_split(
            val_test_patients,
            test_size=test_relative_size,
            random_state=random_state
        )

    # Verification of zero patient overlap
    set_train = set(train_patients)
    set_val = set(val_patients)
    set_test = set(test_patients)

    assert set_train.isdisjoint(set_val), "Data Leakage Alert: Train and Val share patients!"
    assert set_train.isdisjoint(set_test), "Data Leakage Alert: Train and Test share patients!"
    assert set_val.isdisjoint(set_test), "Data Leakage Alert: Val and Test share patients!"

    train_df = df[df[patient_col].isin(set_train)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(set_val)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(set_test)].copy().reset_index(drop=True)

    logger.info(
        f"Split complete [Zero Leakage Verified]:\n"
        f"  Train: {len(train_df)} images ({len(set_train)} patients, {len(set_train)/num_total_patients:.1%})\n"
        f"  Val:   {len(val_df)} images ({len(set_val)} patients, {len(set_val)/num_total_patients:.1%})\n"
        f"  Test:  {len(test_df)} images ({len(set_test)} patients, {len(set_test)/num_total_patients:.1%})"
    )

    return train_df, val_df, test_df
