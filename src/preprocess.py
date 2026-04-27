"""
Preprocessing pipeline for the Telco Churn dataset.

Key responsibilities
--------------------
1. Clean raw data (types, missing values).
2. Engineer features.
3. Synthesise a treatment column — see NOTE below.
4. Split into train / test sets.

NOTE — Synthetic Treatment Column
----------------------------------
The public Telco Churn dataset does not contain a randomised treatment
assignment (i.e. which customers were offered a retention discount).
We simulate one under the following assumptions:

  * Treatment probability is 40 % (mimicking a typical pilot campaign).
  * Assignment is independent of all features, so it is a valid RCT analogue
    for the purposes of training and evaluating uplift models.
  * In a real deployment this column would come from your CRM / A-B test
    infrastructure.

The column is named 'received_retention_offer' (1 = treated, 0 = control).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TREATMENT_PROBABILITY = 0.40
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and handle the TotalCharges edge case."""
    df = df.copy()
    # TotalCharges is sometimes ' ' for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns (binary or ordinal)."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight engineered features."""
    df = df.copy()
    df["charge_per_month"] = df["MonthlyCharges"] / (df["tenure"].clip(lower=1))
    df["is_long_tenure"] = (df["tenure"] >= 24).astype(int)
    return df


def add_synthetic_treatment(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Append a binary treatment column under completely-random assignment.

    This simulates a randomised controlled trial (RCT) where 40 % of
    customers are offered a retention discount. Because assignment is
    random and independent of covariates, the resulting data are valid
    for training uplift / CATE models.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["received_retention_offer"] = rng.binomial(
        n=1, p=TREATMENT_PROBABILITY, size=len(df)
    ).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing chain: clean → features → encode → treatment."""
    df = clean(df)
    df = add_features(df)
    df = encode_categoricals(df)
    df = add_synthetic_treatment(df)
    return df


def split(
    df: pd.DataFrame,
    target_col: str = "Churn",
    treatment_col: str = "received_retention_offer",
    test_size: float = 0.20,
    seed: int = RANDOM_STATE,
):
    """
    Return (X_train, X_test, y_train, y_test, w_train, w_test).

    X — feature matrix
    y — churn label (0/1)
    w — treatment indicator (0/1)
    """
    feature_cols = [c for c in df.columns if c not in (target_col, treatment_col)]
    X = df[feature_cols]
    y = df[target_col]
    w = df[treatment_col]

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, w_tr, w_te


def run(df: pd.DataFrame):
    """End-to-end helper used by the notebook."""
    processed = build_features(df)
    splits = split(processed)
    print(
        f"Train: {splits[0].shape[0]:,} rows | "
        f"Test: {splits[1].shape[0]:,} rows | "
        f"Features: {splits[0].shape[1]}"
    )
    return processed, splits
