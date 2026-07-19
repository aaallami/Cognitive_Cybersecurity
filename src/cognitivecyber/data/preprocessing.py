"""
Schema-agnostic preprocessing pipeline for NIDS flow data.

Steps
-----
1. Drop identifier/leakage columns (IPs, flow IDs, timestamps).
2. Impute missing values (median for numeric, mode for categorical).
3. Encode categorical columns (one-hot, capped cardinality with an
   "other" bucket to avoid explosion on high-cardinality real-world data).
4. Replace inf/-inf (common in CICFlowMeter-derived rate features).
5. Scale numeric features (StandardScaler).
6. Stratified train/val/test split.

This is intentionally dependency-light (pandas + scikit-learn only) so it
runs identically on the synthetic dataset and on any real dataset that
matches one of the registered schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAX_CATEGORY_CARDINALITY = 20


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler


def _cap_cardinality(series: pd.Series, max_categories: int = MAX_CATEGORY_CARDINALITY) -> pd.Series:
    top = series.value_counts().nlargest(max_categories).index
    return series.where(series.isin(top), other="__other__")


def clean_and_encode(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with all preprocessing applied except scaling/splitting."""
    df = df.copy()
    label_col = meta["label_column"]
    drop_cols = [c for c in meta.get("drop_columns", []) if c in df.columns]
    cat_cols = [c for c in meta.get("categorical_columns", []) if c in df.columns]

    y_raw = df[label_col]
    if not meta.get("is_binary", True):
        y = (y_raw.astype(str).str.lower() != "normal").astype(int) if y_raw.dtype == object else y_raw
    else:
        if y_raw.dtype == object:
            # e.g. CICIDS2017 'BENIGN' vs everything else
            y = (y_raw.astype(str).str.upper() != "BENIGN").astype(int)
        else:
            y = y_raw.astype(int)

    X = df.drop(columns=[label_col] + drop_cols, errors="ignore")

    # Replace inf/-inf which are common in *rate* columns of CICFlowMeter exports
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = [c for c in X.columns if c not in numeric_cols]

    for c in numeric_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in object_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "unknown")

    # Cap high-cardinality categoricals, then one-hot encode
    for c in cat_cols:
        if c in X.columns:
            X[c] = _cap_cardinality(X[c].astype(str))

    remaining_object_cols = [c for c in X.columns if X[c].dtype == object]
    if remaining_object_cols:
        X = pd.get_dummies(X, columns=remaining_object_cols, dummy_na=False)

    # Ensure everything is numeric now
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> PreprocessedData:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return PreprocessedData(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=list(X.columns),
        scaler=scaler,
    )


def preprocess(df: pd.DataFrame, meta: dict, **split_kwargs) -> PreprocessedData:
    X, y = clean_and_encode(df, meta)
    return split_and_scale(X, y, **split_kwargs)
