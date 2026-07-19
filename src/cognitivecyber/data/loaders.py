"""
Dataset loading with automatic schema, label, and encoding detection.

If `path` is given, this reads a real dataset file the user has supplied
(CSV/Parquet) and auto-detects: column encoding, the label column, whether
labels are binary or multi-class, categorical columns, and missing-value
markers -- so the same downstream pipeline works across CICIDS2017,
CSE-CIC-IDS2018, UNSW-NB15, TON_IoT, and Bot-IoT without per-dataset code.

If `path` is None, a synthetic schema-compatible dataset is generated
instead (see `cognitivecyber.data.synthetic`), which is what the shipped
notebooks use by default since the real files cannot be fetched inside
this execution environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .schemas import SCHEMAS, detect_schema
from .synthetic import generate_synthetic_flows

logger = logging.getLogger("cognitivecyber.data.loaders")

MISSING_MARKERS = ["-", "NaN", "nan", "?", "Infinity", "-Infinity", ""]
ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252"]


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet with best-effort encoding detection."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                na_values=MISSING_MARKERS,
                low_memory=False,
                skipinitialspace=True,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue
    raise ValueError(f"Could not read {path} with encodings {ENCODINGS_TO_TRY}: {last_err}")


def load_dataset(
    path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    n_synthetic: int = 60_000,
    random_state: int = 42,
):
    """Load a NIDS dataset, real or synthetic, with schema auto-detection.

    Returns
    -------
    df : pd.DataFrame
        Cleaned column names (whitespace-stripped), original data otherwise.
    meta : dict
        {
          "label_column": str,
          "categorical_columns": list[str],
          "drop_columns": list[str],
          "is_binary": bool,
          "schema_name": str,
          "is_synthetic": bool,
        }
    """
    if path is None:
        logger.info(
            "No dataset path supplied -- generating synthetic schema-compatible "
            "flow data (%d samples, seed=%d). Supply `path=` to use a real dataset.",
            n_synthetic,
            random_state,
        )
        df = generate_synthetic_flows(n_samples=n_synthetic, random_state=random_state)
        meta = {
            "label_column": "label",
            "categorical_columns": ["proto", "service", "state"],
            "drop_columns": ["is_synthetic", "attack_cat"],
            "is_binary": True,
            "schema_name": "synthetic_unsw_style",
            "is_synthetic": True,
            "multiclass_column": "attack_cat",
        }
        return df, meta

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}. Place real dataset files under "
            f"datasets/raw/ or pass an explicit path."
        )

    df = _read_any(p)
    df.columns = [c.strip() for c in df.columns]

    schema = SCHEMAS.get(dataset_name) if dataset_name else detect_schema(list(df.columns))
    if schema is None:
        # Generic auto-profiling fallback: guess the label column as the last
        # object/int column whose name looks label-like.
        candidates = [c for c in df.columns if c.lower() in ("label", "class", "attack", "attack_cat", "target")]
        label_col = candidates[0] if candidates else df.columns[-1]
        cat_cols = [c for c in df.columns if df[c].dtype == object and c != label_col]
        meta = {
            "label_column": label_col,
            "categorical_columns": cat_cols,
            "drop_columns": [],
            "is_binary": df[label_col].nunique() <= 2,
            "schema_name": "auto_detected_generic",
            "is_synthetic": False,
        }
        logger.warning(
            "No known schema matched columns; using generic auto-detection "
            "(label_column=%s).",
            label_col,
        )
        return df, meta

    is_binary = df[schema.label_column].nunique() <= 2
    meta = {
        "label_column": schema.label_column,
        "categorical_columns": schema.categorical_columns,
        "drop_columns": schema.drop_columns,
        "is_binary": is_binary,
        "schema_name": schema.name,
        "is_synthetic": False,
    }
    return df, meta
