"""
Dataset schema registry for supported network intrusion detection datasets.

Each entry documents the *real* published schema (column names, label
column, categorical columns, and known encodings) for the corresponding
public dataset. This module does not embed or redistribute any dataset
content -- only column-level metadata used for automatic schema detection
when a user supplies their own copy of the dataset file.

Supported datasets (real-world schemas, user must supply the actual files):
    - CICIDS2017
    - CSE-CIC-IDS2018
    - UNSW-NB15
    - TON_IoT
    - Bot-IoT
    - DARPA Transparent Computing (TC) -- graph/log format, handled separately

If no local dataset file is supplied, `cognitivecyber.data.synthetic`
generates a schema-compatible synthetic dataset so the rest of the pipeline
(preprocessing, baselines, evaluation, figures) can be exercised end-to-end.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetSchema:
    name: str
    label_column: str
    categorical_columns: list
    drop_columns: list
    binary_positive_labels: list  # label values considered "attack"/positive
    notes: str = ""


SCHEMAS = {
    "CICIDS2017": DatasetSchema(
        name="CICIDS2017",
        label_column="Label",
        categorical_columns=[],
        drop_columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"],
        binary_positive_labels=["BENIGN"],  # inverted: everything != BENIGN is attack
        notes=(
            "CICVFlowMeter-derived flow features (78-84 cols depending on "
            "release). 'Label' column contains BENIGN or one of 14 attack "
            "categories (DoS, DDoS, PortScan, Brute Force, Web Attack, "
            "Infiltration, Bot, Heartbleed)."
        ),
    ),
    "CSE-CIC-IDS2018": DatasetSchema(
        name="CSE-CIC-IDS2018",
        label_column="Label",
        categorical_columns=["Protocol"],
        drop_columns=["Timestamp"],
        binary_positive_labels=["Benign"],
        notes="Similar CICFlowMeter feature set to CICIDS2017, collected on AWS infra.",
    ),
    "UNSW-NB15": DatasetSchema(
        name="UNSW-NB15",
        label_column="label",  # binary: 0 normal, 1 attack; 'attack_cat' has 9 categories
        categorical_columns=["proto", "service", "state"],
        drop_columns=["id"],
        binary_positive_labels=[1],
        notes=(
            "49 features. 'label' is binary (0/1); 'attack_cat' has 9 "
            "categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, "
            "Generic, Reconnaissance, Shellcode, Worms)."
        ),
    ),
    "TON_IoT": DatasetSchema(
        name="TON_IoT",
        label_column="label",
        categorical_columns=["proto", "service", "conn_state"],
        drop_columns=["ts", "src_ip", "dst_ip"],
        binary_positive_labels=[1],
        notes="Telemetry from IoT/IIoT sensors + network flows; 'type' gives attack subtype.",
    ),
    "Bot-IoT": DatasetSchema(
        name="Bot-IoT",
        label_column="attack",
        categorical_columns=["proto", "state"],
        drop_columns=["pkSeqID", "saddr", "daddr"],
        binary_positive_labels=[1],
        notes="Botnet traffic from simulated IoT network; 'category'/'subcategory' give attack types.",
    ),
}


def detect_schema(columns: list) -> Optional[DatasetSchema]:
    """Best-effort detection of which known schema a column list matches.

    Returns the DatasetSchema with the highest overlap of label-column /
    categorical-column names, or None if no reasonable match is found
    (caller should fall back to a generic auto-profiling routine).
    """
    cols_lower = {c.lower() for c in columns}
    best, best_score = None, 0
    for schema in SCHEMAS.values():
        score = 0
        if schema.label_column.lower() in cols_lower:
            score += 3
        score += sum(1 for c in schema.categorical_columns if c.lower() in cols_lower)
        if score > best_score:
            best, best_score = schema, score
    return best
