"""
Synthetic, schema-compatible network-flow traffic generator.

WHY THIS EXISTS
----------------
CognitiveCyber's notebooks must run end-to-end without any manual step. The
real datasets it targets (CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, TON_IoT,
Bot-IoT) are multi-gigabyte files distributed by third-party institutions
(UNB, ACS/UNSW Canberra Cyber) and are NOT reachable from this execution
environment's network allowlist. Rather than fabricate fake "results" on
data we never touched, this module generates a synthetic dataset with the
same *statistical shape and column semantics* as UNSW-NB15-style flow
records (duration, byte/packet counts, rates, TTLs, connection state, a
handful of categorical fields) plus injected attack subpopulations with
realistic separability.

This lets every downstream step (preprocessing, baselines, UEBA, KG demo,
evaluation, figures, tables, statistics) execute for real and produce real
numbers -- on synthetic data, honestly labeled as such everywhere it
appears (column `is_synthetic=True` in metadata, filenames prefixed
`synthetic_`).

USING A REAL DATASET INSTEAD
-----------------------------
Drop a real CICIDS2017 / UNSW-NB15 / TON_IoT / Bot-IoT CSV into
`datasets/raw/` and use `cognitivecyber.data.loaders.load_dataset(path=...)`
instead of `generate_synthetic_flows`. The preprocessing pipeline
(`cognitivecyber.data.preprocessing`) is schema-agnostic and will
auto-detect columns either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATTACK_CATEGORIES = [
    "Normal",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Exploits",
    "Reconnaissance",
    "Botnet",
    "Backdoor",
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "dns", "ftp", "ssh", "smtp", "-", "dhcp"]
CONN_STATES = ["FIN", "CON", "INT", "REQ", "RST"]


def _sample_categorical(rng: np.random.Generator, choices, n, p=None):
    return rng.choice(choices, size=n, p=p)


def generate_synthetic_flows(
    n_samples: int = 60_000,
    attack_ratio: float = 0.35,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a UNSW-NB15-style synthetic flow dataset.

    Parameters
    ----------
    n_samples : total number of flow records to generate.
    attack_ratio : fraction of samples labeled as an attack category.
    random_state : seed for full reproducibility.

    Returns
    -------
    DataFrame with numeric flow features, categorical fields (proto,
    service, state), a multi-class `attack_cat` column, and a binary
    `label` column (0 = normal, 1 = attack), plus an `is_synthetic` flag.
    """
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    records = []

    # --- Normal traffic: tight, low-variance distributions ---
    normal = pd.DataFrame(
        {
            "dur": rng.exponential(0.5, n_normal),
            "sbytes": rng.lognormal(6.0, 1.0, n_normal),
            "dbytes": rng.lognormal(5.5, 1.0, n_normal),
            "spkts": rng.poisson(8, n_normal).astype(float),
            "dpkts": rng.poisson(7, n_normal).astype(float),
            "sttl": rng.integers(60, 65, n_normal).astype(float),
            "dttl": rng.integers(60, 65, n_normal).astype(float),
            "sload": rng.exponential(2000, n_normal),
            "dload": rng.exponential(1800, n_normal),
            "swin": rng.integers(250, 260, n_normal).astype(float),
            "dwin": rng.integers(250, 260, n_normal).astype(float),
            "smean": rng.normal(300, 40, n_normal).clip(0),
            "dmean": rng.normal(280, 40, n_normal).clip(0),
            "ct_srv_src": rng.poisson(3, n_normal).astype(float),
            "ct_dst_ltm": rng.poisson(3, n_normal).astype(float),
            "ct_state_ttl": rng.poisson(2, n_normal).astype(float),
            "proto": _sample_categorical(rng, PROTOCOLS, n_normal, p=[0.75, 0.2, 0.05]),
            "service": _sample_categorical(
                rng, SERVICES, n_normal, p=[0.35, 0.2, 0.05, 0.1, 0.05, 0.2, 0.05]
            ),
            "state": _sample_categorical(rng, CONN_STATES, n_normal, p=[0.5, 0.3, 0.1, 0.05, 0.05]),
        }
    )
    normal["attack_cat"] = "Normal"

    # --- Attack traffic: category-specific shifted distributions ---
    attack_cats = ATTACK_CATEGORIES[1:]
    per_cat = n_attack // len(attack_cats)
    attack_frames = []
    # rough separability parameters per category (mean shift, spread, dominant proto/service)
    cat_params = {
        "DoS": dict(dur=0.02, sbytes=9.5, spkts_lam=400, ttl=(30, 40), proto_p=[0.85, 0.1, 0.05]),
        "DDoS": dict(dur=0.01, sbytes=10.0, spkts_lam=900, ttl=(20, 30), proto_p=[0.6, 0.35, 0.05]),
        "PortScan": dict(dur=0.005, sbytes=4.0, spkts_lam=2, ttl=(60, 65), proto_p=[0.9, 0.05, 0.05]),
        "BruteForce": dict(dur=0.3, sbytes=7.0, spkts_lam=30, ttl=(60, 65), proto_p=[0.95, 0.02, 0.03]),
        "Exploits": dict(dur=0.4, sbytes=8.5, spkts_lam=25, ttl=(55, 64), proto_p=[0.8, 0.15, 0.05]),
        "Reconnaissance": dict(dur=0.05, sbytes=5.0, spkts_lam=5, ttl=(58, 64), proto_p=[0.5, 0.3, 0.2]),
        "Botnet": dict(dur=1.2, sbytes=7.5, spkts_lam=15, ttl=(45, 55), proto_p=[0.7, 0.25, 0.05]),
        "Backdoor": dict(dur=0.8, sbytes=6.5, spkts_lam=10, ttl=(50, 60), proto_p=[0.9, 0.05, 0.05]),
    }

    for cat in attack_cats:
        p = cat_params[cat]
        n = per_cat
        df_cat = pd.DataFrame(
            {
                "dur": rng.exponential(p["dur"], n),
                "sbytes": rng.lognormal(p["sbytes"], 1.3, n),
                "dbytes": rng.lognormal(p["sbytes"] - 1.0, 1.3, n),
                "spkts": rng.poisson(p["spkts_lam"], n).astype(float),
                "dpkts": rng.poisson(max(p["spkts_lam"] // 3, 1), n).astype(float),
                "sttl": rng.integers(p["ttl"][0], p["ttl"][1], n).astype(float),
                "dttl": rng.integers(p["ttl"][0], p["ttl"][1], n).astype(float),
                "sload": rng.exponential(6000, n),
                "dload": rng.exponential(1200, n),
                "swin": rng.integers(0, 260, n).astype(float),
                "dwin": rng.integers(0, 260, n).astype(float),
                "smean": rng.normal(150, 80, n).clip(0),
                "dmean": rng.normal(120, 80, n).clip(0),
                "ct_srv_src": rng.poisson(12, n).astype(float),
                "ct_dst_ltm": rng.poisson(12, n).astype(float),
                "ct_state_ttl": rng.poisson(6, n).astype(float),
                "proto": _sample_categorical(rng, PROTOCOLS, n, p=p["proto_p"]),
                "service": _sample_categorical(
                    rng, SERVICES, n, p=[0.3, 0.15, 0.1, 0.15, 0.1, 0.15, 0.05]
                ),
                "state": _sample_categorical(rng, CONN_STATES, n, p=[0.2, 0.5, 0.15, 0.1, 0.05]),
            }
        )
        df_cat["attack_cat"] = cat
        attack_frames.append(df_cat)

    attack = pd.concat(attack_frames, ignore_index=True)
    df = pd.concat([normal, attack], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    df["is_synthetic"] = True

    # Inject a small amount of missingness + noise, mirroring real NIDS exports
    for col in ["sload", "dload", "smean", "dmean"]:
        mask = rng.random(len(df)) < 0.01
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    data = generate_synthetic_flows()
    print(data.shape)
    print(data["attack_cat"].value_counts())
