# Data Provenance

## Why synthetic data ships by default

CognitiveCyber targets five public network-security datasets:

| Dataset | Publisher | Approx. size | Access |
|---|---|---|---|
| CICIDS2017 | Canadian Institute for Cybersecurity (UNB) | ~6 GB | Request form on UNB site |
| CSE-CIC-IDS2018 | UNB / CSE | ~16 GB (AWS-hosted) | Public S3 bucket / UNB site |
| UNSW-NB15 | UNSW Canberra Cyber | ~2 GB | Public download on UNSW research site |
| TON_IoT | UNSW Canberra Cyber (IoT Lab) | Multi-GB, multiple files | Public download / request form |
| Bot-IoT | UNSW Canberra Cyber | ~70 GB (full), ~2 GB (5%-subset) | Public download |

None of these hosts are reachable from this repository's default execution
sandbox (its network allowlist covers package registries and GitHub only,
not university data-hosting domains). Rather than fabricate benchmark
numbers against data we never touched, `cognitivecyber.data.synthetic`
generates a dataset that:

- Uses the **same column semantics** as UNSW-NB15-style flow records
  (duration, byte/packet counts, rates, TTLs, window sizes, categorical
  `proto`/`service`/`state` fields).
- Injects **9 attack categories** (DoS, DDoS, PortScan, BruteForce,
  Exploits, Reconnaissance, Botnet, Backdoor, plus Normal) with
  category-specific, but overlapping, feature distributions -- separable
  enough for classical ML to learn, but not trivially so for every model
  (see `outputs/run_manifest.json` and `tables/unsupervised_baseline_results.csv`
  for realistic anomaly-detector variance).
- Is clearly flagged everywhere it appears: `is_synthetic=True` column,
  `synthetic_flows.csv` filename, and explicit notebook markdown cells.

## Using a real dataset

1. Download one of the datasets above to `datasets/raw/<file>.csv`
   (or `.parquet`).
2. In `notebooks/01_Data_Preprocessing.ipynb`, set:
   ```python
   DATASET_PATH = REPO_ROOT / "datasets" / "raw" / "<file>.csv"
   DATASET_NAME = "UNSW-NB15"  # or None to auto-detect
   ```
3. Re-run notebooks 01 onward. `cognitivecyber.data.loaders.load_dataset`
   auto-detects encoding, the label column, categorical columns, and
   applies the correct binary-label convention (e.g. CICIDS2017's
   `Label == "BENIGN"` vs. UNSW-NB15's `label ∈ {0,1}`) via the schema
   registry in `cognitivecyber/data/schemas.py`.
4. No other pipeline code needs to change -- preprocessing, baselines,
   evaluation, figure, and table generation are schema-agnostic.

## Reported numbers are honest about their source

Every table and figure in this repository is generated from an actual
execution against the dataset in use (synthetic by default). Nothing in
`tables/`, `figures/`, or `outputs/run_manifest.json` is hand-written or
estimated. If you swap in a real dataset, expect metrics to look
different -- generally lower and noisier than the synthetic-data results,
since real traffic has class imbalance, label noise, and overlapping
attack/benign distributions that the synthetic generator does not fully
replicate.
