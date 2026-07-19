# CognitiveCyber

**An Open Cognitive Cybersecurity Research Platform**

Companion implementation for *"Towards Cognitive Cybersecurity: A Human-AI
Collaborative Architecture for Adaptive National Cyber Defense."*

---

## What's actually executed vs. planned

This repository is honest about scope. Two notebooks are **fully
implemented, executed, and tested end-to-end** in this environment; ten
more are **scoped planning notebooks** that document exactly what they'll
contain and what's needed to execute them for real. See
[`docs/architecture.md`](docs/architecture.md) for the full per-component
status table.

| # | Notebook | Status |
|---|---|---|
| 01 | Data Preprocessing | ✅ Executed |
| 02 | Baseline Models | ✅ Executed |
| 03 | UEBA | 🟡 Scoped (directly executable next — no external infra needed) |
| 04 | Knowledge Graph | 🟡 Scoped (needs a locally cached MITRE ATT&CK bundle) |
| 05 | LLM / RAG | 🟡 Scoped (needs LLM weights or API key) |
| 06 | Digital Twin | 🟡 Scoped (executable with networkx once written) |
| 07 | Reinforcement Learning | 🟡 Scoped (needs GPU + `requirements-deep.txt`) |
| 08 | Multi-Agent Framework | 🟡 Scoped (needs LLM backend from 05) |
| 09 | Human-AI Collaboration | 🟡 Scoped (needs 04-08 live) |
| 10 | Cross-System Experiments | 🟡 Scoped (needs licensed SIEM/UEBA/SOAR/XDR products) |
| 11 | Statistical Analysis | 🟡 Scoped (directly executable — just needs a multi-seed sweep of 02) |
| 12 | Paper Figure Assembly | 🟡 Scoped (directly executable against existing `figures/`) |

**Why not fabricate the rest?** Several components require infrastructure
this sandbox genuinely does not have: GPU time for deep sequence
models and RL, live threat-intel feeds, LLM API keys/weights, and
licensed commercial security products. Generating "results" for those
without running anything real would produce numbers nobody could trust —
the opposite of what an IEEE-Transactions-adjacent artifact needs. Instead,
every planning notebook documents the exact path to make it real (see
`docs/experiment_guide.md`).

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/ -v
jupyter lab notebooks/01_Data_Preprocessing.ipynb
```

Full installation options (Docker, Colab): [`docs/installation.md`](docs/installation.md).

## What notebooks 01-02 actually produce

Running them end-to-end (already done once in this repo; outputs are
committed under `figures/`, `tables/`, `outputs/`) trains and evaluates:

- **Supervised:** RandomForest, DecisionTree, XGBoost, LightGBM, CatBoost, SVM, MLP
- **Unsupervised:** IsolationForest, Local Outlier Factor (novelty), One-Class SVM

on a schema-compatible dataset (synthetic by default — see
[`docs/data_provenance.md`](docs/data_provenance.md) for why, and how to
swap in a real CICIDS2017/UNSW-NB15/CSE-CIC-IDS2018/TON_IoT/Bot-IoT file
with zero pipeline code changes).

Real, reproducible outputs from the last executed run:

- `outputs/run_manifest.json` — run configuration + best-model summary
- `tables/*.{csv,xlsx,md,tex}` — full metric tables (accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC, train/inference time, peak memory)
- `figures/*.{png,svg,pdf}` (300 dpi) — ROC curves, PR curves, confusion
  matrices, metric comparison bars, radar chart, computational-cost violin
  plot, score boxplots, correlation matrix, PCA/t-SNE embeddings, feature
  importance (RandomForest, XGBoost)

Re-run anytime with:

```bash
python experiments/run_baseline_benchmark.py
```

## Repository structure

```
CognitiveCyber/
├── README.md
├── requirements.txt              # core (data + classical ML) deps
├── requirements-deep.txt         # optional: CNN/LSTM/Transformer/RL (GPU)
├── requirements-llm.txt          # optional: LLM/RAG
├── requirements-agents.txt       # optional: multi-agent orchestration
├── pyproject.toml
├── Dockerfile / docker-compose.yml
├── LICENSE / CITATION.cff
├── configs/                      # Hydra-style YAML configs
├── datasets/                     # synthetic_flows.csv + datasets/raw/ (user-supplied real data)
├── docs/                         # installation, architecture, data provenance, experiment guide
├── notebooks/                    # 01-12, see status table above
├── src/cognitivecyber/
│   ├── data/                     # schemas, synthetic generator, loaders, preprocessing
│   ├── models/                   # baselines.py (executed), deep_stubs.py (interface only)
│   └── evaluation/               # metrics.py, figures.py, tables.py
├── tests/                        # 16 passing unit tests
├── experiments/                  # run_baseline_benchmark.py (backs notebooks 01-02)
├── figures/ tables/ reports/     # generated outputs (committed from the last run)
└── outputs/                      # run_manifest.json, logs, checkpoints
```

## Testing

```bash
pytest tests/ -v
```

16/16 tests passing, covering the data pipeline (schema detection, cleaning,
encoding, splitting, scaling) and the model/evaluation layer (baseline
construction, metric computation, results-table formatting).

## Contributing / extending

See [`docs/architecture.md`](docs/architecture.md) for the extension
roadmap (Knowledge Graph → LLM/RAG → Digital Twin → RL → Multi-Agent →
Human-AI Collaboration) and [`docs/experiment_guide.md`](docs/experiment_guide.md)
for how to turn a scoped planning notebook into an executed one.

## License

MIT — see [`LICENSE`](LICENSE).
