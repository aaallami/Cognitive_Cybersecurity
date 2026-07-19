# Architecture & Implementation Status

CognitiveCyber implements the multi-layer architecture from "Towards
Cognitive Cybersecurity: A Human-AI Collaborative Architecture for
Adaptive National Cyber Defense." This document tracks, per component,
whether it is **implemented + executed** in this repository, **implemented
but not executed** (documented interface, needs infra this sandbox lacks),
or **planned**.

| Layer | Component | Status | Notes |
|---|---|---|---|
| Data | Ingestion + schema auto-detection | ✅ Implemented + executed | `src/cognitivecyber/data/` ; 5 dataset schemas registered |
| Data | Synthetic schema-compatible generator | ✅ Implemented + executed | Used because real datasets are unreachable from this sandbox |
| Baselines | RF, DT, XGBoost, LightGBM, CatBoost, SVM, MLP | ✅ Implemented + executed | `notebooks/02_Baseline_Models.ipynb`, real metrics |
| Baselines | IsolationForest, LOF, One-Class SVM | ✅ Implemented + executed | Same notebook, unsupervised track |
| Baselines | CNN, LSTM, Transformer, Autoencoder, Deep SVDD | 🟡 Implemented, not executed | `src/cognitivecyber/models/deep_stubs.py`; needs windowed sequence data + GPU |
| UEBA | Behavior/entity profiling, anomaly scoring | 🟡 Partial | See `docs/experiment_guide.md` for the planned notebook 03 scope |
| Knowledge Graph | MITRE ATT&CK / CVE / CWE / CAPEC / STIX mapping | ⬜ Planned | Requires live feeds (MITRE ATT&CK STIX bundles, NVD API) not fetched in this pass |
| LLM/RAG | Incident summarization, MITRE mapping, hallucination mitigation | ⬜ Planned interface only | Requires model weights (Llama/Qwen/Gemma/DeepSeek) or hosted API credentials |
| Digital Twin | Enterprise/attack/defense simulators | ⬜ Planned | Design doc + interface only in this pass |
| RL | PPO / DQN / A2C / SAC adaptive defense | ⬜ Planned interface only | Requires `requirements-deep.txt` + a simulation environment (Gymnasium) |
| Multi-Agent | LangGraph / CrewAI / OpenAI Agents SDK orchestration | ⬜ Planned interface only | Requires LLM backend + API keys |
| Human-AI Collaboration | Trust score, approval workflow, adaptive authority | ⬜ Planned | Depends on the above layers being live |
| Experiments/Stats | Benchmark vs. SIEM/UEBA/SOAR/XDR, significance tests | 🟡 Partial | Baseline-vs-baseline comparison executed; competitor-tool benchmarking needs those tools installed/licensed |

**Legend:** ✅ executed and verifiable in this repo · 🟡 partially implemented
· ⬜ interface/design only, not executed.

## Why the gaps are documented rather than filled with placeholder output

An IEEE-Transactions-adjacent artifact needs every reported number to be
traceable to an actual run. Where a component genuinely requires
infrastructure this sandbox doesn't have (GPU time, live threat-intel
feeds, LLM weights/API keys, licensed competitor tools), this repository:

1. Implements the real interface/contract (so it's a genuine engineering
   deliverable, not vaporware),
2. Documents exactly what's needed to execute it for real, and
3. Does **not** report numbers for it in `tables/` or `figures/`.

## Extension roadmap

- **Knowledge Graph (03-04):** wire `networkx` + a local MITRE ATT&CK STIX
  bundle (download once, cache under `datasets/threat_intel/`) into an
  attack-graph builder; this is fully offline-capable once the bundle is
  cached and doesn't require this sandbox's restricted network.
- **LLM/RAG (05):** the interface in `src/cognitivecyber/llm/` (to be added)
  should accept any Anthropic/OpenAI-compatible client; swap in
  `anthropic.Anthropic()` with a real API key to run end-to-end.
- **Digital Twin (06):** start with a `networkx`-based topology simulator
  and rule-based attack propagation before adding a full discrete-event
  simulation engine (e.g. `simpy`).
- **RL (07):** `requirements-deep.txt` + `gymnasium` custom environment
  wrapping the digital twin; PPO/DQN via `stable-baselines3`.
- **Multi-Agent (08):** LangGraph graph-of-agents wired to the KG + RAG +
  digital-twin tools once those layers exist.
