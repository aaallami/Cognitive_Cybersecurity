# Installation

## Option A — local Python environment

```bash
git clone https://github.com/your-org/CognitiveCyber.git
cd CognitiveCyber
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -v          # confirm the executable core passes
jupyter lab               # open notebooks/01_Data_Preprocessing.ipynb
```

Optional extras (see docs/architecture.md for what each unlocks):

```bash
pip install -r requirements-deep.txt     # CNN/LSTM/Transformer/RL (needs GPU for realistic runtimes)
pip install -r requirements-llm.txt      # LLM/RAG components (needs model weights or API key)
pip install -r requirements-agents.txt   # multi-agent orchestration (needs LLM backend)
```

## Option B — Docker

```bash
docker compose build
docker compose up
# Jupyter Lab at http://localhost:8888
```

To also start the optional MLflow tracking server:

```bash
docker compose --profile mlflow up
# MLflow UI at http://localhost:5000
```

## Option C — Google Colab

Each notebook under `notebooks/` is self-contained: the first code cell
resolves the repo root and installs `src/` onto `sys.path`. To run in
Colab, upload the whole `CognitiveCyber/` folder (or clone it in a cell
with `!git clone ...`) and run `!pip install -r requirements.txt` before
executing the notebook cells.

## Running with a real dataset

See `docs/data_provenance.md` for supported datasets and exactly which
line to change in `01_Data_Preprocessing.ipynb`.
