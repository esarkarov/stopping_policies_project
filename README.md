# Comparing Stopping Policies for Search Agents

Research project by Elvin Sarkarov and Khazar Mammadov (UFAZ).
Supervisor: Dr. Yusif Ibrahimov.

A search agent answers multi-hop questions (HotpotQA, MuSiQue) with a small open-weight model (Qwen3.5-4B via Ollama) and BM25 retrieval. The agent always runs five search steps and saves every step. Stopping policies are compared afterwards on the saved trajectories.

## Requirements

- Docker Desktop (Windows/macOS) or Docker Engine with Compose (Linux)
- Internet access for the first run (Docker images, model of several GB, datasets)
- Optional: an NVIDIA GPU with recent drivers (much faster)

Nothing else has to be installed. Python and all packages live inside the Docker image.

## Setup

Run these commands from the project folder, one after another.

```bash
# 1. Build the image
docker compose build

# 2. Download the development sets (about 300 MB, no model needed)
docker compose run --rm --no-deps app python scripts/download_data.py

# 3. Sample the questions and build one corpus per dataset
docker compose run --rm --no-deps app python scripts/prepare_data.py
```

Step 3 prints these counts. Other machines should print the same numbers (the sampling uses a fixed seed):

| Dataset  | Dev questions | Corpus passages | Pilot | Main |
| -------- | ------------- | --------------- | ----- | ---- |
| HotpotQA | 7405          | 66635           | 30    | 300  |
| MuSiQue  | 2417          | 21100           | 30    | 300  |

The data is not stored in git. It is created in `data/raw/` (downloads) and `data/processed/` (questions and corpora).

If the MuSiQue download fails (Google Drive limit), download the zip manually from
https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view,
save it as `data/raw/musique_data_v1.0.zip` and run step 2 again.

## Run the agent

```bash
# first 3 questions of the pilot (quick check)
docker compose run --rm app python scripts/run_agent.py --dataset hotpotqa --split pilot --limit 3

# full pilot or main run
docker compose run --rm app python scripts/run_agent.py --dataset hotpotqa --split pilot
docker compose run --rm app python scripts/run_agent.py --dataset musique --split main
```

Trajectories are written to `results/trajectories/<dataset>_<split>.jsonl`, one question per line. A stopped run continues where it stopped, and model answers are cached in `results/cache/`.

## Tests

```bash
docker compose run --rm --no-deps app python -m pytest tests -q
```

After changing code, rebuild the image (`docker compose build`) or add `--build` to the run command.

## Stop everything

```bash
docker compose down
```

This keeps the downloaded model. Add `-v` only to delete it as well.

## Project structure

| Path                                   | Purpose                                                 |
| -------------------------------------- | ------------------------------------------------------- |
| `configs/default.yaml`                 | Experiment settings                                     |
| `src/data.py`, `src/hotpot_convert.py` | Loading, sampling and corpus building                   |
| `src/retriever.py`                     | BM25 retrieval                                          |
| `src/llm.py`, `src/prompts.py`         | Model wrapper with cache, prompts                       |
| `src/agent.py`                         | Five-step search agent                                  |
| `scripts/`                             | Entry points (download, prepare, speed test, run agent) |
| `tests/`                               | Unit tests                                              |
| `data/`                                | Datasets and corpora (downloaded, not in git)           |
| `results/`                             | Saved trajectories, tables and plots                    |
