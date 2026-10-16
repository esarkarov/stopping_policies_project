# Comparing Stopping Policies for Search Agents

Research project by Elvin Sarkarov and Khazar Mammadov (UFAZ).
Supervisor: Dr. Yusif Ibrahimov.

## Requirements

- Docker Desktop (Windows/macOS) or Docker Engine with Compose (Linux)
- Optional: an NVIDIA GPU with recent drivers (much faster)

## Quick start

```bash
# 1. Build the image and download the model (first time only, several GB)
docker compose build
docker compose run --rm app python scripts/speed_test.py
```

With an NVIDIA GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm app python scripts/speed_test.py
```

To use a different model size:

```bash
MODEL=qwen3.5:2b docker compose run --rm app python scripts/speed_test.py
```

## Project structure

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Experiment settings |
| `src/` | Agent, retrieval, stopping policies, evaluation |
| `scripts/` | Entry points (speed test, runs, evaluation) |
| `data/` | Datasets and corpora (downloaded, not in git) |
| `results/` | Saved trajectories, tables and plots |
