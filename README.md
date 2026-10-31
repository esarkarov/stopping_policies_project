# Comparing Stopping Policies for Search Agents

Research project by Elvin Sarkarov and Khazar Mammadov (UFAZ).
Supervisor: Dr. Yusif Ibrahimov.

A search agent answers multi-hop questions (HotpotQA, MuSiQue-Ans) with a small open-weight model (Qwen3.5-4B via Ollama) and BM25 retrieval. The agent always runs five search steps and saves every step. Stopping policies are compared afterwards on the saved trajectories:

- **P0** fixed budget, k = 1 to 5 searches
- **P1** self-decided: stop when the model says it has enough evidence
- **P2** answer stability: stop when the draft answer does not change between two searches
- **P3** sub-question coverage: stop when all sub-questions are covered
- **Oracle** stops at the first correct draft (reference only, uses gold answers)

Metrics: EM and F1 (F1 is the main metric), average searches and tokens per question, error categories, and paired bootstrap 95% confidence intervals. Adaptive policies are also compared with a fixed budget of equal average cost, and with a shuffled-stop baseline that tests whether a stopping signal carries information.

## Reproducing the results

Each dataset has two question sets, run with the same agent code (prompts, retriever and model settings frozen at git tag `freeze-v1`):

- **main**: 300 questions, run at `freeze-v1`. The policies P0 to P3 and the oracle were fixed before these runs.
- **confirm**: 400 further questions, disjoint from main and the pilot, run later (tag `confirm-v2`) with unchanged agent code. They are the held-out set for the post-hoc variants declared in `docs/variants_preregistration.md`. The frozen policies are also pooled over main + confirm (700 questions per dataset).

The saved trajectories are in `results/trajectories/`, so every table and figure can be reproduced without running the model. Use the latest commit (the evaluation and analysis code is newer than `freeze-v1`; the agent code is not):

```bash
git clone https://github.com/esarkarov/stopping_policies_project.git
cd stopping_policies_project
git diff --stat freeze-v1 HEAD -- src/agent.py src/prompts.py src/llm.py src/retriever.py   # prints nothing
docker compose build
```

Create the processed questions first with steps 2 and 3 under Setup (no model needed). Then, for `<d>` in `hotpotqa` and `musique`:

```bash
docker compose run --rm --no-deps app python scripts/run_eval.py --dataset <d> --split main
docker compose run --rm --no-deps app python scripts/run_eval.py --dataset <d> --split confirm
docker compose run --rm --no-deps app python scripts/run_analysis.py --dataset <d> --split main
docker compose run --rm --no-deps app python scripts/run_analysis.py --dataset <d> --split confirm
docker compose run --rm --no-deps app python scripts/run_variants.py --dataset <d> --split confirm
docker compose run --rm --no-deps app python scripts/run_pooled.py --dataset <d>
```

To reproduce only the first experiment, check out `freeze-v1` and run the `run_eval.py` and `run_analysis.py` lines with `--split main`.

Generating new trajectories needs the model. All trajectories were generated on a Google Colab T4 GPU (Ollama 0.40.5, no Docker) because the local machines were too slow. Different hardware can give slightly different greedy outputs, so new runs may not match the saved trajectories exactly. `notebooks/run_on_colab.ipynb` contains the notebook used. The confirm runs (and any later run) write a manifest (git commit, Ollama version, model digest, hashes of the agent files) to `results/trajectories/<name>.manifest.jsonl`.

Results and discussion draft: `docs/results_section_draft.md`.

## Requirements

- Docker Desktop (Windows/macOS) or Docker Engine with Compose (Linux)
- Internet access for the first run (Docker images, datasets, and the model if you run the agent)
- Optional: an NVIDIA GPU with recent drivers (much faster for the agent)

Nothing else has to be installed. Python and all packages (pinned in `requirements.txt`) live inside the Docker image.

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

| Dataset  | Dev questions | Corpus passages | Pilot | Main | Confirm |
| -------- | ------------- | --------------- | ----- | ---- | ------- |
| HotpotQA | 7405          | 66635           | 30    | 300  | 400     |
| MuSiQue  | 2417          | 21100           | 30    | 300  | 400     |

The data is not stored in git. It is created in `data/raw/` (downloads) and `data/processed/` (questions and corpora).

If the MuSiQue download fails (Google Drive limit), download the zip manually from
https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view,
save it as `data/raw/musique_data_v1.0.zip` and run step 2 again.

## Check the model speed

The first run downloads the model (`qwen3.5:4b`, several GB, only once).

```bash
docker compose run --rm app python scripts/speed_test.py
```

With an NVIDIA GPU, add the GPU file to every command that uses the model:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm app python scripts/speed_test.py
```

To use a smaller model:

```bash
MODEL=qwen3.5:2b docker compose run --rm app python scripts/speed_test.py
```

## Run the agent

```bash
# first 3 questions of the pilot (quick check)
docker compose run --rm app python scripts/run_agent.py --dataset hotpotqa --split pilot --limit 3

# full pilot, main or confirm run
docker compose run --rm app python scripts/run_agent.py --dataset hotpotqa --split pilot
docker compose run --rm app python scripts/run_agent.py --dataset musique --split main
docker compose run --rm app python scripts/run_agent.py --dataset musique --split confirm
```

Trajectories are written to `results/trajectories/<dataset>_<split>.jsonl`, one question per line. A stopped run continues where it stopped, and model answers are cached in `results/cache/` (not in git). If a trajectory file for the split already exists, the finished questions are skipped, so move or delete it to rerun from scratch.

The agent never sees the output of the stopping checks (no-feedback rule). A repeated search query is retried once with a different-query prompt (`retry_query_prompt`); these retry calls are counted in the cost of every policy.

## Evaluate the stopping policies

No model is needed. The policies are applied to the saved trajectories.

```bash
docker compose run --rm --no-deps app python scripts/run_eval.py --dataset hotpotqa --split main
docker compose run --rm --no-deps app python scripts/run_eval.py --dataset musique --split main
```

Use `--split pilot` or `--split confirm` for the other sets. Outputs in `results/tables/`:

- `<name>_per_question.csv`: outcome of every policy for every question
- `<name>_summary.csv`: EM, F1, searches, calls, tokens and gold recall with bootstrap 95% CIs
- `<name>_errors.csv`: stopping error categories (correct, over-searching, premature stop, not caused by stopping)
- `<name>_vs_fixed.csv`: paired differences of P1, P2, P3 against fixed budgets k = 1 to 5

Plots in `results/figures/`: F1 and EM against average searches and against average tokens.

## Analyse the results

```bash
docker compose run --rm --no-deps app python scripts/run_analysis.py --dataset hotpotqa --split main
docker compose run --rm --no-deps app python scripts/run_analysis.py --dataset musique --split main
```

Additional outputs in `results/tables/`:

- `<name>_stop_steps.csv`: at which search each policy stops
- `<name>_failures.csv`: outcome at the stop (correct, near miss, reading failure, retrieval failure)
- `<name>_examples.csv`: over-searching and premature-stop examples with queries and drafts
- `<name>_by_hops.csv`: results by number of hops (MuSiQue only)

- `<name>_signal.csv`: F1 and EM at the steps where each signal fires and where it does not
- `<name>_shuffled.csv`: real stops against the same stops shuffled across questions (does the signal carry information?)
- `<name>_vs_curve.csv`: P1, P2, P3 against a fixed policy of equal average cost (searches, tokens, and tokens if the check shared the answer call)
- `<name>_p1_threshold.csv`: P1 with P(yes) >= tau (needs `<name>.p1_logprobs.jsonl`, see below)

Figure: `results/figures/<name>_stop_steps.png`.

## Variants, thresholds and pooled results

The variants (`P1_min2`, `P3_min2`, `P1_c2`, `P1&P3`, `P3&P2`) and the P(yes) thresholds were defined after looking at the main set. They are declared in `docs/variants_preregistration.md` and judged on the confirm split only:

```bash
docker compose run --rm --no-deps app python scripts/run_variants.py --dataset hotpotqa --split confirm
```

It writes `<name>_variants_summary.csv`, `_variants_vs_fixed.csv`, `_variants_vs_base.csv` and `_variants_vs_curve.csv` (95% and 99% intervals).

The P(yes) values come from replaying the P1 check on the saved passages. This needs the model (Ollama with logprobs) and writes `results/trajectories/<name>.p1_logprobs.jsonl`; the saved files are in the repository. Run it from the Colab notebook or locally:

```bash
python scripts/run_p1_logprobs.py --dataset hotpotqa --split confirm
```

The frozen policies (P0, P1, P2, P3, oracle) are pooled over main and confirm with `scripts/run_pooled.py`, which writes `<dataset>_pooled_summary.csv`, `_errors.csv`, `_vs_fixed.csv`, `_vs_curve.csv` and `_shuffled.csv`. Variants and thresholds are never pooled.

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

| Path                                              | Purpose                                                                     |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| `configs/default.yaml`                            | Experiment settings                                                         |
| `src/data.py`, `src/hotpot_convert.py`            | Loading, sampling and corpus building                                       |
| `src/retriever.py`                                | BM25 retrieval                                                              |
| `src/llm.py`, `src/prompts.py`                    | Model wrapper with cache, prompts                                           |
| `src/agent.py`                                    | Five-step search agent                                                      |
| `src/policies.py`, `src/metrics.py`               | Stopping rules, cost accounting, EM/F1                                      |
| `src/evaluate.py`, `src/stats.py`, `src/plots.py` | Evaluation tables, bootstrap CIs, figures                                   |
| `src/analysis.py`, `src/pooled.py`                | Stop steps, failure breakdown, signal quality, matched-cost comparison      |
| `src/replay.py`, `src/manifest.py`                | P(yes) replay of the P1 check, run manifest                                 |
| `scripts/`                                        | Entry points (download, prepare, speed test, run agent, evaluate, analysis) |
| `notebooks/`                                      | Colab notebook used for the main and confirm runs and the P(yes) replay     |
| `tests/`                                          | Unit tests                                                                  |
| `docs/`                                           | Results section draft, declaration of the post-hoc variants                 |
| `data/`                                           | Datasets and corpora (downloaded, not in git)                               |
| `results/`                                        | Saved trajectories, tables and plots (cache not in git)                     |

## Use of AI tools

Claude (Anthropic) was used as a coding and writing assistant during this project: for drafting and reviewing code, tests and Docker setup, for analysis code, and for drafting text. All experiments were run by the authors, and the reported numbers come from the saved trajectories and the tables produced by the scripts above. The authors reviewed and are responsible for the code and the text.
