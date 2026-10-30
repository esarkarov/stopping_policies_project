from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import evaluate, load_trajectories, summarize

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["n", "em", "em_lo", "em_hi", "f1", "f1_lo", "f1_hi", "searches", "calls", "tokens", "gold_recall", "all_gold_rate"]


@pytest.mark.parametrize("dataset", ["hotpotqa", "musique"])
def test_committed_trajectories_reproduce_committed_summary(dataset):
    trajectories = ROOT / "results" / "trajectories" / f"{dataset}_main.jsonl"
    summary = ROOT / "results" / "tables" / f"{dataset}_main_summary.csv"
    if not (trajectories.exists() and summary.exists()):
        pytest.skip("saved main results are not available")
    expected = pd.read_csv(summary).set_index("policy")
    actual = summarize(evaluate(load_trajectories(trajectories)), n_boot=10000, seed=42).set_index("policy")
    assert list(actual.index) == list(expected.index)
    pd.testing.assert_frame_equal(actual[COLUMNS], expected[COLUMNS], check_exact=False, rtol=1e-9, atol=1e-12)
