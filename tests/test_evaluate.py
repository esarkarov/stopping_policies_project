import json

import numpy as np
import pytest

from src.evaluate import classify, compare_to_fixed, error_table, evaluate, evaluate_trajectory, load_trajectories, summarize
from src.stats import bootstrap_ci, paired_diff_ci
from tests.helpers import make_traj


def row(rows, policy):
    return next(r for r in rows if r["policy"] == policy)


def test_classify_categories():
    assert classify(2, [False, True, False]) == "correct"
    assert classify(3, [False, True, False]) == "over-searching"
    assert classify(1, [False, False, True]) == "premature stop"
    assert classify(2, [False, False, False]) == "not caused by stopping"


def test_evaluate_trajectory_scores_the_draft_at_the_stop_step():
    traj = make_traj(["Rome", "Paris", "Paris", "Berlin", "Berlin"], p1=[False, True, True, True, True])
    rows = evaluate_trajectory(traj)
    p1 = row(rows, "P1")
    assert p1["stop_step"] == 2 and p1["em"] == 1.0 and p1["category"] == "correct" and not p1["late_stop"]
    k4 = row(rows, "P0_k4")
    assert k4["em"] == 0.0 and k4["category"] == "over-searching"
    k3 = row(rows, "P0_k3")
    assert k3["category"] == "correct" and k3["late_stop"]
    assert row(rows, "P0_k1")["category"] == "premature stop"
    assert row(rows, "Oracle")["stop_step"] == 2


def test_aliases_count_as_correct():
    traj = make_traj(["NYC"] * 5, answer="New York City", aliases=["NYC"])
    assert row(evaluate_trajectory(traj), "P0_k1")["em"] == 1.0


def test_gold_recall_at_stop_step():
    traj = make_traj(["a"] * 5, gold=("g1", "g2"), retrieved=[["g1", "x"], ["y", "z"], ["g2", "w"], ["u", "v"], ["s", "t"]])
    rows = evaluate_trajectory(traj)
    assert row(rows, "P0_k1")["gold_recall"] == 0.5 and not row(rows, "P0_k1")["all_gold"]
    assert row(rows, "P0_k3")["gold_recall"] == 1.0 and row(rows, "P0_k3")["all_gold"]


def test_tables_and_comparison_shapes():
    trajs = [make_traj(["Paris"] * 5, qid=f"q{i}", p1=[True] * 5) for i in range(6)]
    df = evaluate(trajs)
    summary = summarize(df, n_boot=200)
    assert len(summary) == 9 and set(summary["n"]) == {6}
    assert summary.set_index("policy").loc["P0_k3", "searches"] == 3
    errors = error_table(df)
    assert errors.set_index("policy").loc["P0_k1", "correct"] == 6
    versus = compare_to_fixed(df, n_boot=200)
    assert len(versus) == 15 and set(versus["policy"]) == {"P1", "P2", "P3"}
    p1_vs_k3 = versus[(versus["policy"] == "P1") & (versus["versus"] == "P0_k3")].iloc[0]
    assert p1_vs_k3["d_searches"] == -2 and p1_vs_k3["d_f1"] == 0
    assert p1_vs_k3["d_tokens"] == (11 + 31 + 21) - (3 * 11 + 21) and "d_tokens_lo" in versus.columns


def test_bootstrap_is_seeded_and_brackets_the_mean():
    values = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=float)
    a, b = bootstrap_ci(values, 2000, seed=1), bootstrap_ci(values, 2000, seed=1)
    assert a == b and a[1] <= a[0] <= a[2]
    assert paired_diff_ci(values, values, 500) == (0.0, 0.0, 0.0)


def test_loader_drops_duplicates_and_reads_aliases(tmp_path):
    first = make_traj(["a"] * 5, qid="q1")
    second = make_traj(["b"] * 5, qid="q1")
    first.pop("answer_aliases")
    second.pop("answer_aliases")
    other = make_traj(["c"] * 5, qid="q2")
    other["answer_aliases"] = ["Kept"]
    traj_path, q_path = tmp_path / "t.jsonl", tmp_path / "q.jsonl"
    traj_path.write_text("\n".join(json.dumps(t) for t in (first, second, other)))
    q_path.write_text(json.dumps({"id": "q1", "answer_aliases": ["Alias"]}) + "\n" + json.dumps({"id": "q2", "answer_aliases": ["Other"]}))
    with pytest.warns(UserWarning, match="dropped 1 duplicate"):
        loaded = load_trajectories(traj_path, q_path)
    assert [t["id"] for t in loaded] == ["q1", "q2"]
    assert loaded[0]["steps"][0]["draft_answer"] == "a" and loaded[0]["answer_aliases"] == ["Alias"]
    assert loaded[1]["answer_aliases"] == ["Kept"]
