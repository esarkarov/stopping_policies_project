import pandas as pd

from src.analysis import by_hops, failure_breakdown, failure_examples, hop_count, outcome, stop_distribution
from tests.helpers import make_traj


def frame(rows):
    base = {"dataset": "x", "searches": 1, "tokens": 100, "category": "correct", "all_gold": True}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_hop_count_reads_musique_ids_only():
    assert hop_count("2hop__1_2") == 2
    assert hop_count("4hop1__1_2_3_4") == 4
    assert hop_count("5a7326cd5542991f29ee2d11") is None


def test_by_hops_is_none_without_hop_ids():
    df = frame([{"id": "abc", "policy": "P1", "em": 1.0, "f1": 1.0, "stop_step": 1}])
    assert by_hops(df) is None


def test_by_hops_groups_questions():
    df = frame(
        [
            {"id": "2hop__1", "policy": "P1", "em": 1.0, "f1": 1.0, "stop_step": 1},
            {"id": "2hop__2", "policy": "P1", "em": 0.0, "f1": 0.0, "stop_step": 1},
            {"id": "3hop1__3", "policy": "P1", "em": 1.0, "f1": 1.0, "stop_step": 1},
        ]
    )
    table = by_hops(df).set_index("hops")
    assert table.loc[2, "n"] == 2 and table.loc[2, "f1"] == 0.5 and table.loc[3, "n"] == 1


def test_stop_distribution_counts_every_step():
    df = frame([{"id": str(i), "policy": "P1", "em": 0.0, "f1": 0.0, "stop_step": s} for i, s in enumerate([1, 1, 5])])
    row = stop_distribution(df, policies=("P1",)).iloc[0]
    assert (row["step_1"], row["step_2"], row["step_5"]) == (2, 0, 1)


def test_outcome_labels():
    assert outcome({"em": 1.0, "f1": 1.0, "all_gold": False}) == "correct"
    assert outcome({"em": 0.0, "f1": 0.8, "all_gold": False}) == "near miss"
    assert outcome({"em": 0.0, "f1": 0.0, "all_gold": True}) == "reading failure"
    assert outcome({"em": 0.0, "f1": 0.0, "all_gold": False}) == "retrieval failure"


def test_failure_breakdown_has_all_columns():
    df = frame([{"id": "a", "policy": "P1", "em": 0.0, "f1": 0.0, "stop_step": 1, "all_gold": False}])
    row = failure_breakdown(df).iloc[0]
    assert row["retrieval failure"] == 1 and row["correct"] == 0


def test_failure_examples_pick_first_ids_and_stop_draft():
    traj = make_traj(["Rome", "Paris", "Berlin", "Berlin", "Berlin"], qid="q1")
    df = frame([{"id": "q1", "policy": "P1", "em": 0.0, "f1": 0.0, "stop_step": 3, "category": "over-searching"}])
    out = failure_examples(df, [traj], policies=("P1",), n=3)
    assert len(out) == 1 and out.iloc[0]["draft_at_stop"] == "Berlin" and out.iloc[0]["queries"].startswith("query 0 | query 1")
