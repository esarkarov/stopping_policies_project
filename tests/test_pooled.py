import pytest

from src.evaluate import evaluate, summarize
from src.pooled import pool
from tests.helpers import make_traj


def split(prefix, n):
    return [make_traj(["Rome", "Paris", "Paris", "Paris", "Paris"], p1=[False, True, True, True, True], qid=f"{prefix}{i}") for i in range(n)]


def test_pool_concatenates_disjoint_splits_in_order():
    pooled = pool(split("m", 3), split("c", 4))
    assert [t["id"] for t in pooled] == ["m0", "m1", "m2", "c0", "c1", "c2", "c3"]


def test_pool_rejects_a_question_in_both_splits():
    with pytest.raises(ValueError, match="m0"):
        pool(split("m", 3), split("m", 2))


def test_pooled_summary_counts_all_questions():
    summary = summarize(evaluate(pool(split("m", 3), split("c", 4))), n_boot=50, seed=1)
    assert set(summary["n"]) == {7}
