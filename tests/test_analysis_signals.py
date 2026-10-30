import numpy as np

from src.analysis import compare_to_curve, shared_check_tokens, shuffled_stop_baseline, signal_quality, step_scores, with_shared_check_cost
from src.evaluate import evaluate
from tests.helpers import make_traj


def trajectories():
    good = [make_traj(["Rome", "Paris", "Oslo", "Oslo", "Oslo"], p1=[False, True, True, True, True], qid=f"g{i}") for i in range(4)]
    bad = [make_traj(["Rome"] * 5, p1=[False] * 5, qid=f"b{i}") for i in range(4)]
    return good + bad


def test_step_scores_shape_and_values():
    scores = step_scores(trajectories())
    assert scores["f1"].shape == (8, 5)
    assert scores["em"][0].tolist() == [0, 1, 0, 0, 0] and scores["em"][4].tolist() == [0] * 5


def test_signal_quality_separates_signal_from_no_signal():
    table = signal_quality(trajectories(), policies=("P1",))
    step2 = table[(table["policy"] == "P1") & (table["step"] == "2")].iloc[0]
    assert step2["n"] == 8 and step2["n_signal"] == 4
    assert step2["em_signal"] == 1.0 and step2["em_no_signal"] == 0.0
    assert table[table["step"] == "all"].iloc[0]["n"] == 40


def test_shuffled_baseline_is_seeded_and_matches_real_when_stops_are_uninformative():
    trajs = trajectories()
    a = shuffled_stop_baseline(trajs, policies=("P1",), n_perm=200, seed=1)
    b = shuffled_stop_baseline(trajs, policies=("P1",), n_perm=200, seed=1)
    assert a.equals(b)
    f1 = a[a["metric"] == "f1"].iloc[0]
    assert f1["real"] > f1["shuffled_mean"] and f1["shuffled_lo"] <= f1["shuffled_mean"] <= f1["shuffled_hi"]
    assert f1["searches"] == np.mean([2] * 4 + [5] * 4)

    same = shuffled_stop_baseline([make_traj(["Paris"] * 5, qid=f"q{i}", p1=[True] * 5) for i in range(5)], policies=("P1",), n_perm=50)
    assert (same["diff"] == 0).all()


def test_shared_check_cost_only_changes_checked_policies():
    traj = make_traj(list("abcde"))
    assert shared_check_tokens(traj, "P0_k3", 3) == 3 * 11 + 21
    assert shared_check_tokens(traj, "P2", 3) == 3 * 11 + 3 * 21
    assert shared_check_tokens(traj, "P1", 3) == 3 * 11 + 21 + 3 * 1
    assert shared_check_tokens(traj, "P3", 3) == 3 * 11 + 21 + 55 + 3 * 1


def test_compare_to_curve_matches_cost_by_interpolation():
    trajs = [make_traj(["Rome", "Rome", "Paris", "Paris", "Paris"], p1=[False, False, True, True, True], qid=f"q{i}") for i in range(6)]
    df = with_shared_check_cost(evaluate(trajs), trajs)
    curve = compare_to_curve(df, "searches", n_boot=100, policies=("P1",))
    f1 = curve[curve["metric"] == "f1"].iloc[0]
    assert f1["cost_value"] == 3 and f1["adaptive"] == 1.0 and f1["fixed_curve"] == 1.0 and f1["diff"] == 0
    shared = compare_to_curve(df, "tokens_shared", n_boot=100, policies=("P1",))
    assert shared[shared["metric"] == "f1"].iloc[0]["cost_value"] < df[(df["policy"] == "P1")]["tokens"].mean()
