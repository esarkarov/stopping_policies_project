from src.policies import is_abstention, p1_signal, p2_signal, p3_signal, policy_cost, stop_oracle, stop_p1, stop_p2, stop_p3, stop_step
from tests.helpers import make_traj


def test_p1_stops_at_first_yes_or_runs_to_the_end():
    assert stop_p1(make_traj(list("abcde"), p1=[False, False, True, True, True])) == 3
    assert stop_p1(make_traj(list("abcde"))) == 5


def test_p3_needs_all_sub_questions():
    covered = [[True, False], [True, False], [True, True], [True, True], [True, True]]
    assert stop_p3(make_traj(list("abcde"), p3=covered)) == 3
    assert stop_p3(make_traj(list("abcde"))) == 5


def test_p2_needs_two_equal_drafts_and_never_stops_at_step_one():
    assert stop_p2(make_traj(["Paris", "Paris", "Rome", "Rome", "Rome"])) == 2
    assert stop_p2(make_traj(["A", "B", "C", "C", "D"])) == 4
    assert stop_p2(make_traj(["A", "B", "C", "D", "E"])) == 5


def test_p2_normalises_and_ignores_abstentions():
    assert stop_p2(make_traj(["The Paris.", "paris", "x", "y", "z"])) == 2
    assert stop_p2(make_traj(["Unknown", "Unknown", "Not enough information", "Not enough information", "Rome"])) == 5
    assert stop_p2(make_traj(["", "", "Rome", "Rome", "Rome"])) == 4
    assert is_abstention("Not enough information provided") and not is_abstention("Paris")
    assert is_abstention("Not mentioned in passages") and is_abstention("None") and is_abstention("N/A")
    assert not is_abstention("None of the Above Records")


def test_oracle_uses_first_correct_draft_else_best_f1_step():
    assert stop_oracle([False, True, True, False, False], [0.1, 1.0, 1.0, 0.2, 0.0]) == 2
    assert stop_oracle([False] * 5, [0.0, 0.3, 0.6, 0.6, 0.1]) == 3
    assert stop_oracle([False] * 5, [0.0] * 5) == 1


def test_stop_step_dispatch_and_fixed_budget():
    traj = make_traj(list("abcde"))
    assert [stop_step(f"P0_k{k}", traj, [False] * 5, [0.0] * 5) for k in (1, 3, 5)] == [1, 3, 5]


def test_cost_of_each_policy_counts_only_the_calls_it_needs():
    traj = make_traj(list("abcde"))
    assert policy_cost("P0_k2", traj, 2) == {"calls": 3, "tokens": 2 * 11 + 21}
    assert policy_cost("P1", traj, 3) == {"calls": 7, "tokens": 3 * 11 + 3 * 31 + 21}
    assert policy_cost("P2", traj, 3) == {"calls": 6, "tokens": 3 * 11 + 3 * 21}
    assert policy_cost("P3", traj, 3) == {"calls": 8, "tokens": 55 + 3 * 11 + 3 * 41 + 21}


def test_retry_call_is_counted_in_every_policy():
    traj = make_traj(list("abcde"))
    traj["steps"][1]["calls"]["query_retry"] = {"prompt_tokens": 7, "completion_tokens": 1, "seconds": 1.0}
    assert policy_cost("P0_k2", traj, 2) == {"calls": 4, "tokens": 2 * 11 + 8 + 21}
    assert policy_cost("P0_k1", traj, 1) == {"calls": 2, "tokens": 11 + 21}


def test_signals_match_the_stop_rules():
    traj = make_traj(["A", "B", "B", "C", "C"], p1=[False, True, False, False, False], p3=[[True, False], [True, True], [False, False], [False, False], [False, False]])
    steps = traj["steps"]
    assert [p1_signal(s) for s in steps] == [False, True, False, False, False]
    assert [p2_signal(steps, i) for i in range(5)] == [False, False, True, False, True]
    assert [p3_signal(s) for s in steps] == [False, True, False, False, False]
    assert stop_p1(traj) == 2 and stop_p2(traj) == 3 and stop_p3(traj) == 2


def test_variant_stop_rules():
    from src.policies import stop_variant

    p1 = [True, False, True, True, True]
    p3 = [[True, True], [False, False], [True, True], [False, False], [False, False]]
    traj = make_traj(["A", "B", "B", "C", "C"], p1=p1, p3=p3)
    assert stop_p1(traj) == 1 and stop_variant("P1_min2", traj) == 3 and stop_variant("P1_c2", traj) == 4
    assert stop_p3(traj) == 1 and stop_variant("P3_min2", traj) == 3
    assert stop_variant("P1&P3", traj) == 1
    assert stop_variant("P3&P2", traj) == 3
    never = make_traj(list("abcde"))
    assert [stop_variant(v, never) for v in ("P1_min2", "P3_min2", "P1_c2", "P1&P3", "P3&P2")] == [5] * 5
    assert stop_step("P1_c2", traj, [False] * 5, [0.0] * 5) == 4


def test_variant_costs_count_only_the_calls_they_need():
    traj = make_traj(list("abcde"))
    assert policy_cost("P1_min2", traj, 3) == {"calls": 6, "tokens": 3 * 11 + 2 * 31 + 21}
    assert policy_cost("P3_min2", traj, 3) == {"calls": 7, "tokens": 55 + 3 * 11 + 2 * 41 + 21}
    assert policy_cost("P1_c2", traj, 3) == policy_cost("P1", traj, 3)
    assert policy_cost("P1&P3", traj, 3) == {"calls": 11, "tokens": 55 + 3 * 11 + 3 * 31 + 3 * 41 + 21}
    assert policy_cost("P3&P2", traj, 3) == {"calls": 10, "tokens": 55 + 3 * 11 + 3 * 41 + 3 * 21}
