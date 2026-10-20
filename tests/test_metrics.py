from src.metrics import best_over_answers, exact_match, f1_score, normalize_answer


def test_normalize_matches_hotpotqa_rules():
    assert normalize_answer("The Eiffel-Tower!") == "eiffeltower"
    assert normalize_answer("  An  Apple ") == "apple"


def test_exact_match_and_f1():
    assert exact_match("The Paris.", "paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0
    assert abs(f1_score("Andy Roddick", "Andrew Stephen Roddick") - 0.4) < 1e-9
    assert f1_score("", "Paris") == 0.0


def test_yes_no_get_no_partial_credit():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes it is", "yes") == 0.0
    assert f1_score("no", "yes") == 0.0


def test_best_over_aliases():
    assert best_over_answers(exact_match, "NYC", ["New York City", "NYC"]) == 1.0
    assert best_over_answers(exact_match, "LA", ["New York City", "NYC"]) == 0.0
