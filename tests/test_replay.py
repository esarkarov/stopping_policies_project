import numpy as np
import pandas as pd

from src.agent import SearchAgent
from src.analysis import load_p_yes, p1_threshold_curve, p1_threshold_rows
from src.evaluate import evaluate
from src.replay import passages_seen, replay_trajectory, yes_probability
from tests.helpers import make_traj


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, kind, prompt, max_tokens):
        self.prompts.append((kind, prompt))
        text = {"decompose": "1. first\n2. second", "query": "Query: alpha beta", "answer": "Answer: x", "check_enough": "Yes", "check_coverage": "1: yes\n2: no"}.get(kind, "Query: gamma")
        return {"text": text, "prompt_tokens": 5, "completion_tokens": 1, "seconds": 0.1}


class FakeRetriever:
    def __init__(self):
        self.passages = [{"pid": f"p{i}", "title": f"T{i}", "text": f"text {i} " * 5} for i in range(20)]

    def search(self, query, k=2, exclude=frozenset()):
        return [{**p, "score": 1.0} for p in self.passages if p["pid"] not in exclude][:k]


def test_yes_probability_normalises_over_yes_and_no_tokens():
    top = [{"token": "Yes", "logprob": np.log(0.6)}, {"token": "yes", "logprob": np.log(0.1)}, {"token": "No", "logprob": np.log(0.3)}, {"token": "I", "logprob": np.log(0.5)}]
    assert abs(yes_probability(top) - 0.7) < 1e-9
    assert yes_probability([{"token": "I", "logprob": -1.0}]) == 0.0


def test_replayed_check_prompts_are_identical_to_the_agent_prompts():
    llm, retriever = RecordingLLM(), FakeRetriever()
    example = {"id": "q", "dataset": "hotpotqa", "question": "Who?", "answer": "x", "gold_pids": ["p0"]}
    traj = SearchAgent(llm, retriever, max_steps=3).run(example)
    corpus = {p["pid"]: p for p in retriever.passages}
    original = [prompt for kind, prompt in llm.prompts if kind == "check_enough"]
    replayed = []
    row = replay_trajectory(traj, corpus, lambda prompt: (replayed.append(prompt) or "Yes", [{"token": "Yes", "logprob": -0.1}]))
    assert replayed == original and len(replayed) == 3
    assert [s["matches_stored"] for s in row["steps"]] == [True] * 3
    assert len(passages_seen(traj, corpus, 2)) == 4


def test_threshold_rows_stop_at_first_step_reaching_tau():
    trajs = [make_traj(["Rome", "Paris", "Paris", "Paris", "Paris"], qid="a")]
    p_yes = {"a": [0.2, 0.7, 0.95, 0.99, 0.99]}
    assert p1_threshold_rows(trajs, p_yes, 0.5).iloc[0]["stop_step"] == 2
    assert p1_threshold_rows(trajs, p_yes, 0.9).iloc[0]["stop_step"] == 3
    assert p1_threshold_rows(trajs, p_yes, 0.9999).iloc[0]["stop_step"] == 5


def test_threshold_curve_has_one_row_per_tau(tmp_path):
    trajs = [make_traj(["Rome", "Rome", "Paris", "Paris", "Paris"], qid=f"q{i}", p1=[False] * 5) for i in range(6)]
    p_yes = {t["id"]: [0.1, 0.4, 0.9, 0.95, 0.99] for t in trajs}
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(f'{{"id": "{k}", "steps": [' + ",".join(f'{{"p_yes": {v}}}' for v in vals) + "]}" for k, vals in p_yes.items()))
    assert load_p_yes(path) == p_yes
    table = p1_threshold_curve(evaluate(trajs), trajs, p_yes, taus=(0.5, 0.9), n_boot=50)
    assert list(table["tau"]) == [0.5, 0.9] and list(table["searches"]) == [3, 3]
    assert isinstance(table, pd.DataFrame) and "f1_diff_vs_curve_searches" in table.columns
