import pytest

from src.agent import (
    SearchAgent,
    is_repeat,
    parse_answer,
    parse_coverage,
    parse_query,
    parse_subquestions,
    parse_yes_no,
)
from src.retriever import BM25Retriever

PASSAGES = [
    {"pid": f"p{i}", "title": f"Title {i}", "text": f"Text about topic{i} and alpha beta {i}."} for i in range(12)
]
EXAMPLE = {"id": "q1", "dataset": "hotpotqa", "question": "What is alpha beta?", "answer": "x", "answer_aliases": ["y"], "gold_pids": ["p1"]}


class FakeLLM:
    def __init__(self, enough="Yes", coverage="1: yes\n2: yes"):
        self.enough, self.coverage = enough, coverage
        self.prompts = []

    def generate(self, kind, prompt, max_tokens):
        self.prompts.append((kind, prompt))
        text = {
            "decompose": "1. What is alpha?\n2. What is beta?",
            "query": "Query: alpha beta",
            "query_retry": f"Query: topic{len(self.prompts)}",
            "answer": "Answer: forty two.",
            "check_enough": self.enough,
            "check_coverage": self.coverage,
        }[kind]
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 3, "seconds": 0.1, "cached": False}


def make_agent(llm):
    return SearchAgent(llm, BM25Retriever(PASSAGES), max_steps=5, passages_per_step=2)


def test_parsers():
    assert parse_query("Query: capital of France") == "capital of France"
    assert parse_answer("Answer: Christopher Nolan.") == "Christopher Nolan"
    assert parse_answer("thinking...\nAnswer: Paris\nbecause") == "Paris"
    assert parse_yes_no(" Yes.") and not parse_yes_no("No") and not parse_yes_no("")
    assert parse_subquestions("1. who?\n2) where?\nnoise") == ["who?", "where?"]
    assert parse_coverage("1: yes\n2: no", 3) == [True, False, False]


def test_agent_runs_all_steps_and_never_repeats_passages():
    trajectory = make_agent(FakeLLM()).run(EXAMPLE)
    assert [s["step"] for s in trajectory["steps"]] == [1, 2, 3, 4, 5]
    pids = [r["pid"] for s in trajectory["steps"] for r in s["retrieved"]]
    assert len(pids) == 10 and len(set(pids)) == 10
    assert trajectory["subquestions"] == ["What is alpha?", "What is beta?"]
    assert trajectory["answer_aliases"] == ["y"]
    first = trajectory["steps"][0]
    assert first["draft_answer"] == "forty two"
    assert first["p1_enough"] is True and first["p3_covered"] == [True, True]
    assert set(first["calls"]) == {"query", "answer", "check_enough", "check_coverage"}


def test_checks_never_change_what_the_agent_sees():
    agent_yes, agent_no = FakeLLM("Yes", "1: yes\n2: yes"), FakeLLM("No", "1: no\n2: no")
    make_agent(agent_yes).run(EXAMPLE)
    make_agent(agent_no).run(EXAMPLE)
    main_calls = lambda llm: [(k, p) for k, p in llm.prompts if k in ("decompose", "query", "query_retry", "answer")]
    assert main_calls(agent_yes) == main_calls(agent_no)
    for kind, prompt in agent_yes.prompts:
        if kind in ("query", "answer"):
            assert "Sub-questions" not in prompt and "Yes or No" not in prompt


def test_unparsable_query_falls_back_to_question():
    llm = FakeLLM()
    original = llm.generate
    llm.generate = lambda kind, prompt, mt: (
        {**original(kind, prompt, mt), "text": ""} if kind == "query" else original(kind, prompt, mt)
    )
    trajectory = make_agent(llm).run(EXAMPLE)
    assert trajectory["steps"][0]["query"] == EXAMPLE["question"]


def test_is_repeat_ignores_order_and_case():
    assert is_repeat("Alpha beta", ["beta alpha"])
    assert not is_repeat("gamma delta", ["alpha beta"])
    assert not is_repeat("alpha", [])


def test_repeated_query_is_retried_once_and_the_call_is_saved():
    trajectory = make_agent(FakeLLM()).run(EXAMPLE)
    first, second = trajectory["steps"][0], trajectory["steps"][1]
    assert "query_retry" not in first["calls"] and first["query"] == "alpha beta"
    assert "query_retry" in second["calls"] and second["query"].startswith("topic")


def test_retry_prompt_names_the_repeated_query():
    llm = FakeLLM()
    make_agent(llm).run(EXAMPLE)
    retries = [p for k, p in llm.prompts if k == "query_retry"]
    assert retries and all('"alpha beta" repeats an earlier query' in p for p in retries)
