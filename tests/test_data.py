import json

import pytest

from src.data import (
    build_corpus,
    load_hotpotqa,
    load_musique,
    prepare_dataset,
    read_jsonl,
    sample_questions,
)


def hotpot_item(i: int) -> dict:
    return {
        "_id": f"h{i:03d}",
        "question": f"Question {i}?",
        "answer": f"Answer {i}",
        "supporting_facts": [[f"Gold A{i}", 0], [f"Gold B{i}", 1]],
        "context": [
            [f"Gold A{i}", ["First sentence. ", "Second sentence."]],
            [f"Gold B{i}", ["Only sentence."]],
            ["Shared", ["Appears in every question."]],
        ],
    }


def musique_item(i: int, answerable: bool = True) -> dict:
    return {
        "id": f"m{i:03d}",
        "question": f"Question {i}?",
        "answer": f"Answer {i}",
        "answer_aliases": [f"Alias {i}"],
        "answerable": answerable,
        "paragraphs": [
            {"idx": 0, "title": f"Gold {i}", "paragraph_text": "Gold text.", "is_supporting": True},
            {"idx": 1, "title": "Shared", "paragraph_text": "Appears in every question.", "is_supporting": False},
        ],
    }


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "hotpot_dev_distractor_v1.json").write_text(json.dumps([hotpot_item(i) for i in range(20)]))
    items = [musique_item(i) for i in range(20)] + [musique_item(99, answerable=False)]
    (raw / "musique_ans_v1.0_dev.jsonl").write_text("\n".join(json.dumps(x) for x in items))
    return raw


def test_hotpotqa_gold_passages_and_text(raw_dir):
    ex = load_hotpotqa(raw_dir / "hotpot_dev_distractor_v1.json")[0]
    gold = {p["title"] for p in ex["passages"] if p["pid"] in ex["gold_pids"]}
    assert gold == {"Gold A0", "Gold B0"}
    first = next(p for p in ex["passages"] if p["title"] == "Gold A0")
    assert first["text"] == "First sentence. Second sentence."


def test_musique_skips_unanswerable_and_marks_gold(raw_dir):
    examples = load_musique(raw_dir / "musique_ans_v1.0_dev.jsonl")
    assert len(examples) == 20
    ex = examples[0]
    assert ex["answer_aliases"] == ["Alias 0"]
    assert [p["title"] for p in ex["passages"] if p["pid"] in ex["gold_pids"]] == ["Gold 0"]


def test_corpus_is_deduplicated(raw_dir):
    examples = load_hotpotqa(raw_dir / "hotpot_dev_distractor_v1.json")
    corpus = build_corpus(examples)
    assert len(corpus) == 20 * 2 + 1
    assert len({p["pid"] for p in corpus}) == len(corpus)


def test_sampling_is_seeded_and_disjoint(raw_dir):
    examples = load_hotpotqa(raw_dir / "hotpot_dev_distractor_v1.json")
    pilot, main = sample_questions(examples, n_main=8, n_pilot=3, seed=42)
    pilot2, main2 = sample_questions(list(reversed(examples)), n_main=8, n_pilot=3, seed=42)
    assert [e["id"] for e in main] == [e["id"] for e in main2]
    assert [e["id"] for e in pilot] == [e["id"] for e in pilot2]
    assert not {e["id"] for e in pilot} & {e["id"] for e in main}


def test_sampling_rejects_too_many_questions(raw_dir):
    examples = load_hotpotqa(raw_dir / "hotpot_dev_distractor_v1.json")
    with pytest.raises(ValueError):
        sample_questions(examples, n_main=19, n_pilot=3, seed=1)


def test_prepare_dataset_writes_files(raw_dir, tmp_path):
    out = tmp_path / "processed"
    stats = prepare_dataset("musique", raw_dir, out, n_main=8, n_pilot=2, seed=42)
    assert stats["main_questions"] == 8 and stats["pilot_questions"] == 2
    assert len(read_jsonl(out / "musique" / "main.jsonl")) == 8
    corpus_pids = {p["pid"] for p in read_jsonl(out / "musique" / "corpus.jsonl")}
    for q in read_jsonl(out / "musique" / "main.jsonl"):
        assert q["gold_pids"] and set(q["gold_pids"]) <= corpus_pids
        assert "passages" not in q
