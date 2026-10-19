import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def passage_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]


def _passage(title: str, text: str) -> dict:
    title, text = title.strip(), text.strip()
    return {"pid": passage_id(title, text), "title": title, "text": text}


def load_hotpotqa(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    examples = []
    for item in raw:
        passages = [_passage(title, " ".join(s.strip() for s in sents)) for title, sents in item["context"]]
        gold_titles = {title.strip() for title, _ in item["supporting_facts"]}
        examples.append(
            {
                "id": item["_id"],
                "dataset": "hotpotqa",
                "question": item["question"],
                "answer": item["answer"],
                "answer_aliases": [],
                "gold_pids": sorted({p["pid"] for p in passages if p["title"] in gold_titles}),
                "passages": passages,
            }
        )
    return examples


def load_musique(path) -> list[dict]:
    examples = []
    for item in read_jsonl(path):
        if not item.get("answerable", True):
            continue
        passages = [_passage(p["title"], p["paragraph_text"]) for p in item["paragraphs"]]
        gold_pids = {
            passages[i]["pid"] for i, p in enumerate(item["paragraphs"]) if p.get("is_supporting")
        }
        examples.append(
            {
                "id": item["id"],
                "dataset": "musique",
                "question": item["question"],
                "answer": item["answer"],
                "answer_aliases": list(item.get("answer_aliases", [])),
                "gold_pids": sorted(gold_pids),
                "passages": passages,
            }
        )
    return examples


LOADERS = {"hotpotqa": load_hotpotqa, "musique": load_musique}
RAW_FILES = {"hotpotqa": "hotpot_dev_distractor_v1.json", "musique": "musique_ans_v1.0_dev.jsonl"}


def build_corpus(examples: list[dict]) -> list[dict]:
    corpus = {}
    for ex in examples:
        for p in ex["passages"]:
            corpus.setdefault(p["pid"], p)
    return sorted(corpus.values(), key=lambda p: p["pid"])


def sample_questions(examples: list[dict], n_main: int, n_pilot: int, seed: int):
    ordered = sorted(examples, key=lambda ex: ex["id"])
    if n_main + n_pilot > len(ordered):
        raise ValueError(f"requested {n_main + n_pilot} questions, only {len(ordered)} available")
    chosen = random.Random(seed).sample(ordered, n_main + n_pilot)
    return chosen[:n_pilot], chosen[n_pilot:]


def question_record(ex: dict) -> dict:
    return {k: ex[k] for k in ("id", "dataset", "question", "answer", "answer_aliases", "gold_pids")}


def prepare_dataset(name: str, raw_dir, out_dir, n_main: int, n_pilot: int, seed: int) -> dict:
    raw_path = Path(raw_dir) / RAW_FILES[name]
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found; run scripts/download_data.py first")

    examples = LOADERS[name](raw_path)
    corpus = build_corpus(examples)
    pilot, main = sample_questions(examples, n_main, n_pilot, seed)

    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "pilot.jsonl", (question_record(ex) for ex in pilot))
    write_jsonl(out / "main.jsonl", (question_record(ex) for ex in main))

    stats = {
        "dataset": name,
        "seed": seed,
        "dev_questions": len(examples),
        "corpus_passages": len(corpus),
        "pilot_questions": len(pilot),
        "main_questions": len(main),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
