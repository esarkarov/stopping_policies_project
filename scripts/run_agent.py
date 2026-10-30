import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import SearchAgent  
from src.config import load_config  
from src.data import read_jsonl  
from src.llm import OllamaLLM  
from src.manifest import append_manifest, build_manifest  
from src.retriever import BM25Retriever  


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main", "confirm"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_dir = ROOT / cfg["paths"]["data_dir"] / "processed" / args.dataset
    results_dir = ROOT / cfg["paths"]["results_dir"]
    model = os.environ.get("MODEL", cfg["model"])

    questions = read_jsonl(data_dir / f"{args.split}.jsonl")
    if args.limit:
        questions = questions[: args.limit]

    out_path = results_dir / "trajectories" / f"{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.dataset}/{args.split}: {len(done)} done, {len(todo)} to run, model {model}")
    if not todo:
        return

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    manifest = build_manifest(ROOT, cfg, model, host, args.dataset, args.split, len(todo))
    append_manifest(out_path.with_suffix(".manifest.jsonl"), manifest)

    llm = OllamaLLM(
        model,
        temperature=cfg["temperature"],
        think=cfg["think"],
        num_ctx=cfg.get("num_ctx", 4096),
        seed=cfg["data"]["seed"],
        cache_path=results_dir / "cache" / "llm_cache.jsonl",
    )
    retriever = BM25Retriever.from_corpus_file(data_dir / "corpus.jsonl")
    agent = SearchAgent(
        llm,
        retriever,
        max_steps=cfg["agent"]["max_steps"],
        passages_per_step=cfg["agent"]["passages_per_step"],
        max_passage_chars=cfg["agent"].get("max_passage_chars", 1500),
    )

    start = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out:
        for question in tqdm(todo):
            trajectory = agent.run(question)
            out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            out.flush()
    elapsed = time.perf_counter() - start
    print(f"finished {len(todo)} questions in {elapsed:.0f}s ({elapsed / len(todo):.1f}s per question)")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
