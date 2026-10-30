import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import read_jsonl  # noqa: E402
from src.evaluate import load_trajectories  # noqa: E402
from src.replay import chat_logprobs, replay_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main", "confirm"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_dir = ROOT / cfg["paths"]["data_dir"] / "processed" / args.dataset
    traj_path = ROOT / cfg["paths"]["results_dir"] / "trajectories" / f"{args.dataset}_{args.split}.jsonl"
    out_path = traj_path.with_suffix(".p1_logprobs.jsonl")
    model = os.environ.get("MODEL", cfg["model"])
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    options = {
        "temperature": cfg["temperature"],
        "seed": cfg["data"]["seed"],
        "num_ctx": cfg.get("num_ctx", 4096),
        "num_predict": 4,
    }
    max_chars = cfg["agent"].get("max_passage_chars", 1500)

    corpus = {p["pid"]: p for p in read_jsonl(data_dir / "corpus.jsonl")}
    trajectories = load_trajectories(traj_path)
    if args.limit:
        trajectories = trajectories[: args.limit]
    done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [t for t in trajectories if t["id"] not in done]
    print(f"{args.dataset}/{args.split}: {len(done)} done, {len(todo)} to run, model {model}")

    call = lambda prompt: chat_logprobs(host, model, prompt, options)
    start = time.perf_counter()
    checked = mismatched = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for traj in tqdm(todo):
            row = replay_trajectory(traj, corpus, call, max_chars)
            checked += len(row["steps"])
            mismatched += sum(not s["matches_stored"] for s in row["steps"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    if todo:
        print(f"finished in {time.perf_counter() - start:.0f}s; replayed answers differing from the stored ones: {mismatched}/{checked}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
