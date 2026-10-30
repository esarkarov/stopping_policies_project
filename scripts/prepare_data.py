import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  
from src.data import prepare_dataset  
from src.retriever import BM25Retriever  


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_dir = ROOT / cfg["paths"]["data_dir"]
    d = cfg["data"]

    for name in d["datasets"]:
        stats = prepare_dataset(
            name,
            data_dir / "raw",
            data_dir / "processed",
            d["n_questions"],
            d["n_pilot"],
            d["seed"],
            d.get("n_confirm", 0),
        )
        print(stats)

        retriever = BM25Retriever.from_corpus_file(data_dir / "processed" / name / "corpus.jsonl")
        example = retriever.passages[0]["title"]
        top = retriever.search(example, k=2)
        print(f"{name}: BM25 index ready, query '{example}' ->", [p["title"] for p in top])


if __name__ == "__main__":
    main()
