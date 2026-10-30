"""Analysis tables and figures from the evaluation output of run_eval.py.

No model is needed:
    docker compose run --rm --no-deps app python scripts/run_analysis.py --dataset musique --split main
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import by_hops, failure_breakdown, failure_examples, stop_distribution  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import load_trajectories  # noqa: E402
from src.plots import plot_stop_steps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main"])
    parser.add_argument("--examples", type=int, default=5, help="examples per policy and category")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    results = ROOT / cfg["paths"]["results_dir"]
    name = f"{args.dataset}_{args.split}"
    tables, figures = results / "tables", results / "figures"

    df = pd.read_csv(tables / f"{name}_per_question.csv")
    trajectories = load_trajectories(results / "trajectories" / f"{name}.jsonl")

    hops = by_hops(df)
    if hops is not None:
        hops.to_csv(tables / f"{name}_by_hops.csv", index=False)
        print(hops.pivot(index="policy", columns="hops", values="f1").round(3).to_string())

    stops = stop_distribution(df)
    stops.to_csv(tables / f"{name}_stop_steps.csv", index=False)
    title_name = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}[args.dataset]
    plot_stop_steps(stops, f"{title_name} ({args.split}, n={len(trajectories)})", figures / f"{name}_stop_steps.png")
    print(stops.to_string(index=False))

    failures = failure_breakdown(df)
    failures.to_csv(tables / f"{name}_failures.csv", index=False)
    print(failures.to_string(index=False))

    examples = failure_examples(df, trajectories, n=args.examples)
    examples.to_csv(tables / f"{name}_examples.csv", index=False)
    print(f"{len(examples)} examples; tables in {tables}, figures in {figures}")


if __name__ == "__main__":
    main()
