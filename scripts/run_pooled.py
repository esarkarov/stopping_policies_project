import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import compare_to_curve, shuffled_stop_baseline, stop_distribution, with_shared_check_cost  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import compare_to_fixed, error_table, evaluate, load_trajectories, summarize  # noqa: E402
from src.plots import plot_quality_vs_cost, plot_stop_steps  # noqa: E402
from src.pooled import pool  # noqa: E402

SPLITS = ("main", "confirm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples")
    parser.add_argument("--permutations", type=int, default=2000, help="shuffled-stop samples")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    results = ROOT / cfg["paths"]["results_dir"]
    data = ROOT / cfg["paths"]["data_dir"] / "processed" / args.dataset
    seed = cfg["data"]["seed"]
    name = f"{args.dataset}_pooled"

    parts = []
    for split in SPLITS:
        questions = data / f"{split}.jsonl"
        parts.append(load_trajectories(results / "trajectories" / f"{args.dataset}_{split}.jsonl", questions if questions.exists() else None))
    trajectories = pool(*parts)
    print(f"{name}: {len(trajectories)} questions ({' + '.join(str(len(p)) for p in parts)})")

    df = evaluate(trajectories)
    tables, figures = results / "tables", results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary = summarize(df, args.bootstrap, seed)
    summary.to_csv(tables / f"{name}_summary.csv", index=False)
    error_table(df).to_csv(tables / f"{name}_errors.csv", index=False)
    compare_to_fixed(df, args.bootstrap, seed).to_csv(tables / f"{name}_vs_fixed.csv", index=False)

    shared = with_shared_check_cost(df, trajectories)
    curves = pd.concat(
        [compare_to_curve(shared, cost, args.bootstrap, seed) for cost in ("searches", "tokens", "tokens_shared")],
        ignore_index=True,
    )
    curves.to_csv(tables / f"{name}_vs_curve.csv", index=False)
    shuffled = shuffled_stop_baseline(trajectories, n_perm=args.permutations, seed=seed)
    shuffled.to_csv(tables / f"{name}_shuffled.csv", index=False)

    stops = stop_distribution(df)
    stops.to_csv(tables / f"{name}_stop_steps.csv", index=False)
    title_name = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}[args.dataset]
    title = f"{title_name} (pooled, n={len(trajectories)})"
    for metric in ("f1", "em"):
        plot_quality_vs_cost(summary, metric, title, figures / f"{name}_{metric}_vs_searches.png")
        plot_quality_vs_cost(summary, metric, title, figures / f"{name}_{metric}_vs_tokens.png", cost="tokens")
    plot_stop_steps(stops, title, figures / f"{name}_stop_steps.png")

    print(summary[["policy", "n", "em", "f1", "f1_lo", "f1_hi", "searches", "tokens"]].round(3).to_string(index=False))
    print(curves[curves["metric"] == "f1"].round(3).to_string(index=False))
    print(shuffled[shuffled["metric"] == "f1"].round(3).to_string(index=False))
    print(f"tables in {tables}, figures in {figures}")


if __name__ == "__main__":
    main()