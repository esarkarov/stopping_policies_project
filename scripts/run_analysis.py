import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    by_hops,
    compare_to_curve,
    failure_breakdown,
    failure_examples,
    shuffled_stop_baseline,
    signal_quality,
    stop_distribution,
    with_shared_check_cost,
)
from src.config import load_config  # noqa: E402
from src.evaluate import load_trajectories  # noqa: E402
from src.plots import plot_stop_steps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main", "confirm"])
    parser.add_argument("--examples", type=int, default=5, help="examples per policy and category")
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples")
    parser.add_argument("--permutations", type=int, default=2000, help="shuffled-stop samples")
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

    seed = cfg["data"]["seed"]
    signal = signal_quality(trajectories)
    signal.to_csv(tables / f"{name}_signal.csv", index=False)
    print(signal[signal["step"] == "all"].round(3).to_string(index=False))

    shuffled = shuffled_stop_baseline(trajectories, n_perm=args.permutations, seed=seed)
    shuffled.to_csv(tables / f"{name}_shuffled.csv", index=False)
    print(shuffled[shuffled["metric"] == "f1"].round(3).to_string(index=False))

    shared = with_shared_check_cost(df, trajectories)
    curves = pd.concat(
        [compare_to_curve(shared, cost, args.bootstrap, seed) for cost in ("searches", "tokens", "tokens_shared")],
        ignore_index=True,
    )
    curves.to_csv(tables / f"{name}_vs_curve.csv", index=False)
    print(curves[curves["metric"] == "f1"].round(3).to_string(index=False))
    print(f"{len(examples)} examples; tables in {tables}, figures in {figures}")


if __name__ == "__main__":
    main()
