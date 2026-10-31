import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import compare_to_curve, variants_vs_base  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import compare_to_fixed, evaluate, load_trajectories, summarize  # noqa: E402
from src.policies import ADAPTIVE, FIXED_BUDGETS, VARIANTS  # noqa: E402

ALPHAS = (0.05, 0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main", "confirm"])
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    results = ROOT / cfg["paths"]["results_dir"]
    seed = cfg["data"]["seed"]
    name = f"{args.dataset}_{args.split}"

    trajectories = load_trajectories(results / "trajectories" / f"{name}.jsonl")
    print(f"{name}: {len(trajectories)} questions")
    fixed = tuple(f"P0_k{k}" for k in FIXED_BUDGETS)
    df = evaluate(trajectories, (*fixed, *ADAPTIVE, *VARIANTS))

    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = summarize(df, args.bootstrap, seed, order=(*fixed, *ADAPTIVE, *VARIANTS))
    summary.to_csv(tables / f"{name}_variants_summary.csv", index=False)
    compare_to_fixed(df, args.bootstrap, seed, policies=VARIANTS).to_csv(tables / f"{name}_variants_vs_fixed.csv", index=False)

    base = pd.concat([variants_vs_base(df, args.bootstrap, seed, alpha).assign(alpha=alpha) for alpha in ALPHAS], ignore_index=True)
    base.to_csv(tables / f"{name}_variants_vs_base.csv", index=False)

    policies = (*ADAPTIVE, *VARIANTS)
    curves = pd.concat(
        [
            compare_to_curve(df, cost, args.bootstrap, seed, policies=policies, alpha=alpha).assign(alpha=alpha)
            for cost in ("searches", "tokens")
            for alpha in ALPHAS
        ],
        ignore_index=True,
    )
    curves.to_csv(tables / f"{name}_variants_vs_curve.csv", index=False)

    print(summary[["policy", "em", "f1", "searches", "calls", "tokens"]].round(3).to_string(index=False))
    shown = curves[(curves["metric"] == "f1") & (curves["alpha"] == 0.05)]
    print(shown[["policy", "cost", "cost_value", "adaptive", "fixed_curve", "diff", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
    print(f"tables in {tables}")


if __name__ == "__main__":
    main()
