import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluate import compare_to_fixed, error_table, evaluate, load_trajectories, summarize  # noqa: E402
from src.plots import plot_quality_vs_cost  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    parser.add_argument("--split", required=True, choices=["pilot", "main", "confirm"])
    parser.add_argument("--trajectories", default=None, help="path to a trajectory file")
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    results = ROOT / cfg["paths"]["results_dir"]
    seed = cfg["data"]["seed"]
    name = f"{args.dataset}_{args.split}"

    traj_path = Path(args.trajectories) if args.trajectories else results / "trajectories" / f"{name}.jsonl"
    questions_path = ROOT / cfg["paths"]["data_dir"] / "processed" / args.dataset / f"{args.split}.jsonl"
    trajectories = load_trajectories(traj_path, questions_path if questions_path.exists() else None)
    print(f"{name}: {len(trajectories)} questions from {traj_path}")

    df = evaluate(trajectories)
    summary = summarize(df, args.bootstrap, seed)
    errors = error_table(df)
    versus = compare_to_fixed(df, args.bootstrap, seed)

    tables, figures = results / "tables", results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"{name}_per_question.csv", index=False)
    summary.to_csv(tables / f"{name}_summary.csv", index=False)
    errors.to_csv(tables / f"{name}_errors.csv", index=False)
    versus.to_csv(tables / f"{name}_vs_fixed.csv", index=False)

    title_name = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}[args.dataset]
    plot_title = f"{title_name} ({args.split}, n={len(trajectories)})"
    for metric in ("f1", "em"):
        plot_quality_vs_cost(summary, metric, plot_title, figures / f"{name}_{metric}_vs_searches.png")
        plot_quality_vs_cost(summary, metric, plot_title, figures / f"{name}_{metric}_vs_tokens.png", cost="tokens")

    columns = ["policy", "em", "f1", "searches", "calls", "tokens", "gold_recall"]
    print(summary[columns].round(2).to_string(index=False))
    print(errors.to_string(index=False))
    print(f"tables in {tables}, figures in {figures}")


if __name__ == "__main__":
    main()
