import numpy as np
import pandas as pd

from src.data import read_jsonl
from src.metrics import best_over_answers, exact_match, f1_score
from src.policies import ADAPTIVE, FIXED_BUDGETS, POLICIES, policy_cost, stop_step
from src.stats import bootstrap_ci, paired_diff_ci

CATEGORIES = ("correct", "over-searching", "premature stop", "not caused by stopping")


def load_trajectories(path, questions_path=None) -> list[dict]:
    aliases = {}
    if questions_path:
        aliases = {q["id"]: q.get("answer_aliases", []) for q in read_jsonl(questions_path)}
    seen, trajectories = set(), []
    for traj in read_jsonl(path):
        if traj["id"] in seen:
            continue
        seen.add(traj["id"])
        if not traj.get("answer_aliases"):
            traj["answer_aliases"] = aliases.get(traj["id"], [])
        trajectories.append(traj)
    return trajectories


def answers_of(traj: dict) -> list[str]:
    return [traj["answer"], *traj.get("answer_aliases", [])]


def classify(stop: int, correct_by_step: list[bool]) -> str:
    if correct_by_step[stop - 1]:
        return "correct"
    if any(correct_by_step[: stop - 1]):
        return "over-searching"
    if any(correct_by_step[stop:]):
        return "premature stop"
    return "not caused by stopping"


def evaluate_trajectory(traj: dict, policies=POLICIES) -> list[dict]:
    answers = answers_of(traj)
    drafts = [step["draft_answer"] for step in traj["steps"]]
    em_by_step = [best_over_answers(exact_match, d, answers) for d in drafts]
    f1_by_step = [best_over_answers(f1_score, d, answers) for d in drafts]
    correct_by_step = [value == 1.0 for value in em_by_step]
    first_correct = next((i for i, c in enumerate(correct_by_step, 1) if c), None)

    gold = set(traj["gold_pids"])
    rows = []
    for policy in policies:
        stop = stop_step(policy, traj, correct_by_step, f1_by_step)
        retrieved = {r["pid"] for step in traj["steps"][:stop] for r in step["retrieved"]}
        found = len(gold & retrieved)
        category = classify(stop, correct_by_step)
        rows.append(
            {
                "id": traj["id"],
                "dataset": traj["dataset"],
                "policy": policy,
                "stop_step": stop,
                "searches": stop,
                **policy_cost(policy, traj, stop),
                "em": em_by_step[stop - 1],
                "f1": f1_by_step[stop - 1],
                "category": category,
                "late_stop": category == "correct" and first_correct is not None and first_correct < stop,
                "gold_found": found,
                "gold_total": len(gold),
                "gold_recall": found / len(gold) if gold else np.nan,
                "all_gold": found == len(gold),
                "draft": drafts[stop - 1],
            }
        )
    return rows


def evaluate(trajectories: list[dict], policies=POLICIES) -> pd.DataFrame:
    return pd.DataFrame([row for traj in trajectories for row in evaluate_trajectory(traj, policies)])


def summarize(df: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for policy in [p for p in POLICIES if p in set(df["policy"])]:
        part = df[df["policy"] == policy].sort_values("id")
        row = {"policy": policy, "n": len(part)}
        for metric in ("em", "f1", "searches"):
            mean, lo, hi = bootstrap_ci(part[metric], n_boot, seed)
            row.update({metric: mean, f"{metric}_lo": lo, f"{metric}_hi": hi})
        row["calls"] = part["calls"].mean()
        row["tokens"] = part["tokens"].mean()
        row["gold_recall"] = part["gold_recall"].mean()
        row["all_gold_rate"] = part["all_gold"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["policy", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(CATEGORIES), fill_value=0)
    counts["late_stop_among_correct"] = df[df["late_stop"]].groupby("policy").size().reindex(counts.index, fill_value=0)
    return counts.reindex([p for p in POLICIES if p in counts.index]).reset_index()


def compare_to_fixed(df: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    wide = {metric: df.pivot(index="id", columns="policy", values=metric) for metric in ("em", "f1", "searches")}
    rows = []
    for policy in [p for p in ADAPTIVE if p in set(df["policy"])]:
        for k in FIXED_BUDGETS:
            fixed = f"P0_k{k}"
            row = {"policy": policy, "versus": fixed}
            for metric, table in wide.items():
                mean, lo, hi = paired_diff_ci(table[policy], table[fixed], n_boot, seed)
                row.update({f"d_{metric}": mean, f"d_{metric}_lo": lo, f"d_{metric}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)
