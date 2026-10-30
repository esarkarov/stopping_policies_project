import json
import re

import numpy as np
import pandas as pd

from src.evaluate import answers_of
from src.metrics import best_over_answers, exact_match, f1_score
from src.policies import ADAPTIVE, FIXED_BUDGETS, POLICIES, p1_signal, p2_signal, p3_signal, policy_cost, stop_p1, stop_p2, stop_p3
from src.stats import bootstrap_ci, paired_diff_ci

OUTCOMES = ("correct", "near miss", "reading failure", "retrieval failure")
NEAR_MISS_F1 = 0.5


def hop_count(question_id: str):
    match = re.match(r"(\d)hop", question_id)
    return int(match.group(1)) if match else None


def by_hops(df: pd.DataFrame):
    hops = df["id"].map(hop_count)
    if hops.isna().all():
        return None
    part = df.assign(hops=hops)
    table = part.groupby(["hops", "policy"]).agg(
        n=("id", "nunique"), em=("em", "mean"), f1=("f1", "mean"), searches=("searches", "mean"), tokens=("tokens", "mean")
    )
    return table.reset_index().assign(hops=lambda t: t["hops"].astype(int))


def stop_distribution(df: pd.DataFrame, policies=(*ADAPTIVE, "Oracle")) -> pd.DataFrame:
    part = df[df["policy"].isin(policies)]
    counts = part.groupby(["policy", "stop_step"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, 6), fill_value=0).reindex(list(policies))
    counts.columns = [f"step_{c}" for c in counts.columns]
    return counts.reset_index()


def outcome(row) -> str:
    if row["em"] == 1.0:
        return "correct"
    if row["f1"] >= NEAR_MISS_F1:
        return "near miss"
    return "reading failure" if row["all_gold"] else "retrieval failure"


def failure_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df.assign(outcome=df.apply(outcome, axis=1))
    counts = labelled.groupby(["policy", "outcome"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(OUTCOMES), fill_value=0)
    return counts.reindex([p for p in POLICIES if p in counts.index]).reset_index()


def failure_examples(df: pd.DataFrame, trajectories: list[dict], policies=ADAPTIVE, categories=("premature stop", "over-searching"), n: int = 5) -> pd.DataFrame:
    by_id = {t["id"]: t for t in trajectories}
    rows = []
    for policy in policies:
        for category in categories:
            part = df[(df["policy"] == policy) & (df["category"] == category)].sort_values("id").head(n)
            for _, r in part.iterrows():
                traj = by_id[r["id"]]
                steps = traj["steps"]
                rows.append(
                    {
                        "policy": policy,
                        "category": category,
                        "id": r["id"],
                        "stop_step": int(r["stop_step"]),
                        "question": traj["question"],
                        "gold": traj["answer"],
                        "draft_at_stop": steps[int(r["stop_step"]) - 1]["draft_answer"],
                        "draft_at_5": steps[-1]["draft_answer"],
                        "queries": " | ".join(s["query"] for s in steps),
                    }
                )
    return pd.DataFrame(rows)


STOP_FUNCTIONS = {"P1": stop_p1, "P2": stop_p2, "P3": stop_p3}
SIGNALS = {
    "P1": lambda steps, i: p1_signal(steps[i]),
    "P2": p2_signal,
    "P3": lambda steps, i: p3_signal(steps[i]),
}
METRICS = ("f1", "em")


def step_scores(trajectories: list[dict]) -> dict:
    scores = {"em": [], "f1": []}
    for traj in trajectories:
        answers = answers_of(traj)
        drafts = [step["draft_answer"] for step in traj["steps"]]
        scores["em"].append([best_over_answers(exact_match, d, answers) for d in drafts])
        scores["f1"].append([best_over_answers(f1_score, d, answers) for d in drafts])
    return {metric: np.array(values) for metric, values in scores.items()}


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def signal_quality(trajectories: list[dict], policies=ADAPTIVE) -> pd.DataFrame:
    scores = step_scores(trajectories)
    n_steps = scores["em"].shape[1]
    rows = []
    for policy in policies:
        signal = np.array(
            [[SIGNALS[policy](traj["steps"], i) for i in range(n_steps)] for traj in trajectories], dtype=bool
        )
        for label, columns in [(str(i + 1), [i]) for i in range(n_steps)] + [("all", list(range(n_steps)))]:
            mask = signal[:, columns]
            row = {"policy": policy, "step": label, "n": int(mask.size), "n_signal": int(mask.sum())}
            for metric in METRICS:
                values = scores[metric][:, columns]
                row[f"{metric}_signal"] = _mean_or_nan(values[mask])
                row[f"{metric}_no_signal"] = _mean_or_nan(values[~mask])
            rows.append(row)
    return pd.DataFrame(rows)


def shuffled_stop_baseline(trajectories: list[dict], policies=ADAPTIVE, n_perm: int = 2000, seed: int = 42) -> pd.DataFrame:
    scores = step_scores(trajectories)
    rng = np.random.default_rng(seed)
    index = np.arange(len(trajectories))
    rows = []
    for policy in policies:
        stops = np.array([STOP_FUNCTIONS[policy](traj) for traj in trajectories])
        shuffled = rng.permuted(np.tile(stops, (n_perm, 1)), axis=1)
        for metric in METRICS:
            table = scores[metric]
            real = float(table[index, stops - 1].mean())
            null = table[index[None, :], shuffled - 1].mean(axis=1)
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "searches": float(stops.mean()),
                    "real": real,
                    "shuffled_mean": float(null.mean()),
                    "shuffled_lo": float(np.quantile(null, 0.025)),
                    "shuffled_hi": float(np.quantile(null, 0.975)),
                    "diff": real - float(null.mean()),
                    "p_value": (1 + int((null >= real).sum())) / (n_perm + 1),
                }
            )
    return pd.DataFrame(rows)


def shared_check_tokens(traj: dict, policy: str, stop: int) -> int:
    if policy not in ("P1", "P3"):
        return policy_cost(policy, traj, stop)["tokens"]
    steps = traj["steps"][:stop]
    tokens = policy_cost("P0_k1", traj, stop)["tokens"]
    if policy == "P1":
        return tokens + sum(step["calls"]["check_enough"]["completion_tokens"] for step in steps)
    decompose = traj["decompose_call"]
    tokens += decompose["prompt_tokens"] + decompose["completion_tokens"]
    return tokens + sum(step["calls"]["check_coverage"]["completion_tokens"] for step in steps)


def with_shared_check_cost(df: pd.DataFrame, trajectories: list[dict]) -> pd.DataFrame:
    by_id = {traj["id"]: traj for traj in trajectories}
    shared = [
        shared_check_tokens(by_id[row.id], row.policy, int(row.stop_step)) for row in df.itertuples(index=False)
    ]
    return df.assign(tokens_shared=shared)


def compare_to_curve(df: pd.DataFrame, cost: str = "searches", n_boot: int = 10000, seed: int = 42, policies=ADAPTIVE) -> pd.DataFrame:
    fixed = [f"P0_k{k}" for k in FIXED_BUDGETS]
    fixed_cost = np.array([df.loc[df["policy"] == name, cost].mean() for name in fixed])
    wide = {metric: df.pivot(index="id", columns="policy", values=metric) for metric in METRICS}
    rows = []
    for policy in policies:
        value = df.loc[df["policy"] == policy, cost].mean()
        if not fixed_cost[0] <= value <= fixed_cost[-1]:
            continue
        j = min(int(np.searchsorted(fixed_cost, value, side="right")) - 1, len(fixed) - 2)
        weight = (value - fixed_cost[j]) / (fixed_cost[j + 1] - fixed_cost[j])
        for metric in METRICS:
            table = wide[metric]
            curve = (1 - weight) * table[fixed[j]] + weight * table[fixed[j + 1]]
            mean, lo, hi = paired_diff_ci(table[policy], curve, n_boot, seed)
            rows.append(
                {
                    "policy": policy,
                    "cost": cost,
                    "cost_value": float(value),
                    "metric": metric,
                    "adaptive": float(table[policy].mean()),
                    "fixed_curve": float(curve.mean()),
                    "diff": mean,
                    "diff_lo": lo,
                    "diff_hi": hi,
                }
            )
    return pd.DataFrame(rows)


P1_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999)


def load_p_yes(path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["id"]: [step["p_yes"] for step in row["steps"]] for row in rows}


def p1_threshold_rows(trajectories: list[dict], p_yes: dict, tau: float) -> pd.DataFrame:
    scores = step_scores(trajectories)
    rows = []
    for i, traj in enumerate(trajectories):
        values = p_yes[traj["id"]]
        stop = next((s for s, value in enumerate(values, 1) if value >= tau), len(values))
        rows.append(
            {
                "id": traj["id"],
                "policy": f"P1_t{tau}",
                "stop_step": stop,
                "searches": stop,
                "tokens": policy_cost("P1", traj, stop)["tokens"],
                "em": scores["em"][i, stop - 1],
                "f1": scores["f1"][i, stop - 1],
            }
        )
    return pd.DataFrame(rows)


def p1_threshold_curve(df: pd.DataFrame, trajectories: list[dict], p_yes: dict, taus=P1_THRESHOLDS, n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    extra = pd.concat([p1_threshold_rows(trajectories, p_yes, tau) for tau in taus], ignore_index=True)
    names = [f"P1_t{tau}" for tau in taus]
    base = df[df["policy"].str.startswith("P0_k")][["id", "policy", "searches", "tokens", "em", "f1"]]
    combined = pd.concat([base, extra[["id", "policy", "searches", "tokens", "em", "f1"]]], ignore_index=True)
    summary = extra.groupby("policy").agg(searches=("searches", "mean"), tokens=("tokens", "mean"), em=("em", "mean"), f1=("f1", "mean"))
    summary = summary.reindex(names).reset_index()
    summary.insert(1, "tau", list(taus))
    curves = pd.concat([compare_to_curve(combined, cost, n_boot, seed, policies=names) for cost in ("searches", "tokens")], ignore_index=True)
    gaps = curves[curves["metric"] == "f1"].pivot(index="policy", columns="cost", values=["diff", "diff_lo", "diff_hi"])
    gaps.columns = [f"f1_{stat}_vs_curve_{cost}" for stat, cost in gaps.columns]
    return summary.merge(gaps.reset_index(), on="policy", how="left")
