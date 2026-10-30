import re

import pandas as pd

from src.policies import ADAPTIVE, POLICIES

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
