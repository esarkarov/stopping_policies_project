def pool(*parts: list[dict]) -> list[dict]:
    seen, pooled = set(), []
    for trajectories in parts:
        for traj in trajectories:
            if traj["id"] in seen:
                raise ValueError(f"question {traj['id']} appears in more than one split")
            seen.add(traj["id"])
            pooled.append(traj)
    return pooled
