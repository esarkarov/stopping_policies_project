from src.metrics import normalize_answer

FIXED_BUDGETS = (1, 2, 3, 4, 5)
ADAPTIVE = ("P1", "P2", "P3")
POLICIES = tuple(f"P0_k{k}" for k in FIXED_BUDGETS) + ADAPTIVE + ("Oracle",)

ABSTENTION_PHRASES = (
    "unknown",
    "not enough information",
    "no information",
    "insufficient",
    "cannot be determined",
    "not provided",
    "not specified",
    "unclear",
    "no answer",
    "not mentioned",
    "no mention",
)


def is_abstention(draft: str) -> bool:
    text = normalize_answer(draft)
    if not text or text in ("none", "na"):
        return True
    return len(text.split()) <= 10 and any(phrase in text for phrase in ABSTENTION_PHRASES)


def stop_fixed(traj: dict, k: int) -> int:
    return min(k, len(traj["steps"]))


def p1_signal(step: dict) -> bool:
    return bool(step["p1_enough"])


def p2_signal(steps: list[dict], i: int) -> bool:
    if i == 0:
        return False
    draft = steps[i]["draft_answer"]
    if is_abstention(draft):
        return False
    return normalize_answer(draft) == normalize_answer(steps[i - 1]["draft_answer"])


def p3_signal(step: dict) -> bool:
    return bool(step["p3_covered"]) and all(step["p3_covered"])


def stop_p1(traj: dict) -> int:
    for step in traj["steps"]:
        if p1_signal(step):
            return step["step"]
    return len(traj["steps"])


def stop_p2(traj: dict) -> int:
    steps = traj["steps"]
    for i in range(1, len(steps)):
        if p2_signal(steps, i):
            return steps[i]["step"]
    return len(steps)


def stop_p3(traj: dict) -> int:
    for step in traj["steps"]:
        if p3_signal(step):
            return step["step"]
    return len(traj["steps"])


def stop_oracle(correct_by_step: list[bool], f1_by_step: list[float]) -> int:
    for i, correct in enumerate(correct_by_step, 1):
        if correct:
            return i
    best = max(f1_by_step)
    return next(i for i, value in enumerate(f1_by_step, 1) if value == best)


def stop_step(policy: str, traj: dict, correct_by_step: list[bool], f1_by_step: list[float]) -> int:
    if policy.startswith("P0_k"):
        return stop_fixed(traj, int(policy[4:]))
    if policy == "P1":
        return stop_p1(traj)
    if policy == "P2":
        return stop_p2(traj)
    if policy == "P3":
        return stop_p3(traj)
    if policy == "Oracle":
        return stop_oracle(correct_by_step, f1_by_step)
    raise ValueError(f"unknown policy {policy}")


def policy_cost(policy: str, traj: dict, stop: int) -> dict:
    steps = traj["steps"][:stop]
    needed = []
    for step in steps:
        needed.append(step["calls"]["query"])
        if "query_retry" in step["calls"]:
            needed.append(step["calls"]["query_retry"])
    last_answer = steps[-1]["calls"]["answer"]

    if policy.startswith("P0_k") or policy == "Oracle":
        needed += [last_answer]
    elif policy == "P1":
        needed += [step["calls"]["check_enough"] for step in steps] + [last_answer]
    elif policy == "P2":
        needed += [step["calls"]["answer"] for step in steps]
    elif policy == "P3":
        needed += [traj["decompose_call"]]
        needed += [step["calls"]["check_coverage"] for step in steps] + [last_answer]
    else:
        raise ValueError(f"unknown policy {policy}")

    return {
        "calls": len(needed),
        "tokens": sum(call["prompt_tokens"] + call["completion_tokens"] for call in needed),
    }
