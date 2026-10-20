def make_traj(drafts, p1=None, p3=None, answer="Paris", aliases=None, gold=("g1", "g2"), retrieved=None, qid="q1"):
    n = len(drafts)
    p1 = p1 or [False] * n
    p3 = p3 or [[False, False]] * n
    retrieved = retrieved or [[f"x{i}a", f"x{i}b"] for i in range(n)]
    steps = []
    for i in range(n):
        call = lambda tokens: {"prompt_tokens": tokens, "completion_tokens": 1, "seconds": 1.0}
        steps.append(
            {
                "step": i + 1,
                "query": f"query {i}",
                "retrieved": [{"pid": pid, "title": pid, "score": 1.0} for pid in retrieved[i]],
                "draft_answer": drafts[i],
                "draft_raw": f"Answer: {drafts[i]}",
                "p1_enough": p1[i],
                "p1_raw": "Yes" if p1[i] else "No",
                "p3_covered": p3[i],
                "p3_raw": "",
                "calls": {
                    "query": call(10),
                    "answer": call(20),
                    "check_enough": call(30),
                    "check_coverage": call(40),
                },
            }
        )
    return {
        "id": qid,
        "dataset": "hotpotqa",
        "question": "Q?",
        "answer": answer,
        "answer_aliases": aliases or [],
        "gold_pids": list(gold),
        "subquestions": ["a?", "b?"],
        "decompose_call": {"prompt_tokens": 50, "completion_tokens": 5, "seconds": 1.0},
        "decompose_raw": "",
        "steps": steps,
    }
