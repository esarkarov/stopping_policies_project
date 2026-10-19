import re

from src import prompts


def parse_query(text: str) -> str:
    match = re.search(r"Query:\s*(.+)", text, flags=re.IGNORECASE)
    query = match.group(1) if match else text.strip().splitlines()[0] if text.strip() else ""
    return query.strip().strip("\"'").strip()


def parse_answer(text: str) -> str:
    parts = re.split(r"Answer:", text, flags=re.IGNORECASE)
    body = parts[-1] if len(parts) > 1 else text
    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    return lines[0].strip().strip("\"'").rstrip(".").strip() if lines else ""


def parse_yes_no(text: str) -> bool:
    return text.strip().lower().startswith("yes")


def parse_subquestions(text: str, limit: int = 4) -> list[str]:
    found = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+\s*[.):-]\s*(.+)", line)
        if match and match.group(1).strip():
            found.append(match.group(1).strip())
    return found[:limit]


def parse_coverage(text: str, n: int) -> list[bool]:
    flags = {}
    for number, verdict in re.findall(r"(\d+)\s*[:.)-]\s*(yes|no)", text, flags=re.IGNORECASE):
        flags.setdefault(int(number), verdict.lower() == "yes")
    return [flags.get(i, False) for i in range(1, n + 1)]


def _usage(call: dict) -> dict:
    return {k: call[k] for k in ("prompt_tokens", "completion_tokens", "seconds")}


class SearchAgent:
    def __init__(self, llm, retriever, max_steps: int = 5, passages_per_step: int = 2, max_passage_chars: int = 1500):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps
        self.k = passages_per_step
        self.max_chars = max_passage_chars

    def run(self, example: dict) -> dict:
        question = example["question"]

        decomposition = self.llm.generate("decompose", prompts.decompose_prompt(question), 150)
        subquestions = parse_subquestions(decomposition["text"]) or [question]

        seen: list[dict] = []
        queries: list[str] = []
        steps = []
        for t in range(1, self.max_steps + 1):
            q_call = self.llm.generate(
                "query", prompts.query_prompt(question, seen, queries, self.max_chars), 48
            )
            query = parse_query(q_call["text"]) or question
            new = self.retriever.search(query, k=self.k, exclude={p["pid"] for p in seen})
            seen.extend(new)
            queries.append(query)

            a_call = self.llm.generate("answer", prompts.answer_prompt(question, seen, self.max_chars), 48)
            e_call = self.llm.generate("check_enough", prompts.enough_prompt(question, seen, self.max_chars), 4)
            c_call = self.llm.generate(
                "check_coverage", prompts.coverage_prompt(question, subquestions, seen, self.max_chars), 64
            )

            steps.append(
                {
                    "step": t,
                    "query": query,
                    "retrieved": [{"pid": p["pid"], "title": p["title"], "score": round(p["score"], 4)} for p in new],
                    "draft_answer": parse_answer(a_call["text"]),
                    "draft_raw": a_call["text"],
                    "p1_enough": parse_yes_no(e_call["text"]),
                    "p1_raw": e_call["text"],
                    "p3_covered": parse_coverage(c_call["text"], len(subquestions)),
                    "p3_raw": c_call["text"],
                    "calls": {
                        "query": _usage(q_call),
                        "answer": _usage(a_call),
                        "check_enough": _usage(e_call),
                        "check_coverage": _usage(c_call),
                    },
                }
            )

        return {
            "id": example["id"],
            "dataset": example["dataset"],
            "question": question,
            "answer": example["answer"],
            "gold_pids": example["gold_pids"],
            "subquestions": subquestions,
            "decompose_call": _usage(decomposition),
            "decompose_raw": decomposition["text"],
            "steps": steps,
        }
