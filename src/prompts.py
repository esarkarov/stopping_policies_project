def format_passages(passages: list[dict], max_chars: int = 1500) -> str:
    if not passages:
        return "(none yet)"
    return "\n".join(f"[{i}] {p['title']}: {p['text'][:max_chars]}" for i, p in enumerate(passages, 1))


def query_prompt(question: str, passages: list[dict], queries: list[str], max_chars: int) -> str:
    earlier = "\n".join(f"- {q}" for q in queries) if queries else "(none)"
    return (
        "You answer a question by searching a document collection with a keyword search engine.\n\n"
        f"Question: {question}\n\n"
        f"Passages found so far:\n{format_passages(passages, max_chars)}\n\n"
        f"Earlier queries (already used, do not reuse them):\n{earlier}\n\n"
        "Write one new search query (a few keywords) for the information still missing to answer "
        "the question. Name the missing fact, entity or relation that the passages do not yet mention, "
        "and use keywords that do not appear in any earlier query.\n"
        "Reply in the form:\nQuery: <search query>"
    )


def answer_prompt(question: str, passages: list[dict], max_chars: int) -> str:
    return (
        "Answer the question using the passages below. Give a short answer of a few words "
        "with no explanation. If the passages are not enough, give your best guess.\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{format_passages(passages, max_chars)}\n\n"
        "Reply in the form:\nAnswer: <short answer>"
    )


def enough_prompt(question: str, passages: list[dict], max_chars: int) -> str:
    return (
        f"Question: {question}\n\n"
        f"Passages:\n{format_passages(passages, max_chars)}\n\n"
        "Do the passages contain all the information needed to answer the question? "
        "Reply with exactly one word: Yes or No."
    )


def decompose_prompt(question: str) -> str:
    return (
        "Break the question into the simple sub-questions that must be answered to answer it "
        "(between 2 and 4, one per reasoning step). Each sub-question asks for one missing fact "
        "(who, what, when, where, how many). Use only facts stated in the question; do not guess "
        "any answer and do not write yes/no verification questions. "
        "Write one per line, numbered like '1. ...'.\n\n"
        f"Question: {question}"
    )


def coverage_prompt(question: str, subquestions: list[str], passages: list[dict], max_chars: int) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(subquestions, 1))
    return (
        f"Question: {question}\n\n"
        f"Sub-questions:\n{numbered}\n\n"
        f"Passages:\n{format_passages(passages, max_chars)}\n\n"
        "For each sub-question, say whether the passages contain its answer. "
        "Write one line per sub-question in the form '1: yes' or '1: no'."
    )
