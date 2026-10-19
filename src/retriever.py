import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.data import read_jsonl

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.bm25 = BM25Okapi([tokenize(f"{p['title']} {p['text']}") for p in passages])

    @classmethod
    def from_corpus_file(cls, path) -> "BM25Retriever":
        return cls(read_jsonl(path))

    def search(self, query: str, k: int = 2, exclude=frozenset()) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        results = []
        for i in np.argsort(-scores, kind="stable"):
            passage = self.passages[i]
            if passage["pid"] in exclude:
                continue
            results.append({**passage, "score": float(scores[i])})
            if len(results) == k:
                break
        return results
