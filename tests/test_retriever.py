from src.retriever import BM25Retriever, tokenize

PASSAGES = [
    {"pid": "a", "title": "Inception", "text": "Inception is a 2010 film directed by Christopher Nolan."},
    {"pid": "b", "title": "Christopher Nolan", "text": "Christopher Nolan is a British-American film director."},
    {"pid": "c", "title": "Paris", "text": "Paris is the capital of France."},
    {"pid": "d", "title": "Berlin", "text": "Berlin is the capital of Germany."},
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World-2010!") == ["hello", "world", "2010"]


def test_top_result_is_most_relevant():
    result = BM25Retriever(PASSAGES).search("who directed Inception", k=1)
    assert result[0]["pid"] == "a"


def test_exclude_returns_only_new_passages():
    retriever = BM25Retriever(PASSAGES)
    first = retriever.search("Christopher Nolan film", k=2)
    second = retriever.search("Christopher Nolan film", k=2, exclude={p["pid"] for p in first})
    assert len(second) == 2
    assert not {p["pid"] for p in first} & {p["pid"] for p in second}


def test_k_larger_than_corpus():
    assert len(BM25Retriever(PASSAGES).search("capital", k=10)) == len(PASSAGES)


def test_search_is_deterministic():
    retriever = BM25Retriever(PASSAGES)
    a = [p["pid"] for p in retriever.search("capital of France", k=3)]
    b = [p["pid"] for p in retriever.search("capital of France", k=3)]
    assert a == b
