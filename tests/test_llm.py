from types import SimpleNamespace

from src.llm import OllamaLLM


class StubClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="Answer: x"), prompt_eval_count=7, eval_count=3)


def make_llm(tmp_path):
    llm = OllamaLLM("qwen3.5:4b", host="http://stub", cache_path=tmp_path / "cache.jsonl")
    llm.client = StubClient()
    return llm


def test_generate_sends_deterministic_options_and_thinking_off(tmp_path):
    llm = make_llm(tmp_path)
    result = llm.generate("answer", "prompt", 48)
    call = llm.client.calls[0]
    assert call["think"] is False
    assert call["options"]["temperature"] == 0.0 and call["options"]["num_predict"] == 48
    assert result["text"] == "Answer: x" and result["prompt_tokens"] == 7 and result["completion_tokens"] == 3
    assert result["cached"] is False


def test_cache_is_used_and_survives_restart(tmp_path):
    first = make_llm(tmp_path)
    first.generate("answer", "prompt", 48)
    second = make_llm(tmp_path)
    again = second.generate("answer", "prompt", 48)
    assert again["cached"] is True and second.client.calls == []
    second.generate("answer", "another prompt", 48)
    assert len(second.client.calls) == 1
