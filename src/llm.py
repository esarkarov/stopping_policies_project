import hashlib
import json
import os
import time
from pathlib import Path

import ollama


class OllamaLLM:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        temperature: float = 0.0,
        think: bool = False,
        num_ctx: int = 4096,
        seed: int = 42,
        cache_path=None,
    ):
        self.model = model
        self.temperature = temperature
        self.think = think
        self.num_ctx = num_ctx
        self.seed = seed
        self.client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row.pop("key")] = row
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, max_tokens: int) -> str:
        payload = [self.model, self.temperature, self.think, self.num_ctx, self.seed, max_tokens, prompt]
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()

    def generate(self, kind: str, prompt: str, max_tokens: int) -> dict:
        key = self._key(prompt, max_tokens)
        if key in self.cache:
            return {**self.cache[key], "cached": True}

        start = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    think=self.think,
                    options={
                        "temperature": self.temperature,
                        "seed": self.seed,
                        "num_ctx": self.num_ctx,
                        "num_predict": max_tokens,
                    },
                )
                break
            except (ollama.ResponseError, ConnectionError) as error:
                if attempt == 2:
                    raise
                print(f"model call failed ({error}); retrying")
                time.sleep(5)

        result = {
            "text": response.message.content or "",
            "prompt_tokens": response.prompt_eval_count or 0,
            "completion_tokens": response.eval_count or 0,
            "seconds": round(time.perf_counter() - start, 3),
        }
        self.cache[key] = result
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, **result}, ensure_ascii=False) + "\n")
        return {**result, "cached": False}
