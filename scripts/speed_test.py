import os
import time

import ollama

MODEL = os.environ.get("MODEL", "qwen3.5:4b")
client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

PROMPT = (
    "Question: Who directed Inception?\n"
    "Evidence: Inception is a 2010 film directed by Christopher Nolan.\n"
    "Give a short answer after 'Answer:'."
)


def timed_call(think: bool) -> None:
    start = time.time()
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        think=think,
        options={"temperature": 0},
    )
    elapsed = time.time() - start
    text = response["message"]["content"].strip().replace("\n", " ")
    print(f"think={think!s:5} | {elapsed:6.1f} s | {text[:80]!r}")


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    timed_call(think=False)  # first call also loads the model into memory
    for think in (False, True):
        timed_call(think)
