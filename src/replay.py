import json
import math
import urllib.request

from src import prompts
from src.agent import parse_yes_no


def passages_seen(traj: dict, corpus: dict, upto: int) -> list[dict]:
    return [corpus[r["pid"]] for step in traj["steps"][:upto] for r in step["retrieved"]]


def yes_probability(top_logprobs: list[dict]) -> float:
    yes = no = 0.0
    for item in top_logprobs:
        token = item["token"].strip().lower()
        if token == "yes":
            yes += math.exp(item["logprob"])
        elif token == "no":
            no += math.exp(item["logprob"])
    return yes / (yes + no) if yes + no > 0 else 0.0


def chat_logprobs(host: str, model: str, prompt: str, options: dict, top: int = 20, timeout: int = 300):
    base = host if host.startswith("http") else f"http://{host}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": top,
        "options": options,
    }
    request = urllib.request.Request(
        f"{base.rstrip('/')}/api/chat", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    logprobs = reply.get("logprobs") or []
    return reply["message"]["content"] or "", (logprobs[0]["top_logprobs"] if logprobs else [])


def replay_trajectory(traj: dict, corpus: dict, call, max_chars: int = 1500) -> dict:
    steps = []
    for i, step in enumerate(traj["steps"], 1):
        prompt = prompts.enough_prompt(traj["question"], passages_seen(traj, corpus, i), max_chars)
        text, top = call(prompt)
        steps.append(
            {
                "step": step["step"],
                "p_yes": yes_probability(top),
                "text": text,
                "matches_stored": parse_yes_no(text) == step["p1_enough"],
            }
        )
    return {"id": traj["id"], "steps": steps}
