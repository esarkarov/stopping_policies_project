import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_FILES = ("src/agent.py", "src/prompts.py", "src/llm.py", "src/retriever.py")


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info(root) -> dict:
    def run(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit else None
    return {
        "commit": commit or None,
        "describe": (run("describe", "--tags", "--always") or None) if commit else None,
        "dirty": None if status is None else bool(status),
    }


def ollama_info(host: str, model: str) -> dict:
    base = host if host.startswith("http") else f"http://{host}"
    info = {"version": None, "model_digest": None}
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/version", timeout=5) as response:
            info["version"] = json.load(response).get("version")
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/tags", timeout=5) as response:
            for item in json.load(response).get("models", []):
                if item.get("name") in (model, f"{model}:latest"):
                    info["model_digest"] = item.get("digest")
    except (OSError, ValueError):
        pass
    return info


def build_manifest(root, cfg: dict, model: str, host: str, dataset: str, split: str, to_run: int) -> dict:
    root = Path(root)
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "split": split,
        "questions_to_run": to_run,
        "model": model,
        "settings": {
            "temperature": cfg["temperature"],
            "think": cfg["think"],
            "num_ctx": cfg.get("num_ctx", 4096),
            "seed": cfg["data"]["seed"],
            "max_steps": cfg["agent"]["max_steps"],
            "passages_per_step": cfg["agent"]["passages_per_step"],
            "max_passage_chars": cfg["agent"].get("max_passage_chars", 1500),
        },
        "git": git_info(root),
        "ollama": ollama_info(host, model),
        "sources_sha256": {name: file_sha256(root / name) for name in SOURCE_FILES if (root / name).exists()},
    }


def append_manifest(path, manifest: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
