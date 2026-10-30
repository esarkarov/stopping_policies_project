import json

from src.manifest import append_manifest, build_manifest, file_sha256, git_info, ollama_info

CFG = {
    "temperature": 0,
    "think": False,
    "num_ctx": 4096,
    "data": {"seed": 42},
    "agent": {"max_steps": 5, "passages_per_step": 2, "max_passage_chars": 1500},
}


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1")
    first = file_sha256(path)
    path.write_text("x = 2")
    assert first != file_sha256(path) and len(first) == 64


def test_git_info_outside_a_repository_is_empty(tmp_path):
    assert git_info(tmp_path) == {"commit": None, "describe": None, "dirty": None}


def test_ollama_info_is_empty_when_the_server_is_unreachable():
    assert ollama_info("http://127.0.0.1:1", "qwen3.5:4b") == {"version": None, "model_digest": None}


def test_manifest_records_settings_and_source_hashes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prompts.py").write_text("PROMPT = 1")
    manifest = build_manifest(tmp_path, CFG, "qwen3.5:4b", "http://127.0.0.1:1", "hotpotqa", "main", 12)
    assert manifest["settings"]["num_ctx"] == 4096 and manifest["settings"]["max_passage_chars"] == 1500
    assert manifest["questions_to_run"] == 12 and list(manifest["sources_sha256"]) == ["src/prompts.py"]

    path = tmp_path / "run.manifest.jsonl"
    append_manifest(path, manifest)
    append_manifest(path, manifest)
    assert [json.loads(line)["dataset"] for line in path.read_text().splitlines()] == ["hotpotqa", "hotpotqa"]
