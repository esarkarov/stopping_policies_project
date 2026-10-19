import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import load_hotpotqa
from src.hotpot_convert import parquet_to_official_json


def test_parquet_converts_to_official_format(tmp_path):
    table = pa.table(
        {
            "id": ["h1", "h2"],
            "question": ["Q1?", "Q2?"],
            "answer": ["A1", "A2"],
            "type": ["bridge", "comparison"],
            "level": ["hard", "hard"],
            "supporting_facts": [
                {"title": ["Alpha", "Beta"], "sent_id": [0, 1]},
                {"title": ["Gamma"], "sent_id": [2]},
            ],
            "context": [
                {"title": ["Alpha", "Beta", "Noise"], "sentences": [["a1. ", "a2."], ["b1."], ["n1."]]},
                {"title": ["Gamma", "Noise"], "sentences": [["g1."], ["n1."]]},
            ],
        }
    )
    parquet_path, json_path = tmp_path / "dev.parquet", tmp_path / "dev.json"
    pq.write_table(table, parquet_path)

    assert parquet_to_official_json(parquet_path, json_path) == 2
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw[0]["supporting_facts"] == [["Alpha", 0], ["Beta", 1]]
    assert raw[0]["context"][0] == ["Alpha", ["a1. ", "a2."]]

    examples = load_hotpotqa(json_path)
    gold = {p["title"] for p in examples[0]["passages"] if p["pid"] in examples[0]["gold_pids"]}
    assert gold == {"Alpha", "Beta"}