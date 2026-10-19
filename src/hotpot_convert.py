import json

import pyarrow.parquet as pq


def _columns(value, *names):
    if isinstance(value, dict):
        return [value[name] for name in names]
    return [[item[name] for item in value] for name in names]


def parquet_to_official_json(parquet_path, json_path) -> int:
    rows = pq.read_table(parquet_path).to_pylist()
    out = []
    for row in rows:
        ctx_titles, ctx_sentences = _columns(row["context"], "title", "sentences")
        sf_titles, sf_ids = _columns(row["supporting_facts"], "title", "sent_id")
        out.append(
            {
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_facts": [[t, int(i)] for t, i in zip(sf_titles, sf_ids)],
                "context": [[t, list(s)] for t, s in zip(ctx_titles, ctx_sentences)],
            }
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return len(out)