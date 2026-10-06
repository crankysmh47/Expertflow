from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from expertflow.quality.manifest import MMLU_SUBJECTS


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def export_wikitext(source: Path, target: Path) -> dict[str, int]:
    table = pq.read_table(source, columns=["text"])
    rows = [str(value) for value in table.column("text").to_pylist() if str(value).strip()]
    rendered = "\n".join(value.rstrip("\r\n") for value in rows) + "\n"
    _atomic_text(target, rendered)
    return {"source_rows": table.num_rows, "nonempty_rows": len(rows), "bytes": len(rendered.encode("utf-8"))}


def export_mmlu_rows(root: Path, target: Path) -> dict[str, int]:
    exported: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    for subject in MMLU_SUBJECTS:
        source = root / subject / "test-00000-of-00001.parquet"
        table = pq.read_table(source, columns=["question", "subject", "choices", "answer"])
        rows: list[dict[str, Any]] = []
        for row_id, row in enumerate(table.to_pylist()):
            if row["subject"] != subject:
                raise ValueError(f"subject mismatch in {source}: {row['subject']!r}")
            rows.append(
                {
                    "row_id": row_id,
                    "question": str(row["question"]),
                    "choices": [str(choice) for choice in row["choices"]],
                    "answer": int(row["answer"]),
                }
            )
        exported[subject] = rows
        total_rows += len(rows)
    _atomic_text(target, json.dumps(exported, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return {"subjects": len(exported), "rows": total_rows}

