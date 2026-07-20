from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from expertflow.quality.dataset import export_mmlu_rows, export_wikitext
from expertflow.quality.manifest import MMLU_SUBJECTS


def test_export_wikitext_joins_only_nonempty_rows(tmp_path: Path):
    source = tmp_path / "wikitext.parquet"
    target = tmp_path / "wikitext.txt"
    pq.write_table(pa.table({"text": ["", " first ", "   ", "second\n"]}), source)

    metadata = export_wikitext(source, target)

    assert target.read_text(encoding="utf-8") == " first \nsecond\n"
    assert metadata["nonempty_rows"] == 2


def test_export_mmlu_preserves_parquet_row_ids(tmp_path: Path):
    root = tmp_path / "mmlu"
    for subject in MMLU_SUBJECTS:
        directory = root / subject
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "question": [f"{subject}-{index}" for index in range(11)],
                    "subject": [subject] * 11,
                    "choices": [["A", "B", "C", "D"]] * 11,
                    "answer": [index % 4 for index in range(11)],
                }
            ),
            directory / "test-00000-of-00001.parquet",
        )
    target = tmp_path / "mmlu.json"

    metadata = export_mmlu_rows(root, target)
    rows = json.loads(target.read_text(encoding="utf-8"))

    assert metadata == {"subjects": 10, "rows": 110}
    assert rows["abstract_algebra"][0]["row_id"] == 0
    assert rows["abstract_algebra"][10]["row_id"] == 10
