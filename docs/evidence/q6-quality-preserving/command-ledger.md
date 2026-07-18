# Q6 Quality-Preserving Append-Only Ledger

## 2026-07-18 - Contract and isolated baseline

- Created ExpertFlow worktree `C:\models\expertflow\worktrees\q6-quality-preserving` on branch `codex/q6-quality-preserving` from `378ce57`.
- `uv sync --frozen --extra dev --extra predictor` completed successfully.
- Baseline: `uv run pytest -q` -> `178 passed, 2 skipped` in 20.16 seconds.
- Frozen design commit: `c02a851 docs: freeze Q6 quality-preserving streaming design`.
- Frozen implementation-plan commit: `4a0f83d docs: plan Q6 quality-preserving streaming`.
- TDD red: focused quality tests failed at collection because `expertflow.quality` did not exist.
- TDD green: `uv run pytest tests/test_quality_manifest.py tests/test_quality_analysis.py -q` -> 13 passed in 0.09 seconds.
- Quality-contract tooling commit: `a7af77e feat: freeze Q6 quality evaluation contract`.

## 2026-07-18 - Immutable benchmark sources

- Hugging Face dataset metadata resolved in 1.9 seconds.
- WikiText repository `Salesforce/wikitext` resolved to `b08601e04326c79dfdd32d625aee71d232d685c3`.
- MMLU repository `cais/mmlu` resolved to `c30699e8356da336a370243923dbaf21066bb9fe`.
- `uv add --optional quality "huggingface-hub>=0.36,<1" "pyarrow>=20,<22"` completed in 69.5 seconds; locked `huggingface-hub==0.36.2` and `pyarrow==21.0.0`.
- Downloaded only `wikitext-2-raw-v1/test-00000-of-00001.parquet` and the test parquet for each of the ten frozen MMLU subjects with `hf download --repo-type dataset --revision <40-character SHA> --local-dir <path>`; total command duration 6.0 seconds.
- WikiText parquet: `C:\models\expertflow\quality-data\wikitext-b08601e\wikitext-2-raw-v1\test-00000-of-00001.parquet`, SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- TDD red: `tests/test_quality_dataset.py` failed because `expertflow.quality.dataset` did not exist.
- TDD green: focused manifest, analysis, and dataset suite -> 15 passed in 0.33 seconds.
- Export command: `uv run --extra quality python scripts/prepare_quality_data.py --wikitext-parquet C:\models\expertflow\quality-data\wikitext-b08601e\wikitext-2-raw-v1\test-00000-of-00001.parquet --mmlu-root C:\models\expertflow\quality-data\mmlu-c30699e --output-dir C:\models\expertflow\quality-data\frozen-option1-v1`.
- Exported WikiText: 4,358 source rows, 2,891 non-empty rows, 1,287,656 bytes, SHA-256 `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`.
- Exported MMLU: 3,558 rows across ten subjects, 3,281,851 bytes, SHA-256 `a732878ec453dc34b3933f7cf2ffb6fbc558f97c031b8d3882e2ff2f3e4d0e8a`.
