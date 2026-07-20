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

## 2026-07-18 - Q1 static CUDA island

- Created llama.cpp worktree `C:\models\expertflow\worktrees\llama-q6-quality-preserving` on branch `codex/q6-quality-preserving-llama` at exact upstream `a7312ae94f801fc9c6786dc56e38df57b964f697`.
- TDD red: `tests/test_q6_quality_static_source_contract.py` -> 3 failed and 1 passed against pristine source.
- Restored only `src/llama-context.cpp` and `ggml/src/ggml-backend.cpp`; excluded all historical diagnostic bypasses.
- TDD green: source contract -> 4 passed in 0.05 seconds.
- Release CUDA build used MSVC `19.39.33523.0`, v143 `14.39.33519`, CUDA `12.8.93`, Ninja, and `CMAKE_CUDA_ARCHITECTURES=120a-real`; full build completed in 325.4 seconds.
- Built `llama-cli`, `llama-perplexity`, and `llama-tokenize`. `llama-cli.exe` SHA-256: `07ca190fc192efcfb6a40eb5c14c0b8bde1ebe254811e5d2e4e4f23cfd1041cd`. `llama-perplexity.exe` SHA-256: `4867d2463a6e132a63222bc2de36f2f36412f1a906df181d5c13f068d43551cf`.
- Pristine and patched-feature-disabled detached smokes used the same Q4 model, `-ngl 10`, context 512, greedy seed 42, and eight generated tokens. Their normalized generated segments matched exactly.
- Feature-enabled detached smoke completed and returned GPU memory to its pre-run level. Runtime log proved the complete persistent layer-0 bundle: gate/up `285,474,816` bytes, down `142,737,408` bytes, scale `512` bytes; total `428,212,736` bytes (`408.375488 MiB`).
- Isolated llama.cpp commit: `29857466d39cc532cefc1633ac14e521849541fe feat: restore disabled static CUDA expert island`.
- Frozen Q4 model identity: 14,439,361,440 bytes, SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- Frozen quality manifest SHA-256: `4b4a1823e8dd0335e5e788657a8106177af30ac5a660738a8b4f25e3609dca61`.
- Focused manifest, dataset, analysis, and evidence suite -> 17 passed; source-contract module skipped in that invocation because `EXPERTFLOW_LLAMA_SOURCE` was absent. Its explicit environment-enabled invocation passed 4/4 above.

## 2026-07-18 - Q1 scored quality STOP

- Built `llama-server` before scored work so the frozen runtime identity could cover the planned MMLU path. No scored candidate result had been inspected. Added `llama-server.exe` and `llama-server-impl.dll` to the runtime artifacts and regenerated the manifest once before scoring.
- The superseding frozen manifest SHA-256 is `294ccc4e6ef9da9d80ee15ac89d989d6d1eaa44e28bc0a043ab219d436a18719`. The earlier `4b4a...` value above is retained as append-only history and was never used for a scored comparison.
- Pre-score determinism: three fresh feature-on router-probe runs completed in 6.0117, 6.0154, and 6.0089 seconds. Each contained 6 prompt tokens, 16 generated tokens, and 630 router events. Prompt tokens, generated tokens, and normalized router events were exact across all three.
- Feature-off WikiText command: matched `llama-perplexity`, `--chunks 4`, `-c 2048`, `-ngl 10`, `--threads 12`, `--batch-size 512`, `--ubatch-size 1`, `--no-warmup`; static-island flag absent.
- Feature-off reported sequence: `1072.9325, 1411.4224, 1433.1798, 1176.7406`; final `1176.7406 +/- 92.49670`. Raw output SHA-256 `cbc0e6bca59c9f0f4abc40ac34c5949b3d96c281a92b0b4ac4ef15ab2e0b2a1b`.
- Feature-on used the identical command with only `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0`. Reported sequence: `1046.0991, 1388.2249, 1429.3230, 1183.6406`; final `1183.6406 +/- 93.22685`. Raw output SHA-256 `55117c344c20d0761fec0522c8269e816f3d6c4d727af71e4a24c46a8ae665a6`.
- Calculated relative change: `(1183.6406 / 1176.7406) - 1 = 0.0058636542`, or `+0.586365%`. Frozen maximum: `+0.500000%`. Result: FAIL by 0.086365 percentage points.
- Stop rule applied immediately. MMLU, six-prompt long generation, reactive Q4, Q6 expansion, and prediction were not run. No threshold was changed and no repeat was selected after observing the failure.
- Focused final evidence validation with `EXPERTFLOW_LLAMA_SOURCE` set -> 22 passed in 1.38 seconds.
- First full-suite invocation used only the `quality` extra and failed during collection with five repository-root import errors (`scripts`/`tests` not found). This was an invocation/environment failure; no test body ran.
- Corrected reproducible command: `$env:PYTHONPATH="$PWD;$PWD\src"; uv run --extra dev --extra predictor --extra quality pytest -q` -> 196 passed, 3 expected source-contract skips in 4.15 seconds.
