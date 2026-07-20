# Q6 selected-static command ledger

All timestamps are Asia/Karachi (UTC+05:00). Raw runtime artifacts are retained under `C:\models\expertflow\runs\q6-selected-static`; the 22.86 GB GGUF remains outside Git.

## Isolation and verification

- Created ExpertFlow branch/worktree `codex/q6-selected-static` at Q1b commit `834cf3d` and llama.cpp branch/worktree `codex/q6-selected-static-llama` at Q1b commit `38dce264`; prior worktrees were not modified.
- Baseline ExpertFlow verification: `208 passed, 5 skipped`.
- Q6 identity: `C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf`, 22,862,575,520 bytes, SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
- Built Release CUDA llama.cpp with Ninja, MSVC 19.39/v143, CUDA 12.8.93, and `GGML_CUDA=ON`. Two configure attempts failed before compilation: one lacked the Windows SDK resource compiler outside a developer environment; one selected MSYS GCC because of PATH order. The passing configure used `vcvars64.bat`, explicit MSVC compilers, and the bundled Ninja path.

## Stage 1 diagnostic profiling

- Added the disabled-by-default `LLAMA_EXPERTFLOW_SPLIT_PROFILE` aggregate profiler to `ggml/src/ggml-backend.cpp`. It uses fixed storage, synchronizes split completion intentionally, and emits JSON only during scheduler teardown. Its timings are diagnostic and never mixed with throughput results.
- Ran stock-equivalent Q6 at `-ngl 99 --cpu-moe`, 128 generated tokens, with the profiler enabled. Raw profile: `stage1-profile/split-profile.json` (SHA-256 `119a00a72a403b824bfd4f778cfc596a5a5c12e853dd4eab79e53950a7f90563`).
- Placement debug showed CUDA router/top-k, authoritative routed-expert tensors and selected `MUL_MAT_ID`/activation on CPU, then output copied back to CUDA. Selected splits were 2, 4, 32, and 42 for layers 0, 1, 15, and 20.

## Stage 2 static proof

- Enabled the pre-existing Q1b implementation only with `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0,1,15,20`; no new cache, scheduler, kernel, placement, or arena code was added.
- Each Q6 layer copied gate/up 416,317,440 bytes, down 269,615,104 bytes, and scale 512 bytes once. Exact four-layer shadow allocation: 2,743,732,224 bytes.
- The 128-token proof completed without NaN, CUDA error, hidden full expert duplicate, growth, or teardown failure. Raw proof: `stage2-proof-r1`.

## Stage 3 matched runs

- Harness: `scripts/run_q6_selected_static_pairs.ps1`; ten cold-process pairs, alternating `OFF/ON` then `ON/OFF`, prompt `Caching.`, 512 generated tokens, context 2048, `-ngl 99 --cpu-moe`, 12 threads, seed 42, temperature 0, ignore EOS.
- The first complete 10-pair diagnostic used the build default for CUDA graphs. It is preserved at `matched-cli-10x512` but excluded before viewing a protocol-valid verdict because the approved static proof requires CUDA graphs disabled.
- The harness contract was corrected test-first to force `GGML_CUDA_DISABLE_GRAPHS=1` identically in OFF and ON. The corrected matched run is the sole authoritative performance set.

## Stage 4 quality

- Frozen held-out corpus: `C:\models\expertflow\quality-data\q1b-heldout\wikitext-103-validation.txt`, SHA-256 `8ef749789ca0693435d20b3f81d5638c19edcebc5a68586dcf09bdf47ef9542f`; 8 chunks at context 2048, batch 512, microbatch 1, seed 42. The NLL sidecar records 8,184 paired scored tokens.
- An initial OFF quality run and interrupted ON attempts used the build-default graph setting; a subsequent interrupted OFF run used the earlier Q1 development corpus instead of the independently frozen Q1b held-out corpus. All are quarantined from the final comparison. The authoritative OFF/ON comparison uses the Q1b held-out corpus and forces `GGML_CUDA_DISABLE_GRAPHS=1` in both processes.
- The same frozen 100-item zero-shot Q1b MMLU manifest, temperature 0, seed 42, prompt cache disabled, and grammar-constrained `A-D` output is used after the perplexity gate.

## Authoritative results and cleanup

- Corrected performance raw artifact: `matched-cli-10x512-nographs/raw-results.json`, 73,582 bytes, SHA-256 `22d8a5ecebb5a0759af0f96b0cf903629b3b3fda711cfb91bcc41f8e837ce491`.
- Analysis command: `python scripts/analyze_q6_selected_static.py --input <raw-results.json> --output-json docs/evidence/q6-selected-static/performance-summary.json --output-csv docs/evidence/q6-selected-static/run-pairs.csv`.
- Corrected matched result: 19.95 OFF versus 21.44 ON decode TPS; paired +7.5326%, bootstrap 95% `[+4.6570%, +10.4180%]`.
- Streaming latency command used `scripts/run_q6_selected_static.py`, one 512-token pair, ports 18321/18322. First launch failed with Windows `0xC0000135` because the child DLL search path omitted the runtime and CUDA directories. A test-first environment fix was committed and the rerun completed; raw SHA-256 `d63187a60bf158d1dd7a13540f71fb4e2d744f37828d64aae6e9aca355c0c376`.
- PPL analysis command: `python scripts/analyze_q1b_nll.py --reference <off/nll.jsonl> --candidate <on/nll.jsonl> --block-size 128 --bootstrap-samples 10000 --seed 20260718 --threshold 0.01`. Result: 16,096.5661 OFF, 14,944.0877 ON, -7.1598%, 95% `[-11.4499%, -2.5887%]`, PASS.
- Started one local `llama-server` at a time with `GGML_CUDA_DISABLE_GRAPHS=1`, then ran `python scripts/run_q1b_mmlu.py --manifest docs/evidence/q6-quality-preserving/quality-manifest.json`. OFF scored 49/100 and ON scored 52/100. Repeated the 15 disagreements only; all 15 matched the original ON identity, prediction, token IDs, and content.
- Stopped the final server explicitly. No llama CLI, server, or perplexity process remained; system GPU usage returned to the pre-run application baseline.
- Decision: quality and conditional-range performance gates pass, but the quantified path gate fails. With a 256 MiB reserve, only 15 additional full arenas fit from the measured ON peak; optimistic linear scaling reaches 27.0275 TPS, below 28.709, with no additional low-sensitivity layers verified. Terminal verdict: `Q6 PERFORMANCE STOP`.
