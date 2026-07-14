# ExpertFlow Project Log

This append-oriented log records decisions, commands, evidence, failures, and next actions for the OpenAI Build Week project. Measurements are never rewritten as estimates, and estimates are never presented as measurements.

## 2026-07-14

### 21:25 PKT — Project execution started

- User approved the revised runtime direction with a qualification: llama.cpp instrumentation is a 24-hour feasibility gate, not an assumption that dynamic expert movement is easy.
- Runtime direction: pinned Gemma 4 26B A4B Q4 GGUF for real inference; minimal llama.cpp routing telemetry patch; Python for validation, analysis, simulation, recommendation, and replay.
- Q8 was removed from the critical path.
- Repository began as an unborn `main` branch containing only the untracked `expertflow_hackathon_spec_v0_11.md`.
- Created branch `codex/expertflow-stage0`; no linked worktree was created because the repository had no initial commit.
- Repository policy: model weights live outside Git under `D:\models\expertflow`; generated runs and reports remain ignored.

### 21:25 PKT — Machine and artifact preflight

- GPU after reboot: NVIDIA GeForce RTX 5060 Ti, 2,075 MiB used, 13,976 MiB free.
- System RAM previously measured: 31.1 GiB.
- Free disk at preflight: approximately 137.8 GiB on `C:` and 137.7 GiB on `D:`.
- No Q4 GGUF was found in `D:\models\expertflow` or the Hugging Face cache.
- Canonical repository: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`.
- Pinned repository revision: `21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15`.
- Canonical text-only file: `gemma-4-26B_q4_0-it.gguf`.
- Remote content length: `14,439,361,440` bytes.
- Remote ETag: `21005eb9bd80c75b5236d5b8e9828b5b887609f0cdd9158e86ea3e16044928f4`.
- Next action: commit the approved design and Stage 0 execution plan, then download and verify the pinned Q4 artifact.

