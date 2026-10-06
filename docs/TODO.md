# ExpertFlow product tasks

Updated 2026-10-07. Deliverable: useful Windows alpha for people bringing local
GGUF models and a compatible llama.cpp runtime. The complete earlier design and
research task history is preserved in [the archived roadmap](research/TODO.md).

## Windows alpha engineering

- [x] Define the local-files, single-user Windows/NVIDIA alpha contract.
- [x] Inspect hardware and bounded GGUF metadata; verify context and native dependencies.
- [x] Install explicitly chosen runtime archives with trusted checksum, resume and safe extraction.
- [x] Save a working baseline profile with real load/generation verification.
- [x] Measure baseline speed, first-token latency, sampled memory and process cleanup within fixed budgets.
- [x] Restrict tuning to reviewed exact controls; compare upstream defaults and independently confirm gains.
- [x] Retain negative outcomes and input profiles; reconstruct raw receipts without native launches.
- [x] Stream one-shot and interactive chat; serve the upstream API/web UI on loopback.
- [x] Stop owned processes safely; preserve existing listeners and unrelated processes.
- [x] Test dense GPU/RAM long prompts, Gemma Q4, Granite, multi-turn chat and disconnect/recovery on the tested host.
- [x] Review the whole implementation and fix the identified correctness issues.
- [x] Validate installed Windows wheel outside checkout without extras; check portable Linux installation.
- [x] Provide reproducible small wheel/source kit, notices, checksums and Windows/Linux model-free CI.
- [x] Organize current docs and archive historical research under `docs/research`.
- [x] Remove tool authorship, session dumps and obsolete presentation naming.
- [ ] Verify final reorganized tree and checksummed kit; integrate through a PR into `main`.

## First-user use

- [ ] The first participant installs the kit and completes the [pilot tasks](local-product-pilot.md).
- [ ] Record their timings, failures and redacted diagnostics; fix observed blockers.

The owner can be the first participant. These are human tasks; automated and
assisted checks do not count as unaided user acceptance.

## Public-release gates

- [ ] Five intended users complete the fixed pilot tasks and contribute actual feedback.
- [ ] Observe voluntary return use over two weeks.
- [ ] Qualify a second NVIDIA GPU capacity and Linux native inference in separately registered protocols.
- [ ] Address real onboarding/compatibility failures; then decide public package/release distribution.
- [ ] Evaluate an optional thin TUI only if pilot users struggle with profiles or job progress.

Broader devices, multimodal/multi-GPU serving, quality-changing optimization,
KV compression and dynamic residency are later scoped work, with separate proof
gates. Stars and user counts are goals, not substitutes for evidence of usefulness.
