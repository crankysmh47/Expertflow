# Compiler execution checkpoint — 2026-10-03

Branch: `ef-v2`. Starting commit: `bb4fef6`. Implementation and review fixes:
`c27711a`. The user authorized plan repairs, execution and model recovery.

Tasks 0–10 are implemented. The recovered 22,862,575,520-byte Q6 model matches
SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
Pinned binaries, companion DLLs, CUDA runtime, six patches and supplied source
revision passed identity checks. Artifact READY is separate from a live verdict.

The final whole-branch review identified four Important and three Minor issues.
One fix pass added actual launch/model/tokenization binding, independent owned
run provenance, fail-closed memory observations and teardown, an absolute HTTP
deadline, explicit legacy/compiled run modes, one reserve subtraction and a
separate completion wall metric. Regression tests observed RED before fixes.
The full CPU gate passed **481 tests**, with six external-source skips.
Compileall and diff checks passed; native PDH/NVML telemetry and idle probes pass.

Task 11 reached the terminal `VALIDATION-STOP` gate. Historical replay emits `RECORDED-DIAGNOSTIC` with no plan;
historical CLI response divergence and quality failure remain explicit. The
first live run was stopped by terminating only its verified owned server. Its
database is diagnostic-only because it predates independent-run proofs.

Fresh live gate: `C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/`.
The fresh database and raw records are under that directory. Compile stdout,
stderr and exit status are persisted there. The sequence is stock warmups and
three retained cold runs per distinct placement, pristine/fork equivalence,
exact-ineligible static diagnostics, ten selected-stock confirmations and sealed
replay. All 27 runs have distinct owned-run proof. Ten confirmations averaged
24.411 TPS (3.901% CV); sealed replay was 25.383 TPS, 3.980% above its own mean.
That exceeds the frozen absolute 2% tolerance, so the pending plan remains
diagnostic-only. No published execution plan exists. Native token, memory,
ownership and cleanup gates passed. `validate` and standalone `run --plan` were
not attempted because the terminal compile gate failed. Later tracks are blocked.
See `verification.json`; no GPU process remains running.

Final clean-checkout verification at `6ea456c`: frozen installation, 481 tests
passed / six external-source skips, compileall and diff checks passed, Git status
clean. The tested compiler code is `c27711a`; subsequent commits record evidence.
The implementation and fixed-budget verdict are complete. Live product acceptance
and later research tracks remain blocked by the selected replay tolerance gate.

Resume from the ignored `.superpowers/sdd/2026-08-28-inference-compiler-spine/`
ledger. Check live processes and artifacts before launching anything again.
Keep the pre-existing PROJECT_LOG edit, R1 report and OpenCode transcript outside
compiler commits. Do not merge, push or begin later research tracks here.
