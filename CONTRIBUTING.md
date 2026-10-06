# Contributing

Branch from `main` and propose changes through a pull request. Preserve unrelated changes
and the immutable evidence/source archives for completed studies.

```powershell
uv sync --frozen --extra dev --extra quality --extra predictor
uv run --no-sync pytest -q
uv run --no-sync python -m compileall -q src/expertflow
git diff --check
```

The entire research suite can take about half an hour. For a local-product change,
first run its `tests/test_local_*.py` files (pass explicit paths on PowerShell),
then run the full suite before proposing integration. Source-contract checks
need their exact pinned external source tree. CPU tests cannot qualify native
performance or a platform. Never commit model weights, user prompts or credentials.

Portable orchestration lives in `src/expertflow/product/`; the public namespace is
`expertflow local`. Reuse services for future UI work. Compiler exactness and
historical `stock` registrations are separate contracts. A new exact tuning
policy must document arithmetic equivalence, runtime dependencies, artifact and
host/workload scope, with adversarial tests and a bounded native protocol.
Token agreement on a few samples alone does not qualify a new numerical path.

For new family/platform support, include a concrete user need, the exact artifact
and runtime identities, installed-wheel evidence, real load/run/serve checks and
scope limitations. Keep unrelated refactoring separate. A change that adds a
runtime flag cannot inherit a speed or quality claim.

For bugs, provide OS/GPU/driver, app/runtime versions, command exit and the relevant
error. Use public example prompts. Redact usernames, local paths, private prompts,
model-access tokens and client keys from shared logs. A no-gain comparison is a
valid outcome, not a reason to change an acceptance threshold.
