# Judge bundle

This folder contains the release assets for the Observatory submission.

## Quick verification

From the repository root:

```powershell
uv sync --extra dev
powershell -ExecutionPolicy Bypass -File submission\verify.ps1
```

Expected result:

```text
ExpertFlow submission verification passed.
events=8 demands=64 static_hits=26 lru_hits=19
```

The verification script:

- runs the checked-in replay-fixture test;
- regenerates the estimated static/LRU simulation;
- checks the expected event and hit totals;
- verifies the bundled Observatory SHA-256.

No GPU, model weights, CUDA toolkit, or private service is needed for this path.

## Open the product

Open `observatory.html` directly, or serve the folder locally:

```powershell
uv run python -m http.server 8765 --directory submission
```

Then visit:

```text
http://127.0.0.1:8765/observatory.html
```

The report has no remote scripts or assets.

## Files

| File | Purpose |
|---|---|
| `observatory.html` | Self-contained hardware, locality, policy, and recommendation report |
| `demo-script.md` | Three-minute recording script |
| `final-scorecard.md` | Final claim and evidence table |
| `verify.ps1` | CPU-only judge verification |
| `assets/observatory-top.png` | Visual QA capture from the local HTTP path |

## Important boundary

The release does not claim a runtime speedup. The final predictive experiment
was exact and prevented genuine misses, but decode TPS was 1.15% lower overall.
Live caching and predictive transfer remain disabled by default.
