# Demo video fallback

If live CUDA inference is unavailable, use the model-free replay without changing the narration:

```console
uv sync --frozen
uv run expertflow demo --replay
```

Show the hash verification, then open the offline dashboard and local SVG cards. Add a visible caption: "Recorded measured evidence replay; not a live benchmark." Do not substitute simulated cache data or imply that the replay executed the model.

If the local server fails, omit the live agent response and show `docs/evidence/product-release/agentic-demo.json` with the caption "previously measured live run." Preserve the four-slot nondeterminism caveat.
