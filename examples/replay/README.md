# Replay fixture

This fixture contains eight layer-0 events copied from the real Gemma 4 Q4 probe trace. The source run used the public prompt in `configs/baseline-prompt.txt`; no prompt text or model weights are included here.

`trace.jsonl` is labeled `previously_measured` in `expected.json`. Its recorded SHA-256 uses UTF-8 with LF-normalized line endings so the fixture has the same identity in Windows and Unix checkouts. The cache outcomes remain estimates: the fixture only makes the simulator easy to reproduce on a CPU-only machine.

Run the comparison with an eight-slot per-layer budget:

```powershell
uv run expertflow simulate examples\replay\trace.jsonl `
  --capacity-per-layer 8 `
  --output replay-simulation.json
```

The static-hotset and LRU totals should match `expected.json`. The complete 1,350-event run stays outside Git under `C:\models\expertflow\runs\q4-probe` on the development machine.
