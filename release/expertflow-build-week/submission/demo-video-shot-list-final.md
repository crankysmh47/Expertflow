# Final ExpertFlow recording checklist — live proof first

## Capture A — live TPS rehearsal (record this first)

1. Close unrelated GPU-heavy applications.
2. Open Windows Terminal in the repository at 1920×1080 with 22 pt text.
3. Run `.\scripts\live-tps-demo.ps1 -Mode Demo`.
4. Do not start the final voice-over until this run succeeds.
5. Preserve the whole recording, then extract these four shots:

| Final time | Keep from capture | Duration |
|---:|---|---:|
| 0:57 | Verified identity and matched configuration | 5 s |
| 1:02 | Stock result line | 6 s |
| 1:08 | ExpertFlow result line | 6 s |
| 1:14 | `LIVE RESULT` table and evidence path | 14 s |

The verified rehearsal artifact is `C:\models\expertflow\runs\live-demo\final-rehearsal-20260722\summary.json`.

## Capture B — animated deck

Open `submission/demo-video-slideshow.html` through the local server. Capture every scene for at least two seconds longer than its final slot so cuts have room.

| Final time | Scene | Exact visual |
|---:|---:|---|
| 0:00 | 1 | Personal problem and placement-compiler opening |
| 0:17 | 2 | Stock CPU boundary versus CUDA-resident route |
| 0:34 | 3 | GPT-5.6 ideation and Codex engineering loop |
| 1:28 | 4 | Authoritative ten-pair result |
| 1:39 | 5 | Rejected predictive-cache branch |
| 1:59 | 6 | Hardware-specific compiled Q6 placement |
| 2:20 | 7 | Replay, optimize/place, and serve workflow |
| 2:40 | 8 | Replay, live run, and pinned-source reproduction |
| 2:51 | 9 | Closing frame |

## Capture C — two brief proof inserts

- During scene 3, show the end of `PROJECT_LOG.md` or `uv run pytest -q` for two to three seconds.
- During scene 7, show `uv run expertflow demo --replay` completing with `status=pass` for two to three seconds.

## Edit

- Use straight cuts between terminal and deck. The deck already contains motion; do not add white flashes, zoom bursts, or template transitions.
- Keep the routed circuitry visible around every slide.
- Use only the ExpertFlow green/gold palette for titles or subtitles.
- Add one small subtitle under the live table: `ONE MATCHED LIVE REHEARSAL — AUTHORITATIVE RESULT: TEN PAIRS`.
- End at or before 2:59.
