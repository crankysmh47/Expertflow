# ExpertFlow recording runbook

Follow this file from top to bottom. Do not begin the final voice-over until the live benchmark and all visual captures have succeeded.

## Final deliverable

- Maximum duration: `2:59`
- Resolution: `1920 × 1080`
- Frame rate: `30 FPS`
- Format: H.264 MP4
- Presentation: screen recording plus voice-over; no webcam required
- Final narration: [`demo-video-script-final.md`](demo-video-script-final.md)

---

## Phase 1 — Prepare the machine

- [ ] Restart the computer if GPU memory is unexpectedly occupied.
- [ ] Close Chrome, games, model servers, and unrelated GPU-heavy programs.
- [ ] Keep only OBS and Windows Terminal open for the benchmark.
- [ ] Open Windows Terminal and maximize it.
- [ ] Set the terminal font to Cascadia Mono or Consolas at `22–24 pt`.
- [ ] Open the repository:

```powershell
cd C:\sem4\Expertflow
```

- [ ] Set the verified model and runtime paths:

```powershell
$env:EXPERTFLOW_MODEL_PATH = "C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf"
$env:EXPERTFLOW_LLAMA_CLI = "C:\models\expertflow\builds\llama-q6-placement-final\bin\llama-cli.exe"
```

- [ ] Confirm both files exist:

```powershell
Test-Path $env:EXPERTFLOW_MODEL_PATH
Test-Path $env:EXPERTFLOW_LLAMA_CLI
```

Both commands must print `True`.

- [ ] Clear the terminal:

```powershell
Clear-Host
```

---

## Phase 2 — Record the live TPS proof

- [ ] Start OBS recording before entering the command.
- [ ] Run the one-pair live demonstration:

```powershell
.\scripts\live-tps-demo.ps1 -Mode Demo
```

- [ ] Do not interrupt model hashing or either inference process.
- [ ] Wait for all of the following to appear:

```text
Identity:         verified
[RESULT off]
[RESULT on]
LIVE RESULT
```

- [ ] Leave the final result table visible for at least five seconds.
- [ ] Stop OBS recording.
- [ ] Copy the evidence path printed at the bottom into a temporary note.
- [ ] Preserve the complete original recording even though the waiting portions will be removed from the final edit.

### Stop gate

Do not continue if:

- identity verification fails;
- either llama process crashes;
- stock or ExpertFlow produces no result;
- the evidence summary is missing; or
- the terminal contains private information that cannot be cropped safely.

Fix or rerun the benchmark before recording the remaining video.

---

## Phase 3 — Extract the live terminal sequence

Create four clips from the successful recording:

| Final position | Keep on screen | Duration |
|---:|---|---:|
| `0:57–1:02` | Verified model/runtime identity and matched configuration | 5 s |
| `1:02–1:08` | Stock `[RESULT off]` line | 6 s |
| `1:08–1:14` | ExpertFlow `[RESULT on]` line | 6 s |
| `1:14–1:28` | Complete `LIVE RESULT` table and evidence path | 14 s |

- [ ] Use straight cuts between the clips.
- [ ] Do not fabricate or retype terminal output.
- [ ] Do not call the one-pair improvement the authoritative benchmark.
- [ ] Add this subtitle throughout the terminal sequence:

```text
ONE LIVE PAIR · AUTHORITATIVE RESULT USES TEN PAIRS
```

---

## Phase 4 — Record the animated slideshow

- [ ] Open a second terminal in the repository.
- [ ] Start the local server:

```powershell
cd C:\sem4\Expertflow
py -m http.server 8767 --bind 127.0.0.1
```

- [ ] Open this URL:

```text
http://127.0.0.1:8767/submission/demo-video-slideshow.html
```

- [ ] Set the browser to `1920 × 1080` and `100%` zoom.
- [ ] Hide bookmarks, downloads, and unrelated tabs.
- [ ] Start on scene 1.
- [ ] Start OBS recording.
- [ ] Capture each scene for at least three seconds longer than its required duration.
- [ ] Use Arrow Down or Page Down exactly once to advance.
- [ ] Use the `REPLAY` button before recapturing an animation.
- [ ] Stop OBS only after scene 9 has remained visible for five seconds.

### Slideshow timeline

| Final timeline | Scene | Visual |
|---:|---:|---|
| `0:00–0:17` | 1 | Personal problem and ExpertFlow opening |
| `0:17–0:34` | 2 | Hidden CPU/GPU boundary |
| `0:34–0:57` | 3 | GPT-5.6 and Codex workflow |
| `0:57–1:28` | Terminal | Live matched TPS proof |
| `1:28–1:39` | 4 | Authoritative ten-pair result |
| `1:39–1:59` | 5 | Predictive cache rejected by measurement |
| `1:59–2:20` | 6 | Compiled Q6 placement |
| `2:20–2:40` | 7 | Runnable product interfaces |
| `2:40–2:51` | 8 | Replay, live test, and rebuild paths |
| `2:51–2:59` | 9 | Closing frame |

---

## Phase 5 — Record two proof inserts

### Codex engineering evidence

- [ ] Open [`../PROJECT_LOG.md`](../PROJECT_LOG.md).
- [ ] Scroll slowly through a section containing experiments, decisions, commands, and measurements.
- [ ] Record approximately five seconds.
- [ ] Use two or three seconds of this during scene 3.

### Judge replay

- [ ] Return to Windows Terminal.
- [ ] Run:

```powershell
cd C:\sem4\Expertflow
uv run expertflow demo --replay
```

- [ ] Record the output containing:

```text
"status": "pass"
"evidence_hashes_verified": true
```

- [ ] Use two or three seconds of this during scene 7.

---

## Phase 6 — Build the silent edit

- [ ] Place the slideshow scenes on the exact timeline above.
- [ ] Insert the four terminal clips at `0:57–1:28`.
- [ ] Insert the project-log shot during scene 3.
- [ ] Insert the replay result during scene 7.
- [ ] Use only straight cuts.
- [ ] Do not add white flashes, template transitions, or extra zoom effects.
- [ ] Confirm every benchmark value remains readable at normal playback speed.
- [ ] Confirm the silent edit ends at or before `2:59`.

---

## Phase 7 — Record the voice-over

- [ ] Open [`demo-video-script-final.md`](demo-video-script-final.md).
- [ ] Play the silent edit while recording the narration.
- [ ] Speak as if explaining the project to another developer.
- [ ] Pause briefly after the live result and the authoritative result.
- [ ] Emphasize these ideas naturally:

```text
I wanted the higher-quality model.
The problem was placement.
Here is something concrete.
The evidence changed the product.
```

- [ ] If a sentence feels unnatural, preserve its meaning but say it in your own words.
- [ ] Keep the narration below `2:59`.
- [ ] Keep voice peaks near `−6 dB`.
- [ ] Use no music, or keep it quiet enough that every benchmark number is unmistakable.

---

## Phase 8 — Export and inspect

- [ ] Export as H.264 MP4.
- [ ] Use `1920 × 1080`, `30 FPS`, and approximately `10–16 Mbps` video bitrate.
- [ ] Use AAC audio at `192 kbps` or higher.
- [ ] Watch the exported file once from beginning to end.
- [ ] Confirm the final duration is below three minutes.
- [ ] Confirm stock and ExpertFlow live results are both visible.
- [ ] Confirm `28.13 TPS`, `22.967 TPS`, and `+22.48%` are readable.
- [ ] Confirm Codex and GPT-5.6 are explicitly discussed.
- [ ] Confirm the GitHub repository or replay command appears on screen.
- [ ] Confirm no tokens, private tabs, unrelated files, or personal notifications appear.

---

## Phase 9 — Upload

- [ ] Upload the MP4 to YouTube.
- [ ] Set visibility to `Public` or `Unlisted`, never `Private`.
- [ ] Wait for HD processing to complete.
- [ ] Open the link in a private/incognito window and confirm it plays.
- [ ] Copy the final YouTube URL into the Devpost submission.

## Emergency fallback

If the live benchmark cannot be recorded cleanly, do not fake it. Record this instead:

```powershell
uv run expertflow demo --replay
```

Label it visibly:

```text
HASH-VERIFIED RECORDED EVIDENCE REPLAY · NOT A LIVE BENCHMARK
```

The preferred submission remains the real live TPS path above.
