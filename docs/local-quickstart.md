# Run your local GGUF with ExpertFlow

This is the Windows/NVIDIA alpha workflow. Python 3.11+ is required.
Use a local text-generation GGUF and a compatible llama.cpp directory containing
`llama-cli.exe`, `llama-server.exe` and their runtime DLLs. Native acceptance on
other platforms and independent-user acceptance remain open; see the
[support matrix](support-matrix.md).

## Install without cloning

To build a kit from a source checkout with `uv` installed, choose a fresh directory:

```powershell
uv run python scripts/build_local_release.py --output release/windows-test-kit
```

Copy the kit to your installation folder. For testing on a second PC, use the
[Windows testing guide](windows-testing.md).

Install the built wheel from the release package:

```powershell
py -m venv .expertflow-env
.expertflow-env\Scripts\python.exe -m pip install .\expertflow_local-0.2.0a1-py3-none-any.whl
.expertflow-env\Scripts\expertflow.exe local --help
```

The wheel has no mandatory third-party Python dependencies. Development,
predictor and quality extras are unnecessary for this workflow. Add the
environment's `Scripts` directory to your shell PATH or activate it to use the
short commands below. This alpha has not been uploaded to PyPI.

## Check your runtime and model

```powershell
expertflow local doctor --runtime "C:\llama\bin" --model "C:\models\model.gguf"
```

If a Windows probe exits with a missing DLL error, install the runtime's required
libraries or supply their directory explicitly:

```powershell
expertflow local doctor --runtime "C:\llama\bin" --dll-dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
```

Use the CUDA version your runtime was built against. Do not copy the example
version into a different build blindly. Doctor does not close applications or
start a model. The runtime's current flags/version are probed; unknown builds
can be inspected/launched but do not inherit an exact tuning qualification.

To explicitly download a runtime archive, use its exact HTTPS asset URL and a
trusted upstream SHA256 digest. Place them in shell variables, then run:

```powershell
expertflow local install-runtime --url $runtimeAssetUrl --sha256 $runtimeAssetSha256 --destination "C:\llama\verified-runtime"
```

Downloads are resumable with bounded retries; archive extraction rejects links
and paths outside its staging directory. A checksum failure retains `.part`
diagnostics. Existing destinations are preserved. Models are local files in this
alpha; obtain them from the model publisher and obey their license.

## Create a baseline profile

```powershell
expertflow local setup --model "C:\models\model.gguf" --runtime "C:\llama\bin" --context 4096 --output .\my-model.json
```

Repeat `--dll-dir` during setup if your runtime needs an external dependency
directory. Setup records content identities, estimates F16 KV/memory, performs
a real load and short generation check, and writes a profile you can reuse.
`RUNNABLE-UNTUNED` means that load check passed, with no optimization claim.

Requested context is preserved. Exceeding model context or a runtime reduction
fails clearly. GPU layers/threads use the runtime defaults unless you explicitly
set `--gpu-layers` or `--threads`. You may request `--cpu-moe` when supported.
`--no-probe` saves `UNVERIFIED-PROFILE`; it is useful for offline preparation and
does not establish a successful load. Failed probes preserve the profile/logs.

## Run or chat

```powershell
expertflow local run --profile .\my-model.json --prompt "Explain what a KV cache does."
expertflow local run --profile .\my-model.json
```

The second command starts an interactive chat. Enter `/exit` or press Ctrl+C to
finish. Conversation history stays in that process; no cloud, account or prompt
upload is required. Run verifies model/runtime identities before starting a
loopback server and streams its chat response. Runtime chat-template support is
required for instruct/chat models.

## Use an existing client or the upstream web UI

```powershell
expertflow local serve --profile .\my-model.json --port 8080 --output-dir .\server-session
```

The command prints a loopback endpoint and web UI URL. Keep that terminal open.
Point your client at `http://127.0.0.1:8080/v1`, or open the upstream UI at
`http://127.0.0.1:8080`. A basic PowerShell request is:

```powershell
$request = @{ messages = @(@{ role = 'user'; content = 'Say hello.' }); max_tokens = 64; temperature = 0 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType application/json -Body $request
```

Stop with Ctrl+C, or from another terminal:

```powershell
expertflow local status --session-dir .\server-session
expertflow local stop --session-dir .\server-session
```

Saved process birth/image checks prevent an unrelated or reused PID from being
stopped. Existing listeners are never killed. This alpha serves single-user
text chat on loopback; public network exposure is outside its interface.

## Measure before spending time tuning

```powershell
expertflow local bench --profile .\my-model.json --budget-seconds 600 --output-dir .\bench-01
expertflow local tune --profile .\my-model.json --budget-seconds 600 --output-dir .\tune-01
```

Each output directory must be fresh. Jobs freeze their inputs/source/budget and
retain raw responses, token IDs, memory samples, logs and process receipts.
The default benchmark uses one process, one 16-token warmup and three 256-token
completions. It reports decode speed and observed streaming time to first token.
Its GPU memory peak is sampled after healthy load, not a load-allocation maximum.

Exact tuning currently admits only the audited Gemma Q6/Q4 and Granite Q6
artifacts with the pinned pristine runtime, Windows x86 and declared placement.
It searches eligible thread settings while preserving arithmetic/placement/KV;
it does not change quantization, offload or attention. Other models/builds return
`UNSUPPORTED` with zero tuning processes; their untuned launch remains available.

Confirmation uses separately held-out prompts, five alternating pairs, exact
token guards, fixed variance and a lower 95% gain bound above 5%. There is fixed
30-second interprocess spacing and a maximum of 14 processes. A small wall budget
may stop before confirmation. `INCONCLUSIVE` preserves the baseline, as does
`NO-MEASURABLE-GAIN`. No retries or extra samples extend a completed job.
Only `VERIFIED-IMPROVEMENT` writes a separately selected profile; the input
profile is never silently replaced. This procedure does not promise a speedup
or establish a global optimum or serving-throughput improvement.

Inspect a profile with `local report --profile .\my-model.json --json`. Saved
receipts are archived observations; run a new benchmark for fresh performance.

## Troubleshooting and local storage

Reconstruct a saved benchmark/tuning job without loading a model, or explicitly
export a summary for a bug report:

```powershell
expertflow local verify-job --job-dir "path/from/job/output"
expertflow local support --job-dir "path/from/job/output" --output support.json
```

Review the support file before sharing. It includes status, cost, context and
receipt digests; excludes prompts, responses, logs, names, paths and environment
variables; and uploads nothing. Compatibility and rollback are documented in
[runtime updates](runtime-updates.md).

| Result | Action |
| --- | --- |
| Missing DLL/dependencies | Correct runtime installation or add its dependency directory with `--dll-dir`. |
| Model cannot fit/load | Read `server.log`; explicitly reduce context or GPU layers, or choose a fitting artifact. |
| Context exceeds model limit | Choose an explicit supported context. Setup never silently shrinks it. |
| Port occupied | Choose another `--port`; the other application is left running. |
| Model/runtime changed | Create a new profile; old evidence cannot qualify changed identities. |
| GPU/driver changed | Create a new profile before measuring. |
| No gain/inconclusive | Keep the baseline profile; additional time does not authorize reusing the old job budget. |
| Bad/missing exact policy | Use untuned run/serve/bench; do not assume a control is exact from token samples alone. |

Default profiles/jobs/sessions live under `%LOCALAPPDATA%\ExpertFlow` on Windows
and `$XDG_DATA_HOME/expertflow` (or `~/.local/share/expertflow`) on Linux.
`EXPERTFLOW_HOME` overrides this location. Inspection is read-only; commands
create only their explicit outputs/state. `profiles list/show/remove` manages
profiles; removal preserves weights/runtime and old logs/revisions.

Command exit 0 means the action completed (including a neutral comparison).
Exit 2 means invalid/unsupported/blocked/inconclusive input or measurement;
130 means interactive cancellation. Historical research commands retain their
own exit conventions. Private local profiles may contain paths and runtime
metadata: share only redacted diagnostics, never prompts or credentials.

To upgrade, install the next wheel into the environment. Recheck its release
notes and runtime compatibility; preserve profiles/backups and register a new
profile for changed runtime/model identities. Removing the Python environment
removes the application; delete explicitly chosen state directories separately
if desired. Model files are never deleted by profile or application removal.
