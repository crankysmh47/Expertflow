# Test ExpertFlow on another Windows PC

This guide checks whether the private alpha installs, runs a local model, serves
chat and produces usable benchmark reports on a second machine. It does not
assume that tuning is qualified for that machine.

## 1. Prepare the kit and machine

Use Windows x64, Python 3.11 or newer, an NVIDIA GPU with its driver, and enough
RAM/disk space for your chosen model. You also need:

- The ExpertFlow alpha kit, including its wheel and `SHA256SUMS.json`.
- A local text-generation GGUF that fits the machine. A smaller model is fine.
- A compatible llama.cpp build with both `llama-cli.exe` and `llama-server.exe`,
  plus the build's runtime DLLs.

Models and native runtimes are not included in the kit. Use publisher files and
follow their licenses. For a comparison with the first PC, use identical model
and runtime files and record the settings on both machines.

If building the kit from a checkout with `uv` installed, choose a fresh output
directory:

```powershell
uv run python scripts/build_local_release.py --output release/windows-test-kit
```

Copy the whole kit to a new folder on the test PC, such as
`C:\ExpertFlow-Test\kit`. Keep the model and runtime in their own folders. Open
PowerShell in the kit folder and verify the kit before installing:

```powershell
$checksums = Get-Content .\SHA256SUMS.json -Raw | ConvertFrom-Json
foreach ($file in $checksums.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath $file.Name -Algorithm SHA256).Hash
    if ($actual -ne $file.Value) { throw "Checksum mismatch: $($file.Name)" }
}
"Kit checksums match."
```

Do not transfer a working profile from the first PC: create a new one here so
paths, hardware and load checks belong to this machine.

## 2. Install and check the runtime

Run these commands in the kit folder. Change the model and runtime paths to
your files; `$runtime` must be the directory containing both executables.

```powershell
py -m venv .expertflow-env
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required." }
.expertflow-env\Scripts\python.exe -m pip install .\expertflow_local-0.2.0a1-py3-none-any.whl
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed." }
$ef = (Resolve-Path .\.expertflow-env\Scripts\expertflow.exe).Path
$model = "C:\models\model.gguf"
$runtime = "C:\llama\bin"
& $ef local --help
& $ef local doctor --runtime $runtime --model $model
```

Doctor should identify the hardware and runtime without loading the model. If
it reports a missing DLL, fix the runtime installation or add
`--dll-dir "C:\path\to\matching\dependencies"` to **both** doctor and setup.
Use the dependency version required by your runtime. See the
[quickstart](local-quickstart.md) for details.

Record the Windows version, CPU, installed RAM, GPU and driver. These commands
also record the installed package and model identity:

```powershell
nvidia-smi
.expertflow-env\Scripts\python.exe -m pip show expertflow-local
Get-FileHash -LiteralPath $model -Algorithm SHA256
```

For a split GGUF, record hashes for every shard. Runtime content identities are
recorded in the profile created next.

## 3. Create a profile and try chat

```powershell
& $ef local setup --model $model --runtime $runtime --context 4096 --output .\profile.json
```

Continue when the result is `RUNNABLE-UNTUNED`: a real load and short generation
check passed. This is a working baseline, not a speedup claim. If it cannot
load, keep the error and logs. Try a smaller model, supported context, or explicit
`--gpu-layers` setting, and save the next attempt under a fresh profile filename.
Use that filename in the commands below. Do not use `--no-probe` for this test.

```powershell
& $ef local run --profile .\profile.json --prompt "Explain what VRAM is in two sentences." --predict 128
& $ef local run --profile .\profile.json
```

Expect a nonempty response. In interactive chat, ask a question, then a follow-up
that refers to it. Enter `/exit` to finish. Check that the owned model process
exits after chat ends. These checks test operation; they do not establish the
model's answer quality.

## 4. Test the web UI, API and shutdown

In the same terminal:

```powershell
& $ef local serve --profile .\profile.json --port 8080 --output-dir .\server-01
```

Leave it open and visit `http://127.0.0.1:8080`. Send a message in the web UI.
Then open a second PowerShell terminal **in the same kit folder**:

```powershell
$ef = (Resolve-Path .\.expertflow-env\Scripts\expertflow.exe).Path
$request = @{ messages = @(@{ role = "user"; content = "Say hello in one sentence." }); max_tokens = 128; temperature = 0 } | ConvertTo-Json -Depth 5
$response = Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType application/json -Body $request
$response.choices[0].message.content
& $ef local status --session-dir .\server-01
& $ef local stop --session-dir .\server-01
& $ef local status --session-dir .\server-01
```

Expect an API response, a running status before stopping, and a stopped status
afterward. The first terminal should finish, and the owned server should exit.
Ctrl+C in the first terminal is also supported. If port 8080 is occupied, choose
another port in both commands and URLs; leave the other application running.
Use a fresh session directory for another attempt.

## 5. Benchmark and verify the report

Return to the first terminal. Finish competing GPU work before benchmarking.
Each job needs a fresh output directory.

```powershell
& $ef local bench --profile .\profile.json --budget-seconds 600 --predict 256 --repeats 3 --output-dir .\bench-01
& $ef local verify-job --job-dir .\bench-01
& $ef local support --job-dir .\bench-01 --output .\support-bench-01.json
```

Expect a completed measurement with generation speed and first-token latency,
followed by successful receipt verification. Verification does not load a model.
An inconclusive or blocked benchmark is useful failure evidence, but does not
count as a completed performance test. Preserve its result and explanation.

Read `bench-01\report.json` for the measurement. Compare numbers only with their
model, runtime, context, settings and workload attached. Memory measurements are
sampled after model load, not the maximum memory needed during loading.

## 6. Optional: check the tuning outcome

```powershell
& $ef local tune --profile .\profile.json --budget-seconds 600 --output-dir .\tune-01
```

Tuning currently searches thread settings under narrow model/runtime/hardware
policies. `UNSUPPORTED` is expected for many second-machine setups: it means
there is no reviewed tuning policy, even if chat and benchmarking work.
`NO-MEASURABLE-GAIN` and `INCONCLUSIVE` retain the baseline. Only
`VERIFIED-IMPROVEMENT` selects a separate profile after held-out confirmation.
Do not treat ordinary launch success as tuning qualification.

## What to send back

Record pass/fail for installation, setup, two-turn chat, web UI/API, shutdown,
benchmarking and receipt verification. Include hardware/driver details, model
and runtime identities, context/settings, time spent, any help needed, and the
first failing command if there was one. Also note whether you would use
ExpertFlow again and what felt useful or confusing.

Review `support-bench-01.json` before sharing it. It is an opt-in redacted export
and uploads nothing. Keep profiles, raw logs and receipts private unless you
have checked them for paths, prompts and other sensitive information.

A successful run establishes this basic workflow on this machine. Wider hardware
support still needs context/memory checks, and user acceptance needs actual
first-user feedback. See the [support matrix](support-matrix.md) and
[pilot guide](local-product-pilot.md) for those remaining gates.
