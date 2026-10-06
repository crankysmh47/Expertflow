<p align="center">
  <img src="docs/assets/expertflow-logo.png" alt="ExpertFlow" width="640">
</p>

# ExpertFlow

**Bring a GGUF. Save a working setup. Measure it. Run it locally.**

ExpertFlow is a command-line companion for llama.cpp. It inspects your model and
hardware, saves the runtime and settings that loaded successfully, measures a
baseline within a fixed budget, and starts local chat or an API for your client.
No account, telemetry, or prompt upload is required.

The Windows/NVIDIA alpha is being qualified. The tested machine is an RTX 5060
Ti with 16 GB VRAM; another GPU capacity and independent pilot results remain
open. See the [support matrix](docs/support-matrix.md) for actual tested models
and limits. This package does not promise automatic speedups.

## Try the alpha

Python 3.11+ and a local GGUF are required. Bring a llama.cpp directory containing
both `llama-cli` and `llama-server`, including its runtime dependencies. Model
weights and native binaries are not bundled. The alpha wheel is a local build;
there is no published PyPI installation promised here.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install ./expertflow_local-0.2.0a1-py3-none-any.whl
.venv/Scripts/expertflow local doctor
.venv/Scripts/expertflow local setup --model "D:/models/model.gguf" --runtime "D:/llama/bin" --context 4096 --output profile.json
.venv/Scripts/expertflow local run --profile profile.json --prompt "Explain what VRAM is in two sentences."
```

CUDA DLLs missing on Windows? Add `--dll-dir` during setup. The
[quickstart](docs/local-quickstart.md) covers installation, runtime discovery,
interactive chat, failed loads, storage, upgrades, and removal.

## Measure and keep the result

```powershell
expertflow local bench --profile profile.json --budget-seconds 120
expertflow local tune --profile profile.json --budget-seconds 600
expertflow local report --profile profile.json --json
```

Benchmarking reports decode speed, first-token latency, raw timings, and sampled
memory. Its numbers describe that profile and workload. Tuning searches only
reviewed exact scheduling controls, then uses separate held-out confirmation
before selecting a changed profile. Unknown models or runtimes can still run
and be inspected; they do not inherit a tuning contract.

A real Granite run on the tested host finished with `NO-MEASURABLE-GAIN` after
four launches and 124.89 seconds. The baseline stayed usable and unchanged.
`INCONCLUSIVE` also keeps the baseline. An `UNSUPPORTED` tuning result means
there is no reviewed tuning policy for that setup, rather than a promise that
the model cannot run. [Native receipts and limits](docs/evidence/local-product-20261006/report.md)
include the negative result alongside operational checks.

```powershell
expertflow local verify-job --job-dir "path/from/benchmark/output"
expertflow local support --job-dir "path/from/benchmark/output" --output support.json
```

Receipt verification launches no model. Support export is opt-in and saves a
redacted summary for you to review; it sends nothing.

## Use your existing client

```powershell
expertflow local serve --profile profile.json --port 8080
```

Open `http://127.0.0.1:8080` for the upstream web UI, or point an API client at
`http://127.0.0.1:8080/v1`. Serving binds to loopback and currently targets
single-user text chat. Ctrl+C stops the owned server; saved session identities
also support `expertflow local status` and `expertflow local stop`.

A TUI is deferred until pilot feedback shows a need. The core workflow and
upstream chat UI are available without another interface to maintain.

## Contribute and follow progress

[Contributing](CONTRIBUTING.md) explains model-free tests, native qualification,
and useful bug reports. [Tasks](docs/TODO.md) tracks delivered work and open
hardware/human gates. [Changelog](CHANGELOG.md) describes compatibility changes.

The earlier compiler studies remain available in the
[research overview](docs/research-overview.md), [Product architecture](docs/PRODUCT.md),
and [current evidence status](docs/STATUS.md). They retain their original
quality gates and negative verdicts. Historical replay, judge packaging, and the
[Live dashboard](https://expertflow-zeta.vercel.app) are separate from this alpha;
the [Deployment guide](DEPLOYMENT.md) describes that earlier interface.

[Repository](https://github.com/crankysmh47/Expertflow). MIT; see [LICENSE](LICENSE)
and [third-party notices](THIRD_PARTY_NOTICES.md). Codex with GPT-5.6-sol
managed the engineering workflow of the earlier research; the human chose the evidence
gates and product direction.
