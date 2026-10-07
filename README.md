
# ExpertFlow

**Save a working local-model setup. Know how it performs. Use it again.**

ExpertFlow is a setup and benchmarking tool for running GGUF language models
with llama.cpp. It turns your model, runtime and launch settings into a reusable
profile, so you can keep track of what works on your hardware without rebuilding
the same command each time.

- **Check your setup:** inspect hardware and dependencies, then verify that the model loads.
- **Measure performance:** benchmark generation speed, first-token latency and sampled memory within a time budget.
- **Run your model:** chat in the terminal, use llama.cpp's web UI, or serve a local API for your existing client.

Everything runs locally. No account, telemetry or prompt upload is required.

## Get started

The current release is a **private Windows/NVIDIA alpha**, tested on one
16 GB GPU. See the [support matrix](docs/support-matrix.md) for tested models
and remaining checks.

You need Python 3.11+, a local GGUF model, and a llama.cpp directory containing
`llama-cli.exe`, `llama-server.exe` and their dependencies. Models and native
runtimes are supplied separately.

From an unpacked alpha kit, run in PowerShell:

```powershell
py -m venv .expertflow-env
.expertflow-env\Scripts\python.exe -m pip install .\expertflow_local-0.2.0a1-py3-none-any.whl
.expertflow-env\Scripts\expertflow.exe local setup --model "C:\models\model.gguf" --runtime "C:\llama\bin" --context 4096 --output profile.json
.expertflow-env\Scripts\expertflow.exe local run --profile profile.json --prompt "Explain what VRAM is in two sentences."
```

The [quickstart](docs/local-quickstart.md) covers building the kit from source,
installation and troubleshooting. The alpha is not published on PyPI.
For a fresh PC, follow [testing on another Windows machine](docs/windows-testing.md).

## Measure or serve

Use the same saved profile for both:

```powershell
.expertflow-env\Scripts\expertflow.exe local bench --profile profile.json --budget-seconds 600
.expertflow-env\Scripts\expertflow.exe local serve --profile profile.json --port 8080
```

Serving opens the upstream web UI at `http://127.0.0.1:8080` and an API at
`http://127.0.0.1:8080/v1`. It currently targets single-user text chat.

Optional tuning tests thread settings for a small set of qualified models and
runtimes. A changed profile is selected only after independent confirmation;
otherwise you keep the baseline. Broad automatic tuning is still on the roadmap,
and a speedup is not guaranteed.

## Learn more

[Documentation](docs/README.md) · [Architecture](docs/PRODUCT.md) ·
[Roadmap](docs/TODO.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

Earlier compiler studies and their results live in the [research archive](docs/research/README.md).

MIT licensed. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
