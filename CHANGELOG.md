# Changelog

## 0.2.0a1 — Windows local-model alpha (2026-10-07)

- Inspect GGUF files/hardware and explicitly install checksum-verified runtimes.
- Save a verified baseline profile and launch streaming chat or loopback API/web UI.
- Measure fixed workloads and reconstruct receipts without loading a model.
- Tune reviewed exact thread controls with held-out confirmation; preserve negative outcomes.
- Export opt-in redacted support summaries and stop owned processes using OS identity checks.
- Build a small checksummed wheel/source kit; archive historical research under `docs/research`.

Native acceptance is limited to the tested Windows/NVIDIA host and artifacts.
Independent-user results, return use, Linux GPU and a second GPU capacity remain
public-release gates. Models/runtimes are external; no PyPI release or speedup
claim accompanies this alpha. See [support](docs/support-matrix.md).
