# Release status

Updated 2026-10-07. Version **0.2.0a1** is a Windows/NVIDIA alpha for first-user
use. The engineering workflow and native checks are complete on one RTX 5060 Ti
16 GB host, driver 616.92, llama.cpp a7312ae9 with CUDA 12.8.

Implemented: checksum-verified explicit runtime installation, bounded GGUF
inspection, verified baseline profiles, fixed-budget benchmarks, reviewed exact
tuning, receipt reconstruction, opt-in support export, streaming chat, local
API/web UI and identity-safe process control. See the [support matrix](support-matrix.md)
for exact artifact/platform limits and [native receipts](evidence/local-product-20261006/report.md).
The Granite comparison found no measurable gain; its baseline remains useful.

Model-free Windows/Linux installation checks do not qualify Linux native inference.
Independent-user tasks, return use, a second GPU capacity and Linux GPU checks
remain public-release gates. No human completion or performance benefit is implied.
A TUI is deferred until observed pilot friction justifies it.

Build the kit from the repository:

```powershell
uv run python scripts/build_local_release.py --output release/local-alpha-0.2.0a1
```

The kit contains a wheel, source distribution, guides, notices and checksums.
It includes no weights or native runtime and is not a PyPI release. Use
[the quickstart](local-quickstart.md) to install outside the checkout.

Engineering validation: **988 passed, 7 skipped** in the full suite; skips require
the external patched llama.cpp source checkout. Exact Windows wheel installation
without extras, Ubuntu WSL help/doctor, source installation and reproducible kit
checks passed. Native receipts cover dense GPU/RAM long prompts, Gemma Q4 and
Granite operations; final wheel 8K/multi-turn/disconnect/cleanup passed.

[PR #1](https://github.com/crankysmh47/Expertflow/pull/1) is merged into `main`.
Windows/Linux Python 3.11/3.12 artifact contracts and Windows/Linux/macOS
model-free replay checks passed on GitHub. Native qualification remains the
Windows host described above.

Earlier results and open compiler questions are in [research status](research/STATUS.md).
