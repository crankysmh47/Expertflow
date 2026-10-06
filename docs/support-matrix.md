# Local-product support and qualification

Updated 2026-10-06. Windows/NVIDIA alpha implementation; final release gate open.
The user can be the first participant. A second NVIDIA machine is not currently
available; qualification on another capacity and Linux remain pending.

| Surface | Evidence today | Release status |
| --- | --- | --- |
| Installed wheel, no research checkout | Windows empty-directory help/doctor tested without extras | Implemented; final rebuilt artifact checks pending |
| GGUF v2/v3 little-endian metadata/profiles | Bounded parser tests; real Granite Q6 inspected | Implemented |
| Granite Q6 on RTX 5060 Ti 16 GB, driver 616.92, pristine a7312ae9/CUDA 12.8 | Real setup/load, measured diagnostic, API chat/UI/status/identity-checked stop passed | One tested host/artifact combination |
| Exact thread tuning | Fresh Granite search closed NO-MEASURABLE-GAIN: four processes, 124.89s | Baseline retained; no new gain claim |
| Gemma Q6/Q4 local-product workflow | Historical compiler acceptance only | New local-product native workflow unverified |
| Dense Qwen2.5 0.5B Q4_K_M | Real GPU-layer 8K and RAM 4K prompts, two chat turns, disconnect/recovery and cleanup | Operational checks passed on this host; no tuning policy |
| Linux Python workflow | Portable code and planned CI | Linux execution/native qualification pending |
| Second GPU capacity | No machine available | Unverified |
| Independent human tasks | Pilot task sheet ready; first participant identified | Results/timings pending |
| TUI | Conditional on observed friction | Deferred |
| Other runtimes/families/devices | Bring-your-own discovery/untuned launch where the runtime supports it | No inherited tuning/native support claim |

Successful model-free tests do not establish GPU support. A setup load receipt
does not establish performance, quality, independent-user usefulness or serving
capacity. New versions/drivers/workloads require new scoped qualification;
historical compiler stock results remain governed by [STATUS.md](STATUS.md).

The native target for the initial alpha is single-user text chat, local files,
one NVIDIA GPU and loopback serving. Runtime defaults are resolved by upstream;
the app records the requested profile and actual context/props/logs. Multi-GPU,
multimodal, public network service and other backends require their own support
slices. Source data and negative results remain available in the evidence pages.
