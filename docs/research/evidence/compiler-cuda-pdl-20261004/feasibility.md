# PDL scheduling feasibility: qualified, performance unmeasured

Pinned upstream `a7312ae94f801fc9c6786dc56e38df57b964f697` builds actual
`sm_120a` code. In `ggml-cuda/common.cuh`, `GGML_CUDA_PDL=0` selects ordinary
launch of the same compiled kernel, arguments, grid, block, shared memory and
stream. Default enables the programmatic launch attribute for eligible kernels.
This avoids changing compiled arithmetic or the CUDA12.8 inference runtime.
The [CUDA guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html)
describes the dependency synchronization required by this scheduling mechanism.

The native fixture chains RMS norm, Q6_K matrix multiplication and softmax:
widths256/768, output rows17/128/129, tokens1/3/39, three changing inputs per shape.
Each process completes54 computations across18 shapes, verifies finite output,
and writes every output byte. PDL on captures84 attributed launches/72 classic;
off captures0/156. Both have zero CUDA launch errors and successful unsubscribe.
Their output SHA is
`94c35bfb01b13b6bf457c18ad3d3683ea6d27c457cdcb1c51d01023f8267d7b6`.

The final probe is archived under
`C:/models/expertflow/runs/cuda-pdl-feasibility-20261004-qualified`.
[feasibility.json](feasibility.json) binds source/executable hashes, command,
environment, PID, successful owned exit, raw logs/output and actual loaded DLL
paths/hashes captured by Toolhelp32. Live qualification verifies loaded
ggml-base/ggml-cuda/cudart against the pinned binding and rehashes all recorded
CUDA/CUPTI libraries. Standalone fixture operations do not import ggml.dll;
the full server binding still verifies every pinned dependency.

All failed diagnostic roots remain under `C:/models/expertflow/runs/`:

| Root suffix | Retained result |
| --- | --- |
| cuda-pdl-feasibility-20261004 | computations completed; unsubscribe assertion |
| -detach-diagnostic | missing CUDA DLL search path; exit0xC0000135 |
| -detach-diagnostic-cuda-path | CUPTI invalid device, zero captured launches |
| -state-diagnostic | explicit fatal CUPTI invalid-device notification |
| -cupti129 | newer generated headers require CUDA12.9 types; build failed, old executable could not load with new-only tool path |
| -cupti129-abi | first successful on/off check with CUDA12.8 headers and CUPTI12.9 import library/DLL |
| -qualified | repeat with executable/actual loaded-library capture, both exit0 |

Official redistribution metadata and package hashes are recorded in
[cuda-audit-tools.json](cuda-audit-tools.json) and
[cupti-compatibility-tools.json](cupti-compatibility-tools.json).
The original tooling download was23,514,970bytes; the bounded compatibility
package added13,359,912bytes. Both are local unpacked tools. CUPTI12.8's fatal
notification and unsubscribe failure are consistent with an instrumentation
compatibility failure: swapping only CUPTI to12.9 resolves them. This is an
observed diagnosis, not a claim about every CUDA12.8 installation.

The first full CPU suite ended688passed/7source-environment skips/1failure: a
reference test stopped at its source-freeze guard during an edit. The failure
and logs are retained. The fresh full check with reviewed source stable passed
692tests with7source-environment skips in467.48s. Review identified and resolved
failure classification, terminal audit and native executable/DLL provenance gaps.
Separate phase-source checks use their actual private phase-profile checkout;
those patches are absent from the pristine inference build by design.

This qualifies one bounded Q6 scheduling diagnostic. It establishes neither a
TPS improvement nor universal bitwise equivalence. No accepted product plan is
replaced by this fixture. The CUDA graph allocation optimizer and arithmetic
controls remain outside this experiment.
