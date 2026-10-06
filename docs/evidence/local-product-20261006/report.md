# Local companion: operational evidence, 2026-10-06

Windows 11, RTX 5060 Ti 16 GB, driver 616.92, 32 GB RAM; pristine llama.cpp
a7312ae9/CUDA 12.8. This is one machine. Independent users and another GPU
capacity have not been qualified. No result below reopens closed research.

| Registered check | Actual collection | Result and limit |
| --- | --- | --- |
| Granite Q6 setup | One process, context 4096, 8 generated tokens | PASS-OPERATIONAL-LOAD, 6.047s; load smoke only |
| Granite baseline diagnostic | One process, three 256-token samples of one raw prompt | MEASURED, mean 451.955 tokens/s, mean TTFT 0.026s; descriptive, no speedup |
| Granite local API/UI | One process, one short API chat, UI GET, status and identity-checked stop | Operational pass and cleanup; no throughput qualification |
| Granite fixed thread search | Baseline plus 8/12/16 threads, three samples each, 30s spacing, 600s ceiling | NO-MEASURABLE-GAIN, four processes/124.89s; no confirmation; baseline retained |
| Dense Qwen GPU-layer profile | One process, context 16384, real 8192 processed tokens, two chat turns, disconnect/recovery, six generation requests | PASS-OPERATIONAL-SOAK, cleanup, 3.766s; operational only |
| Dense Qwen RAM profile | One process, GPU layers 0, context 8192, real 4096 processed tokens, same six-request sequence | PASS-OPERATIONAL-SOAK, cleanup, 4.250s; operational only |

Dense artifact: Qwen2.5-0.5B-Instruct Q4_K_M, publisher revision
`9217f5db79a29953eb74d5343926648285ec7e67`, SHA256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
See [publisher pin](dense-artifact-pin.json). Dense launch support carries no
MoE routing or exact tuning policy.

Registrations, source snapshots, responses, logs and cleanup receipts are
retained in the local job directories recorded by the compact receipt index.
Each study used a fresh directory and fixed process budget. Memory sampling
starts after healthy load; it does not establish a load-time peak. Timing
numbers retain their recorded workload and host.

Final release-artifact validation, whole-branch review, Gemma local-workflow
coverage, human pilot timings and return use remain open. No new accepted tuning
improvement was demonstrated. Human feedback must establish setup/diagnosis
utility; an agent demonstration cannot count as a pilot participant.
