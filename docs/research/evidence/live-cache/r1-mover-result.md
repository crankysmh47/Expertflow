# R1 mover characterization result

Date: 2026-08-28  
Branch: `ef-v2`  
Commit measured: `93ecc24`  
Verdict: **NO-GO for R2 runtime integration**

## Measurement contract

The committed `expertflow mover-benchmark` command was run three independent
times with pinned host memory and three times with write-combined pinned host
memory. The measured device was an NVIDIA GeForce RTX 5060 Ti. The CUDA
Runtime API library was:

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll`

Each trial used slot sizes 3,346,048 and 26,768,384 bytes; slot counts 1, 2,
4, 8, and 32; queue depths 1, 2, 4, 8, and 16; 30 measured batches; 10 warmup
copies; 200 ready-latency samples; and a loaded leg of eight 67,108,864-byte
background copies. Trial reports are retained under `runs/ef-v2/` and are
ignored by Git.

## Pooled gate results

Raw samples from all three trials per staging mode were pooled. Ready-latency
rows therefore contain 600 samples. Eight-slot batching rows contain 90 CUDA
event samples per side. Speedup is pooled individual-copy mean divided by
pooled contiguous-batch mean.

| Staging | Slot bytes | Idle p50 | Idle p95 | Loaded p50 | Loaded p95 | 8-slot batched mean | 8-slot individual mean | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pinned | 3,346,048 | 0.2579 ms | 0.2776 ms | 37.5727 ms | 38.1562 ms | 1.8596 ms | 1.8794 ms | 1.0106x |
| pinned | 26,768,384 | 1.8976 ms | 1.9472 ms | 39.2480 ms | 39.8663 ms | 14.9572 ms | 14.9003 ms | 0.9962x |
| pinned_wc | 3,346,048 | 0.2576 ms | 0.2768 ms | 38.0995 ms | 38.7928 ms | 1.8811 ms | 1.8864 ms | 1.0028x |
| pinned_wc | 26,768,384 | 1.8953 ms | 1.9964 ms | 39.7396 ms | 40.4567 ms | 15.0410 ms | 15.0567 ms | 1.0010x |

## Gate decision

Both predeclared gates fail:

1. The best 3,346,048-byte loaded p95 is 38.1562 ms, not below 0.75 ms. It is
   about 50.9 times the threshold.
2. The best pooled eight-slot speedup is 1.0106x, not greater than 2x.

Write-combined pinned memory did not materially improve either gate. These
measurements reproduce the earlier P2 symptom: the copy itself is fast when
idle, but queue-to-ready latency under copy-engine load is approximately one
decode-forward interval. R2 simulation may remain useful as an analytical
upper bound, but the declared protocol does not authorize concurrent-prefetch
runtime integration or another `llama.cpp` change.

## Reporting limitation found

The JSON reports pass `1` rather than `slot_bytes` into the ready-latency
summary, so `ready_latency.host_ready_latency.mean_gib_per_second` is
dimensionally invalid. The raw host-ready samples and their millisecond
summaries are unaffected and are the authoritative inputs above. Do not cite
the ready-latency throughput field until that reporting defect is corrected.

## Trial integrity

```text
7A101C465B784752084F4AABFB61715871FF66673B7B469146BE1ED6F0F16C5E  mover-pinned-r1-trial1.json
095A8095DA031764B7C7AF3DFC4A8D667EA08F84636710689B7A041B6A4034D8  mover-pinned-r1-trial2.json
7161679312C078C75F500425C5A658C4F33C8D0943303CAA848002C825AB04E1  mover-pinned-r1-trial3.json
5C140B38902F46280114856FE2DC44C0C912D07C4106C41F3948BA7D0C51CAA6  mover-pinned-wc-r1-trial1.json
A5D3410D842B1286E5E5565991045BEF5FB699497DE773E51D002FC0A0F11BC1  mover-pinned-wc-r1-trial2.json
20E88DB7A960FF72751A7FEEC9767F51D82A21D8246686944B75C1237CFD6AFA  mover-pinned-wc-r1-trial3.json
```
