# Bounded stock utility proof

The placement feasibility audit found no new exact mechanism in the pinned
operations. Test the conditional stock-autotuning product on pristine Gemma Q6.
This is a fresh utility question; prior thread-search rejections remain closed.

## Baselines and scope

Use the pinned Q6 model, host/runtime, context4096/predict512/seed42, concurrency1,
F16 KV, batch2048/ub512, ngl99 and CPU-MoE. The comparison's deployment defaults
are **resolved thread and graph defaults**, not an unconstrained all-flags
out-of-box invocation. The Windows pinned source resolves physical-core threads
(8 on the full-affinity single-socket 8/16 host) and default graphs on. Explicit
equivalent flags keep the owned launcher fixed. Hash/source-audit the native
default resolution before collection. All unrelated controls remain fixed.

The automatic tuner generates topology anchors8/12/16 crossed with graphs on/off
and ranks three complete blocks by geometric ratios to the default. The scripted
manual-grid baseline independently collects the same six configurations/three
blocks and selects highest arithmetic mean TPS; ties retain defaults. It is a
documented manual grid, not a claim that every expert human would tune this way.

Tuning cost's gate unit is the independently measured native candidate count:
18 automatic and18 manual evaluations. Native load/tokenize/completion/teardown
phase time is separately reconstructed from immutable artifacts; lower wall
time and human-effort savings are not asserted. No prior TPS values enter either
ranking or confirmation.

## Frozen sequence and budget per workload

1. Own-reference10: defaults, exact native prompt/generated tokens, CV<=10%.
2. Automatic screen18 and separate manual screen18, three seeded complete blocks.
3. Freeze their selections; ten balanced default/automatic pairs20.
4. Ten balanced manual/automatic pairs20, independent of all discovery runs.
5. Only if utility passes: fresh paired selected-plan product acceptance20,
   then one standalone accepted-plan consumer process.

Maximum **107 native processes per workload**: 10+18+18+20+20+20+1. Utility-only
consumption is86. No trials, retries, discarded ordinary samples or alternative
finalists. Each stage stops on identity, token, memory, ownership or cleanup
failure. Source/protocol/host cannot change during collection. All launches bind
the frozen manifest in an EvidenceStore-hashed artifact. Product acceptance uses
the existing separately frozen paired-product gate; a speed result cannot waive it.

Utility requires >=5% paired geometric TPS improvement over resolved defaults,
positive bootstrap95% lower bound, both comparisons' arm CVs<=10%, manual-result
bootstrap90% interval strictly within[-2%,+2%], and native evaluation count no
greater than manual. Bootstrap/schedule use existing seed20261003 and10,000
whole-pair draws. These are new utility gates, not changed historical gates.

The first workload is the existing frozen prompt. If it passes utility/product/
consumer gates, repeat the same algorithm once on a separately frozen held-out
code prompt at the same shapes and a fresh107-process budget. No source, search
space, ranking or acceptance changes between workloads. Total maximum214;
failure of the first gate prevents later stages and limits the claim accordingly.

Freeze both inputs before the first launch: main
`configs/compiler/gemma4-q6-single-request.json` with `configs/baseline-prompt.txt`,
and transfer `configs/compiler/gemma4-q6-utility-transfer.json` with
`configs/compiler/stock-utility-heldout-prompt.txt`. Transfer tests workload coverage
on the same model/host, not a new model or independent human task-quality result.

A failed utility gate narrows the product to validated configuration selection/
reproduction. Passing utility supports the declared controls/workloads/host only,
not faster-than-tuned-stock custom placement, universal optimality or new quality
preservation for offload/KV changes.
