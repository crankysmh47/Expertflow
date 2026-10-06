# Phase-aware diagnostic profiling protocol

Implement the approved bounded extension of the existing split profiler on ef-v2.
Preserve all old model/source/binary pins and verdicts. New native source lives in
C:/models/expertflow/worktrees/llama-phase-profile-20261004, based on 451224ab.
New binaries live in a separate build directory. This is diagnostic infrastructure,
not a product speedup or a change to the accepted twelve-thread workload.

Explicit phases are initialization/other, warmup, prefill, decode and mixed.
Common model warmup marks its evaluation calls. The server marks prefill/decode
from slot state before llama_decode, never from token count or expert tensor shape.
Mixed batches are labeled mixed and rejected for the frozen concurrency-one probe.
The llama context passes actual microbatch token counts to the scheduler once per
graph evaluation. Count processed tokens separately from split calls: this request
must show 39 prefill tokens and 511 decode-forward tokens for 512 generated tokens
(the first generated token is sampled from prompt logits).

Keep counters separate by phase and split ID, bounded at five phases times 256
split slots. Preserve per-split timing semantics: CPU graph-compute call wall,
CUDA host submission, completion wait, and input boundary including routing,
copies and waits. Pure transfer duration remains unavailable in the generic API;
do not invent a copy-only metric or infer bandwidth. Synchronized mode is labeled
diagnostic and perturbs overlap. Default-off execution collects no counters and
adds no synchronization. Use a separate schema2 profile; old schema1 stays intact.

Re-use the frozen CPU and CUDA backend DLLs/import libraries in the diagnostic
build where ABI-compatible, and verify their hashes. Rebuild the scheduler,
llama and server pieces. Record the new source revision, patch and every binary
hash. Any missing dependency, ABI/load failure, token mismatch, memory/reserve or
cleanup failure stops native work; never replace a frozen build in place.

Verify analysis with malformed phase/accounting fixtures, build the native
instrumentation, run applicable source contracts and the CPU suite, then make
one final code review and one fix pass. Fixed native budget: fresh stock control,
diagnostic build with profiling off, then three synchronized phase profiles.
No retries/discards. All rows diagnostic-only/measured=False. Preserve raw tokens,
owned memory/process/cleanup evidence, explicit phase counts and profile hashes.
Use the existing tested private-console shutdown for profiler flushing.

Report the decode-only synchronized split breakdown and its limitations. Select
at most one concrete subsequent kernel/transfer hypothesis if supported; write
its numerical contract and acceptance experiment before any performance test.
Do not automatically publish a compiler plan, revise the original validation
gate, rerun eight threads, or reopen reactive caching.
