# Execute the single CPU expert prefetch experiment

Authority: [frozen numerical and measurement protocol](../specs/2026-10-04-cpu-expert-prefetch-experiment.md).
User authorized continuation until a real blocker. Keep ExpertFlow changes on
ef-v2, native work in a separate private worktree, original builds unchanged.

1. Verify pristine source a7312ae and actual stock CPU compile commands. Build
   only the candidate CPU backend using the identical compiler, definitions,
   flags and source set. Stock uses /arch:AVX2, GGML_NATIVE=OFF. The diagnostic
   fork's AVX512 setting is irrelevant to this stock comparison.
2. Before implementation, build/run deterministic expert parity fixtures against
   stock and demonstrate the compiled caller lacks the proposed prefetch hint.
   Add exactly the guarded next-row Q6_K hint. Check its compiled presence,
   unchanged dot-product source and normalized disassembly, unchanged ABI, and
   bitwise stock/candidate fixture outputs. Any numerical discrepancy stops
   before native timing.
3. Create a pinned kernel-only runtime from copies of pristine stock binaries,
   replacing only ggml-cpu.dll. Record all dependency hashes, source patch,
   compile commands, fixture and machine-code evidence. Keep all profiling off.
4. Freeze a fresh EvidenceStore and ten balanced pairs before launch. Use the
   existing owned-process runner and bootstrap implementation, exact reference
   tokens, memory/reserve and cleanup checks. Retain all twenty processes without
   retries. A launch/correctness/identity failure is terminal, not a resampling
   opportunity. No source/protocol edits after freezing.
5. Independently reverify each stored artifact and pairing; record PASS,
   VALIDATION-STOP or INCONCLUSIVE using the frozen gates. Perform one fresh
   final review and one fix pass if needed; no additional native runs. Commit
   evidence on ef-v2, preserve the original compiler verdict and user files.

Completion: numerical gate and fixed experiment have a durable terminal result;
no speedup claim without passing all gates; no product publication, merge/push,
alternate prefetch distances or new optimization hypothesis in this experiment.
