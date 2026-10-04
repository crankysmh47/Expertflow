# Reuse review and resolution

Independent read-only review covered f22bb54 through 0a7a284. One blocking P1:
the Q4 provider checked pinned manifest text but runtime self-consistency alone
allowed a direct API caller to supply unrelated binaries. No other blocking
finding was identified. Native correctness and performance were not reviewed:
Q4 had no retained native samples.

The regression used a real RuntimeBinding with a self-consistent unrelated
server and the unchanged approved manifest. It failed because eligibility
incorrectly admitted the binding. Both builtin providers now require manifest
server/dependency/CUDA identity matching, unique complete dependency inventory
in the server directory, no extra DLLs, both pinned launcher binaries, and
actual file verification. Ten real-file contract cases cover valid inventory,
server, missing/duplicate/changed/moved dependencies, missing/changed CUDA,
extra DLLs and changed CLI. Targeted result: 33 passed.

Historical Q6 manifests remain bound to their original frozen checkout. This
fix does not reinterpret or replace Q6's measured recommendation. Post-fix full
suite: 670 passed, 7 source-environment skips; explicit pinned native source
subset: 6 passed. Actual stock RuntimeBinding passed the strict check with
identity d13355831e463b73536253b5fc9204e1fba3537f918c2ac9cf6e172bb33d59cc.
Actual Q4 evidence is recorded separately in the execution ledger.
