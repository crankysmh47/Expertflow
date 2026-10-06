# Compatibility, updates and rollback

The alpha uses profile schema version 1, separate from historical compiler
plans. Unknown versions fail with a regeneration instruction. There is no
silent migration of accepted evidence.

Profiles retain absolute locations and content hashes. After moving a model or
installing another runtime, run `local setup` again. Keep the original profile
and binaries until the replacement loads. Setup writes atomically and keeps a
backup when replacing a profile. Launch never upgrades binaries.

Install a newer wheel in a separate environment first; repeat doctor, setup,
chat and client serving. Roll back using the previous wheel/environment and its
original model/runtime/profile. Alpha CLI incompatibilities belong in the
changelog.

For an upstream runtime update, choose its exact revision and trusted archive
digest; check licenses/dependencies and install beside the current version.
Run Windows/Linux model-free and installed-wheel checks. Discover actual flags;
unknown versions remain untuned. Freeze native model/runtime/host/workload,
candidate list, context, wall/process budgets and cleanup gates before launch.
Exercise load, processed long prompts, chat, serving, disconnect, stop/restart
and relevant RAM/GPU placement. Preserve failed attempts.

A new exact scheduling policy needs source/numerical qualification and a fresh
held-out comparison against that runtime's real defaults. Agreement on a few
prompts alone is not a universal exactness proof. Publish actual support cells
and negative results. Never apply old evidence to a changed binary, model,
driver, workload or context. Python checks do not qualify CUDA, another GPU
capacity or Linux inference; human utility remains a separate gate.
