# Three-minute Observatory demo

## 0:00-0:20 — The expensive systems mistake

Show the title of `observatory.html`.

Say:

> Sparse MoE models activate only a few experts at a time, but that does not
> automatically mean expert offloading will work well. Before changing the
> runtime, we need to know the real working set, transfer cost, and VRAM budget.

## 0:20-0:45 — Inspect the machine and model

Show the measured hardware and model section.

Say:

> ExpertFlow profiled Gemma 4 26B A4B Q4_0 on an RTX 5060 Ti. It measured all
> 3,840 layer-expert objects instead of estimating from the GGUF file. One
> aligned packed expert slot is 3,346,048 bytes.

## 0:45-1:20 — Show the locality result

Show the per-domain and held-out locality views.

Say:

> Routing is structured, but it is not uniform. The frozen Static-96 policy,
> selected from training conversations only, reaches 87.57% on untouched
> validation and test conversations. Conversation-reset LRU reaches 86.34%.
> Topic shifts and domains still matter, which is why the report keeps the
> prompt-level breakdown.

## 1:20-1:55 — Replay the policies

Use the causal event timeline and the small judge fixture.

Say:

> The replay explains each demand instead of showing only a hit-rate bar. On
> the bundled eight-event fixture there are 64 expert demands: static gets 26
> hits and LRU gets 19. These are estimates over previously measured routing
> events, and the UI labels them that way.

## 1:55-2:25 — Show the machine recommendation

Show the recommendation and memory envelope.

Say:

> A 96-slot plan across 21 target layers projects to 6,433.14 MiB and leaves
> 800.86 MiB of configurable reserve in the frozen profile. The standalone
> pinned CUDA benchmark measured one aligned expert copy at 0.234016
> milliseconds p50 and 0.234272 p95. ExpertFlow keeps those transfer
> measurements separate from Vulkan timing and simulator estimates.

## 2:25-2:45 — Give the honest runtime decision

Show the final scorecard.

Say:

> We also built the exact live path. The final predictor prevented real blocking
> misses, but decode TPS was 1.15% lower overall. That means the release keeps
> live caching off. We are not turning a negative benchmark into a speedup
> claim.

## 2:45-3:00 — Close on product value

Return to the report overview.

Say:

> ExpertFlow turns a risky runtime rewrite into a reproducible decision. A judge
> can inspect the mechanism without our GPU, while the full evidence shows
> exactly what worked, what did not, and why.
