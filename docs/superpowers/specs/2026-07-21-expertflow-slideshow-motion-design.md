# ExpertFlow Slideshow Narrative and Motion Design

## Goal

Turn the submission slideshow into a cinematic, offline recording surface that presents ExpertFlow as an evidence-driven placement compiler, not a hand-picked static-layer configuration.

## Narrative

1. **Constraint:** a high-quality quantized MoE model fits imperfectly on a 16 GB GPU. Stock layer-level offload leaves routed expert work crossing the CPU/GPU boundary.
2. **Investigation:** GPT-5.6 shaped the hypotheses. Codex instrumented the runtime, built exact observers and caches, ran parity and performance gates, and preserved failed results.
3. **Reversal:** reactive caching, prediction, and asynchronous prefetch were plausible but did not produce the best end-to-end result. The evidence changed the product.
4. **Compiler insight:** ExpertFlow profiles the model and hardware, inventories complete expert banks, scores CPU relief against VRAM cost and reserve, validates exactness and quality, and emits a deployment manifest before graph construction.
5. **Compiled result:** for the measured RTX 5060 Ti 16 GB target, the emitted Q6 plan places 12 complete expert banks on CUDA and measures 28.13 decode TPS, 22.48% above the strongest fair stock result.
6. **Product:** replay proves the evidence without a GPU; optimize emits a machine-specific plan; run and serve apply it. Static residency is one compiled policy, not ExpertFlow's identity.
7. **Codex:** the submission emphasizes that GPT-5.6 and Codex managed the full experimental loop, including ideas, implementation, tests, stop conditions, evidence, documentation, and release packaging.

## Slide Sequence

1. **Opening:** ExpertFlow and the measured result; restrained logo reveal.
2. **Hidden boundary:** stock cross-backend flow versus an ExpertFlow CUDA-resident route.
3. **Codex loop:** ideate, instrument, test, reject, and ship as a visible engineering circuit.
4. **Measured result:** stock bar establishes the reference before the ExpertFlow value resolves.
5. **Evidence changed the product:** predictive-cache branch visibly stops; placement-compiler branch continues.
6. **Compiler output:** the placement map is labelled as a hardware-specific emitted plan, not universally optimal layers.
7. **Runnable surface:** replay, optimize/place, and serve activate as one connected workflow.
8. **Evidence boundaries:** claims and limitations enter plainly, without celebratory effects.
9. **Closing:** the compiler framing and measured result resolve together.

## Motion System

- CSS and small vanilla JavaScript only; no network dependencies.
- An `IntersectionObserver` makes one slide active at a time and restarts its entrance sequence when revisited.
- SVGs load as same-origin objects so the page can apply controlled element-level reveals without rewriting source artwork.
- Motion vocabulary: full-canvas routed circuitry, luminous packets flowing along those traces, node activation, shallow camera movement, staggered evidence reveals, and one-shot number emphasis.
- Foreground white/cream sweeps are prohibited. Illumination must remain attached to the background circuit routes and nodes.
- Each slide uses a distinct sequence, but all motion follows the PCB/carbon/gold identity.
- No generic bouncing, continuous glitching, spinning, or template-style transitions.
- Navigation remains native scroll-snap plus keyboard control. A replay control restarts the current slide.
- `prefers-reduced-motion` disables transforms, trace drawing, and stagger delays while preserving complete content.

## Responsive and Recording Behavior

- Every visual remains fully contained at desktop and narrow viewport sizes.
- The page never introduces horizontal overflow.
- Captions, slide counts, navigation, and controls remain above the artwork and readable at 720p capture resolution.
- SVG-load failure leaves a visible textual fallback rather than a blank frame.
- Animations must not alter benchmark wording or measured claims.

## Verification

- Source-contract tests assert the narrative labels, active-slide observer, SVG-object embedding, replay control, reduced-motion path, offline-only assets, and containment rules.
- Browser inspection checks all nine slides at 1280 x 720, active-state changes, replay behavior, SVG animation injection, keyboard navigation, and zero horizontal overflow.
- The full applicable test suite and deterministic release build run before commit and push.
