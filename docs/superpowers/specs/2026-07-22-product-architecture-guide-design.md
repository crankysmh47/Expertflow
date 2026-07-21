# Product Architecture Guide Design

## Goal

Add one canonical, judge-friendly document that explains ExpertFlow as a product and as a system without requiring readers to reconstruct it from experiment logs.

## Audience and hierarchy

The guide serves judges, model-runtime engineers, and technically curious users. It sits between the concise README and the detailed evidence tree: the README sells the result, `docs/PRODUCT.md` explains the architecture, and evidence documents substantiate individual claims.

## Content

`docs/PRODUCT.md` will cover the problem, the stock execution boundary, the profiling and plan-compilation pipeline, runtime components, complete expert-bank integrity, the shipped twelve-layer Q6 plan, CLI and serving flow, measured results, the rejected cache path, reproduction routes, limitations, and the roles of GPT-5.6 and Codex. It will reuse the existing architecture, placement, cache-decision, profile, and workflow visuals.

The guide must clearly distinguish the general compiler architecture from the hardware-specific measured output. It must not imply a universal optimal layer set, dynamic caching, prediction, per-token transfers, cross-platform live verification, or stronger quality claims than the release scorecard supports.

## Integration

README and dashboard will link to the guide. The release builder will package it. Focused tests will verify those links, the architecture sections, diagram references, exact measured claims, and release inclusion.

