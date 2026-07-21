# ExpertFlow final video, dashboard, and deployment design

## Outcome

Ship a judge-facing package with three independent proof surfaces: a human, under-three-minute video centered on a real matched TPS run; a visually striking static dashboard that explains the placement compiler; and a public, deployment-ready dashboard with direct GitHub and local reproduction paths.

## Video

The story begins with the creator's personal desire to get more from a 16 GB GPU and the frustration of being forced toward smaller quants. The live `scripts/live-tps-demo.ps1 -Mode Demo` run is the central concrete proof. The authoritative ten-pair result remains visually and verbally distinct from the one-pair rehearsal. The voice stays conversational and uses short sentences rather than product-copy phrasing.

## Dashboard

The page is a self-contained narrative hardware console, not a grid of equal-weight cards. It moves through: personal constraint, hidden CPU boundary, dominant measured outcome, two runnable proof actions, animated 30-layer placement map, evidence-driven pivot, Codex/GPT-5.6 engineering loop, and three reproduction paths. Black carbon, green PCB, gold routing, visible copper traces, vias, grain, and restrained signal animation provide the identity. Evidence qualifications remain accessible in a compact ledger rather than controlling the opening view.

## Deployment and reproduction

The repository root serves the static dashboard through a minimal Vercel configuration, with clean URLs routing `/` and `/dashboard` to the packaged dashboard. Public GitHub, replay, live benchmark, build, supported-platform, model-hash, and local-hardware instructions remain visible. No model, binary, external font, analytics, or remote runtime dependency is introduced.

## Acceptance

- Video script is under three minutes and includes the real matched TPS capture.
- Dashboard shows `28.13`, `22.967`, `22.48%`, selected layers, Codex/GPT-5.6, replay/live/rebuild paths, GitHub, and local setup.
- Dashboard remains self-contained, responsive, keyboard-readable, reduced-motion aware, and honest about evidence classes.
- Vercel configuration validates locally and production deployment is attempted only with existing authentication.
- Tests, release verification, browser inspection, and repository cleanliness pass before push.
