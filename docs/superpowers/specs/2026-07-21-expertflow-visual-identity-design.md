# ExpertFlow visual identity design

## Objective

Give ExpertFlow a recognizable submission identity derived from the supplied logo. Replace generic benchmark cards with visuals that communicate routed expert placement while keeping every displayed claim tied to committed evidence.

## Direction

The identity combines routed-circuit structure with carbon-and-gold presentation.

- PCB routing supplies the visual grammar: traces, chips, expert nodes, layer grids, and illuminated CUDA paths.
- Carbon-black fields provide hierarchy for titles, results, and important decisions.
- Gold is reserved for the ExpertFlow mark, selected placement boundaries, and the primary measured result.
- Green indicates active routing and CUDA-resident expert banks.
- Muted gray indicates unchanged or inactive paths.
- Restrained red identifies CPU bottlenecks or rejected paths.

The result should feel like a hardware instrument rather than a generic dashboard.

## Brand system

### Palette

| Role | Color |
|---|---|
| PCB background | `#0B3D20` |
| PCB surface | `#12572B` |
| Active routing | `#8CF59B` |
| Secondary trace | `#4FBF68` |
| Carbon | `#101311` |
| Graphite | `#242925` |
| Metallic gold | `#D6A84A` |
| Warm white | `#F4F2E9` |
| Muted measurement | `#9FB9A5` |
| CPU/rejected path | `#BF665E` |

### Typography

- Use a bold geometric system sans-serif for headings.
- Use monospace for measurements, layer IDs, hashes, commands, and runtime labels.
- Use uppercase sparingly for hardware labels such as `CUDA`, `CPU`, `Q6`, and `EXPERT BANK`.
- Avoid decorative display fonts and excessive letter spacing.

### Logo treatment

- Preserve the supplied artwork as the canonical full logo.
- Add a repository-local copy with a descriptive filename.
- Use the full PCB image as the README hero and submission/video title background.
- Create a compact vector wordmark treatment for diagrams without redrawing or misrepresenting the supplied central mark.
- Never place the gold accent on a low-contrast light background.

## Asset redesign

### Architecture

Show the stock path as a dim, interrupted trace that crosses from CUDA to CPU and back. Show ExpertFlow as a continuous illuminated route through selected CUDA-resident expert banks. Use chips and traces rather than rectangular feature cards.

### Result

Place `28.13 TPS` as the dominant gold measurement on carbon. Route a green trace from `22.967 stock TPS` to `+22.48%`, with hardware and quantization labels integrated as PCB annotations.

### Placement map

Represent model layers as chip pads on a board. Selected layers `[0-9, 15, 20]` glow green with gold boundaries; unchanged layers remain graphite. The graphic must state that complete 128-expert Q6 banks are placed, not individual predicted experts.

### Cache decision

Use two routed paths: predictive caching ends at a measured `NO CACHE OPPORTUNITY` stop node, while full static residency continues to the shipped placement. Clearly label simulation versus measured runtime evidence.

### Product profiles

Replace cards with three connected ports on one board: replay, live placement, and OpenAI-compatible serving. Each port exposes only its verified command and measurement.

### Video frames

Create consistent title, architecture, measured result, Codex workflow, limitations, and closing frames. Motion will come from editor zooms, trace reveals, and cuts; the SVGs themselves remain deterministic and easy to render.

## README structure

The README will answer these questions in order:

1. What is ExpertFlow and what measured result did it achieve?
2. Why does this problem exist?
3. Can a judge replay it immediately?
4. How does placement work?
5. How did Codex and GPT-5.6 drive the project?
6. Can it run live?
7. What are the evidence limits?
8. How can the result be reproduced?

The hero will include the logo, the one-line product description, the measured result, and the two-command replay. Installation prerequisites will not lead the page.

## Codex and GPT-5.6 account

The submission will state that GPT-5.6 was used throughout ideation and project progression. Codex with GPT-5.6-sol managed the engineering workflow end to end, including isolated worktrees, source investigation, instrumentation, tests, benchmark design, measurement collection, failed-path isolation, implementation tweaks, evidence packaging, reproducibility checks, and release polish.

The user remains the product and evidence authority: they selected the problem, approved or rejected directions, authorized scope changes, and set the scientific gates. This distinction keeps the account specific and credible while accurately reflecting Codex's unusually broad role.

## Evidence constraints

- Preserve `28.13 TPS`, `22.967 stock TPS`, and `22.48%` as the primary matched result.
- Keep the four-slot server result separate from single-stream decode TPS.
- Do not claim a filled 262,144-token context.
- State that the strict perplexity confidence gate was not met.
- State that predictive caching was simulated and rejected, not shipped.
- Keep live-platform support limited to the verified Windows/NVIDIA configuration.
- Do not imply that a visual animation is a live runtime trace.

## Files in scope

- `README.md`
- `docs/assets/*`
- `submission/demo-video-assets/*`
- `release/expertflow-build-week/dashboard.html` and its source/template if generated
- repository-local logo assets
- submission copy where visual references or Codex attribution need updating

Runtime code, benchmark evidence, measured results, and the protected llama.cpp branches are out of scope.

## Validation

- Verify every referenced asset exists and renders in GitHub-compatible Markdown.
- Render or inspect every SVG at its intended aspect ratio.
- Serve the dashboard locally and inspect it at desktop width.
- Run the evidence replay and applicable test suite after documentation changes.
- Scan README and submission copy for unsupported or conflated claims.
- Confirm the repository remains free of private absolute paths in public-facing instructions.
- Confirm the supplied logo file and all derived assets are tracked, while temporary render files are not.

## Rollback

Keep the identity work in the existing isolated `codex/final-polish` branch. Commit the design separately before asset and README implementation. The implementation commit can be reverted without changing runtime or evidence history.
