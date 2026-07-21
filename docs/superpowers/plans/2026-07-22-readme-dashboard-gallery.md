# README Dashboard Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the README the dashboard's visual identity through two real screenshots and a compact judge-links block.

**Architecture:** Capture the self-contained dashboard at a consistent desktop viewport, store the PNGs with existing assets, and replace the older inline visual sequence without changing product claims.

**Tech Stack:** Markdown, static HTML, headless Chrome, PNG, pytest.

## Global Constraints

- Keep the logo and all measured claims unchanged.
- Remove README embeds for `architecture.svg`, `result.svg`, `placement-map.svg`, `cache-decision.svg`, `profile-cards.svg`, and the workflow SVG.
- Add two descriptive dashboard screenshots and four judge-facing links.

### Task 1: Capture and contract

- [ ] Add a failing README contract for the two PNGs, judge links, and retired embeds.
- [ ] Capture the dashboard hero and architecture/placement view at 1440x900.
- [ ] Update README copy and apply the humanizer review.
- [ ] Run focused tests, rebuild the release, verify the archive, commit, and push `main`.

