#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"
command -v uv >/dev/null 2>&1 || { echo "uv is missing. Install it from https://docs.astral.sh/uv/" >&2; exit 2; }
uv sync --frozen || { echo "Setup failed. Run: uv sync --frozen" >&2; exit 2; }
uv run expertflow demo --replay || { echo "Replay failed. Run: uv run expertflow demo --replay" >&2; exit 2; }
printf 'Offline dashboard: %s\n' "$ROOT/release/expertflow-build-week/dashboard.html"
