#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -d "$SCRIPT_DIR/../release/expertflow-build-week" ]; then
  RELEASE=$(CDPATH= cd -- "$SCRIPT_DIR/../release/expertflow-build-week" && pwd)
else
  RELEASE=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
fi
python3 "$RELEASE/scripts/verify_release.py" || { echo "Release verification failed. Restore the archive and rerun scripts/verify-release.sh." >&2; exit 2; }
