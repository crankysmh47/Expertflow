"""Platform-local state paths; inspecting a path never creates it."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from typing import Mapping


def data_root(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("EXPERTFLOW_HOME"):
        return Path(env["EXPERTFLOW_HOME"]).expanduser().resolve()
    home = Path.home()
    if sys.platform == "win32":
        return Path(env.get("LOCALAPPDATA", home / "AppData/Local")) / "ExpertFlow"
    if sys.platform == "darwin":
        return home / "Library/Application Support/ExpertFlow"
    return Path(env.get("XDG_DATA_HOME", home / ".local/share")) / "expertflow"
