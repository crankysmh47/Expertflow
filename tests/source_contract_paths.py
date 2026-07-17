import os
from pathlib import Path

import pytest


EXPERTFLOW = Path(__file__).resolve().parents[1]
_llama_source = os.environ.get("EXPERTFLOW_LLAMA_SOURCE")
if not _llama_source:
    pytest.skip(
        "set EXPERTFLOW_LLAMA_SOURCE to the patched llama.cpp checkout for source-contract tests",
        allow_module_level=True,
    )
LLAMA = Path(_llama_source).resolve()
if not (LLAMA / "ggml/src/ggml-backend.cpp").is_file():
    pytest.fail(f"EXPERTFLOW_LLAMA_SOURCE is not a patched llama.cpp checkout: {LLAMA}")
