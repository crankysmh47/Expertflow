"""Streaming JSONL input for canonical routing events."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from expertflow.trace.schema import RouterTraceEvent, parse_router_event


def load_router_events(path: Path) -> Iterator[RouterTraceEvent]:
    """Yield validated records from ``path`` without loading it into memory."""

    with path.open("r", encoding="utf-8") as stream:
        for record_number, line in enumerate(stream, start=1):
            yield parse_router_event(line, record_number=record_number)
