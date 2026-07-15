"""Frozen expanded-corpus construction and strict leakage checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from expertflow.trace.io import load_router_events


DOMAINS = (
    "general_instruction",
    "code",
    "math_reasoning",
    "translation_multilingual",
    "structured_output",
    "topic_shift",
)
SPLIT_COUNTS = {"train": 60, "validation": 12, "test": 12}
PER_DOMAIN_SPLITS = {"train": 10, "validation": 2, "test": 2}


class ExpandedManifestError(ValueError):
    """Raised when a frozen expanded manifest violates its contract."""


@dataclass(frozen=True)
class ExpandedConversation:
    conversation_id: str
    split: str
    domain: str
    template_id: str
    task_family: str
    prompt: str
    prompt_sha256: str
    normalized_prompt_sha256: str


@dataclass(frozen=True)
class ExpandedManifest:
    dataset_id: str
    split_counts: dict[str, int]
    domains: tuple[str, ...]
    conversations: tuple[ExpandedConversation, ...]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalize_prompt(prompt: str) -> str:
    value = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def normalized_prompt_sha256(prompt: str) -> str:
    return prompt_sha256(normalize_prompt(prompt))


def _require_string(row: dict[str, object], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExpandedManifestError(f"conversations[{index}].{key} must be non-empty")
    return value


def load_expanded_manifest(path: Path) -> ExpandedManifest:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpandedManifestError(f"cannot read expanded manifest: {error}") from error
    if not isinstance(root, dict) or root.get("schema_version") != "1.0.0":
        raise ExpandedManifestError("unsupported expanded manifest")
    if root.get("split_counts") != SPLIT_COUNTS:
        raise ExpandedManifestError(f"split_counts must equal {SPLIT_COUNTS}")
    if root.get("domains") != list(DOMAINS):
        raise ExpandedManifestError("domains must equal the frozen six-domain order")
    raw_rows = root.get("conversations")
    if not isinstance(raw_rows, list):
        raise ExpandedManifestError("conversations must be an array")

    rows: list[ExpandedConversation] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ExpandedManifestError(f"conversations[{index}] must be an object")
        prompt = _require_string(raw, "prompt", index)
        exact = _require_string(raw, "prompt_sha256", index)
        normalized = _require_string(raw, "normalized_prompt_sha256", index)
        if exact != prompt_sha256(prompt):
            raise ExpandedManifestError(f"conversations[{index}].prompt_sha256 is stale")
        if normalized != normalized_prompt_sha256(prompt):
            raise ExpandedManifestError(
                f"conversations[{index}].normalized_prompt_sha256 is stale"
            )
        rows.append(
            ExpandedConversation(
                conversation_id=_require_string(raw, "conversation_id", index),
                split=_require_string(raw, "split", index),
                domain=_require_string(raw, "domain", index),
                template_id=_require_string(raw, "template_id", index),
                task_family=_require_string(raw, "task_family", index),
                prompt=prompt,
                prompt_sha256=exact,
                normalized_prompt_sha256=normalized,
            )
        )

    if len(rows) != 84:
        raise ExpandedManifestError("manifest must contain exactly 84 conversations")
    for field in ("conversation_id", "template_id", "prompt_sha256", "normalized_prompt_sha256"):
        values = [getattr(row, field) for row in rows]
        if len(set(values)) != len(values):
            raise ExpandedManifestError(f"duplicate {field}")
    if dict(Counter(row.split for row in rows)) != SPLIT_COUNTS:
        raise ExpandedManifestError("observed split counts do not match frozen counts")
    for domain in DOMAINS:
        observed = Counter(row.split for row in rows if row.domain == domain)
        if dict(observed) != PER_DOMAIN_SPLITS:
            raise ExpandedManifestError(f"domain {domain} must have exact 10/2/2 split")
    if {row.domain for row in rows} != set(DOMAINS):
        raise ExpandedManifestError("unexpected domain")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[row.task_family].add(row.split)
    crossing = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    if crossing:
        raise ExpandedManifestError(f"task_family crosses splits: {crossing}")

    return ExpandedManifest(
        dataset_id=_require_string(root, "dataset_id", -1),
        split_counts=dict(SPLIT_COUNTS),
        domains=DOMAINS,
        conversations=tuple(rows),
    )


def build_frozen_manifest(prompt_bank: dict[str, list[tuple[str, str]]]) -> dict[str, object]:
    """Build the deterministic 10/2/2-per-domain manifest before trace collection."""
    rows: list[dict[str, str]] = []
    split_by_index = ["train"] * 10 + ["validation"] * 2 + ["test"] * 2
    for domain in DOMAINS:
        prompts = prompt_bank.get(domain)
        if prompts is None or len(prompts) != 14:
            raise ExpandedManifestError(f"prompt bank for {domain} must contain 14 rows")
        for ordinal, ((family, prompt), split) in enumerate(zip(prompts, split_by_index), 1):
            prefix = {"train": "tr", "validation": "va", "test": "te"}[split]
            conversation_id = f"exp84-{prefix}-{domain}-{ordinal:02d}"
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "split": split,
                    "domain": domain,
                    "template_id": f"tpl-{domain}-{ordinal:02d}-v1",
                    "task_family": f"{split}:{family}",
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256(prompt),
                    "normalized_prompt_sha256": normalized_prompt_sha256(prompt),
                }
            )
    return {
        "schema_version": "1.0.0",
        "dataset_id": "trace-v2-canonical-expanded-84-v1",
        "status": "frozen_before_collection",
        "source_policy": "Public synthetic prompts authored for ExpertFlow; no private conversation data.",
        "split_policy": "Conversation IDs and 60/12/12 membership frozen before trace generation; no post-result movement.",
        "deduplication_policy": {
            "exact": "unique UTF-8 SHA-256",
            "normalized": "NFKC, casefold, Unicode word tokens, collapsed whitespace, unique SHA-256",
            "near_duplicate": "task-family isolation plus token similarity review before collection",
        },
        "split_counts": SPLIT_COUNTS,
        "per_domain_split_counts": PER_DOMAIN_SPLITS,
        "domains": list(DOMAINS),
        "conversations": rows,
    }


def validate_canonical_shard(path: Path, conversation_id: str) -> dict[str, int]:
    """Strictly accept only complete, causally ordered 30-layer forwards."""
    try:
        events = list(load_router_events(path))
    except (OSError, ValueError) as error:
        raise ExpandedManifestError(f"invalid canonical trace: {error}") from error
    if not events:
        raise ExpandedManifestError("canonical trace is empty")
    if any(event.conversation_id != conversation_id for event in events):
        raise ExpandedManifestError("trace conversation_id mismatch")
    if any(len(event.selected_expert_ids) != 8 for event in events):
        raise ExpandedManifestError("every event must select exactly eight experts")
    if any(len(set(event.selected_expert_ids)) != 8 for event in events):
        raise ExpandedManifestError("selected expert IDs must be unique")
    if any(expert < 0 or expert >= 128 for event in events for expert in event.selected_expert_ids):
        raise ExpandedManifestError("selected expert ID is outside [0, 128)")

    grouped: dict[tuple[int, int, int, str], list[object]] = defaultdict(list)
    previous_hook = -1
    previous_timestamp = -1
    for event in events:
        if event.hook_order <= previous_hook:
            raise ExpandedManifestError("hook_order is not strictly increasing")
        if event.observed_at_ns < previous_timestamp:
            raise ExpandedManifestError("observer timestamps are not causal")
        previous_hook = event.hook_order
        previous_timestamp = event.observed_at_ns
        grouped[(event.forward_id, event.token_index, event.token_id, event.phase)].append(event)
    for key, forward_events in grouped.items():
        layers = [event.layer_id for event in forward_events]
        if layers != list(range(30)):
            raise ExpandedManifestError(f"forward {key} does not contain ordered layers 0..29")
    return {
        "event_count": len(events),
        "forward_count": len(grouped),
        "layer_count": 30,
    }


def select_collection_rows(
    manifest: ExpandedManifest,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    unseal_test: bool = False,
) -> tuple[ExpandedConversation, ...]:
    """Select checkpoint order while keeping the frozen test split sealed."""
    if "test" in splits and not unseal_test:
        raise ExpandedManifestError("expanded test split is sealed")
    unknown = set(splits) - set(SPLIT_COUNTS)
    if unknown:
        raise ExpandedManifestError(f"unknown collection splits: {sorted(unknown)}")
    selected: list[ExpandedConversation] = []
    for split in splits:
        by_domain = {
            domain: [
                row
                for row in manifest.conversations
                if row.split == split and row.domain == domain
            ]
            for domain in DOMAINS
        }
        width = max((len(rows) for rows in by_domain.values()), default=0)
        for ordinal in range(width):
            for domain in DOMAINS:
                rows = by_domain[domain]
                if ordinal < len(rows):
                    selected.append(rows[ordinal])
    return tuple(selected)
