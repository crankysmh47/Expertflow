import json
from pathlib import Path

import pytest

from expertflow.collection import (
    CollectionConfig,
    EXPECTED_DOMAIN_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    NativeProcessResult,
    collect_trace_pairs,
    load_corpus_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "configs" / "q4-physical-feasibility-corpus.json"


def test_checked_in_corpus_freezes_maximum_spec_probe() -> None:
    corpus = load_corpus_manifest(CORPUS_PATH)

    assert corpus.dataset_id == "expertflow-q4-physical-feasibility-v1"
    assert len(corpus.conversations) == 40
    assert corpus.split_counts == EXPECTED_SPLIT_COUNTS
    assert corpus.domain_counts == EXPECTED_DOMAIN_COUNTS
    assert len({item.conversation_id for item in corpus.conversations}) == 40
    assert {
        item.domain
        for item in corpus.conversations
        if item.split in {"validation", "test"}
    } == set(EXPECTED_DOMAIN_COUNTS)
    assert all(item.prompt.strip() for item in corpus.conversations)


def test_rejects_duplicate_conversation_ids(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["conversations"][1]["conversation_id"] = payload[
        "conversations"
    ][0]["conversation_id"]
    invalid = tmp_path / "duplicate.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conversation_id values must be unique"):
        load_corpus_manifest(invalid)


def test_rejects_declared_counts_that_do_not_match_rows(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["conversations"][0]["split"] = "test"
    invalid = tmp_path / "bad-counts.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="observed split counts"):
        load_corpus_manifest(invalid)


def test_collects_pair_with_provenance_and_resumes_valid_shard(
    tmp_path: Path,
) -> None:
    corpus_payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_payload["split_counts"] = {"train": 1}
    corpus_payload["domain_counts"] = {"general_chat": 1}
    corpus_payload["conversations"] = [corpus_payload["conversations"][0]]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(corpus_payload), encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    probe = tmp_path / "runtime" / "probe.exe"
    probe.parent.mkdir()
    probe.write_bytes(b"probe")
    (probe.parent / "backend.dll").write_bytes(b"runtime")
    calls: list[list[str]] = []

    def fake_runner(argv, *, cwd, stdout_path, stderr_path):
        calls.append(argv)
        assert cwd == probe.parent.resolve()
        stdout_path.write_text("stdout", encoding="utf-8")
        stderr_path.write_text("stderr", encoding="utf-8")
        tokens_path = Path(argv[argv.index("--tokens") + 1])
        tokens_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "prompt_token_ids": [1, 2],
                    "generated_token_ids": [3, 4],
                }
            ),
            encoding="utf-8",
        )
        if "--trace" in argv:
            trace_path = Path(argv[argv.index("--trace") + 1])
            trace_path.write_text(
                '{"schema_version":"1.0.0","request_id":"req-001",'
                '"conversation_id":"conv-001","turn_index":0,'
                '"phase":"prefill","forward_id":0,"hook_order":0,'
                '"token_index":0,"token_id":1,"layer_id":0,'
                '"selected_expert_ids":[1,2,3,4,5,6,7,8],'
                '"selected_expert_weights":null,"observed_at_ns":1}\n',
                encoding="utf-8",
            )
        return NativeProcessResult(
            return_code=0,
            started_at="2026-07-15T00:00:00+00:00",
            ended_at="2026-07-15T00:00:01+00:00",
            duration_seconds=1.0,
        )

    config = CollectionConfig(
        probe=probe,
        model=model,
        model_sha256=(
            "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fa"
            "d799250a0f70d652b6b825e4"
        ),
        output_dir=tmp_path / "runs",
        n_predict=64,
        gpu_layers=10,
        threads=12,
    )
    first = collect_trace_pairs(
        corpus,
        config,
        process_runner=fake_runner,
        expected_split_counts={"train": 1},
        expected_domain_counts={"general_chat": 1},
    )

    assert first["summary"] == {
        "conversation_count": 1,
        "passed": 1,
        "failed": 0,
        "skipped_valid": 0,
    }
    shard = first["shards"][0]
    assert shard["conversation_id"] == "train-general-01"
    assert shard["latest_status"] == "passed"
    assert shard["attempts"][0]["parity"]["generated_matches"] is True
    assert shard["attempts"][0]["trace"]["event_count"] == 1
    assert len(calls) == 2

    second = collect_trace_pairs(
        corpus,
        config,
        process_runner=fake_runner,
        expected_split_counts={"train": 1},
        expected_domain_counts={"general_chat": 1},
    )

    assert second["summary"]["skipped_valid"] == 1
    assert len(calls) == 2
    persisted = json.loads(
        (config.output_dir / "collection-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["shards"][0]["latest_status"] == "passed"


def test_retry_preserves_failed_attempt_artifacts(tmp_path: Path) -> None:
    corpus_payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_payload["split_counts"] = {"train": 1}
    corpus_payload["domain_counts"] = {"general_chat": 1}
    corpus_payload["conversations"] = [corpus_payload["conversations"][0]]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(corpus_payload), encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    probe = tmp_path / "runtime" / "probe.exe"
    probe.parent.mkdir()
    probe.write_bytes(b"probe")
    calls: list[list[str]] = []

    def fake_runner(argv, *, cwd, stdout_path, stderr_path):
        call_index = len(calls)
        calls.append(argv)
        stdout_path.write_text("stdout", encoding="utf-8")
        stderr_path.write_text("stderr", encoding="utf-8")
        is_instrumented = "--trace" in argv
        retry_index = call_index // 2
        generated = [4] if retry_index == 0 and is_instrumented else [3]
        tokens_path = Path(argv[argv.index("--tokens") + 1])
        tokens_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "prompt_token_ids": [1, 2],
                    "generated_token_ids": generated,
                }
            ),
            encoding="utf-8",
        )
        if is_instrumented:
            trace_path = Path(argv[argv.index("--trace") + 1])
            trace_path.write_text(
                '{"schema_version":"1.0.0","request_id":"req-001",'
                '"conversation_id":"conv-001","turn_index":0,'
                '"phase":"prefill","forward_id":0,"hook_order":0,'
                '"token_index":0,"token_id":1,"layer_id":0,'
                '"selected_expert_ids":[1,2,3,4,5,6,7,8],'
                '"selected_expert_weights":null,"observed_at_ns":1}\n',
                encoding="utf-8",
            )
        return NativeProcessResult(
            return_code=0,
            started_at="2026-07-15T00:00:00+00:00",
            ended_at="2026-07-15T00:00:01+00:00",
            duration_seconds=1.0,
        )

    config = CollectionConfig(
        probe=probe,
        model=model,
        model_sha256=(
            "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fa"
            "d799250a0f70d652b6b825e4"
        ),
        output_dir=tmp_path / "runs",
    )
    first = collect_trace_pairs(
        corpus,
        config,
        process_runner=fake_runner,
        expected_split_counts={"train": 1},
        expected_domain_counts={"general_chat": 1},
    )
    first_attempt = first["shards"][0]["attempts"][0]
    first_tokens = first_attempt["instrumented"]["tokens"]["path"]
    assert first["shards"][0]["latest_status"] == "parity_failed"

    second = collect_trace_pairs(
        corpus,
        config,
        process_runner=fake_runner,
        expected_split_counts={"train": 1},
        expected_domain_counts={"general_chat": 1},
    )

    attempts = second["shards"][0]["attempts"]
    assert second["shards"][0]["latest_status"] == "passed"
    assert len(attempts) == 2
    assert attempts[0]["instrumented"]["tokens"]["path"] == first_tokens
    assert attempts[1]["instrumented"]["tokens"]["path"] != first_tokens
    assert Path(first_tokens).is_file()
