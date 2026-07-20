from __future__ import annotations

from source_contract_paths import LLAMA


def _backend_source() -> str:
    return (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")


def test_q6_split_profile_is_bounded_default_off_and_deferred() -> None:
    source = _backend_source()

    assert 'getenv("LLAMA_EXPERTFLOW_SPLIT_PROFILE")' in source
    assert "EXPERTFLOW_SPLIT_PROFILE_MAX_RECORDS" in source
    assert "ExpertFlow split profile capacity exceeded" in source
    assert "expertflow_split_profile_write" in source
    assert source.index("expertflow_split_profile_write(sched)") > source.index("void ggml_backend_sched_free")


def test_q6_split_profile_labels_diagnostic_synchronization() -> None:
    source = _backend_source()

    assert '\\"diagnostic_synchronization\\":true' in source
    assert "if (sched->expertflow_split_profile_enabled)" in source
    assert "expertflow_split_profile_record" in source
    assert "ggml_backend_synchronize(split_backend)" in source
