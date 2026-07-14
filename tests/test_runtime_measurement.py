import pytest

from expertflow.runtime.measurement import parse_compute_apps_csv


def test_parses_and_combines_gpu_memory_by_process() -> None:
    result = parse_compute_apps_csv("123, 512\n456, 1024\n123, 64\n")

    assert result == {123: 576, 456: 1024}


def test_empty_compute_apps_snapshot_is_valid() -> None:
    assert parse_compute_apps_csv("") == {}


def test_skips_wddm_processes_without_per_process_memory() -> None:
    assert parse_compute_apps_csv("123, [N/A]\n456, 20\n") == {456: 20}


def test_rejects_malformed_compute_apps_snapshot() -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_compute_apps_csv("not-a-pid, 20")
