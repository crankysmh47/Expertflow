from pathlib import Path

from expertflow.doctor import parse_nvidia_smi_csv, tool_availability


def test_parses_nvidia_smi_snapshot() -> None:
    snapshots = parse_nvidia_smi_csv(
        "0, NVIDIA GeForce RTX 5060 Ti, 591.44, 16311, 2075, 14236\n"
    )

    assert len(snapshots) == 1
    gpu = snapshots[0]
    assert gpu.index == 0
    assert gpu.name == "NVIDIA GeForce RTX 5060 Ti"
    assert gpu.driver_version == "591.44"
    assert gpu.memory_total_mib == 16_311
    assert gpu.memory_used_mib == 2_075
    assert gpu.memory_free_mib == 14_236


def test_tool_availability_records_resolved_paths() -> None:
    locations = {
        "cmake": r"C:\tools\cmake.exe",
        "nvcc": None,
    }

    result = tool_availability(
        ("cmake", "nvcc"), resolver=lambda name: locations[name]
    )

    assert result == {
        "cmake": r"C:\tools\cmake.exe",
        "nvcc": None,
    }


def test_doctor_artifact_root_can_be_on_c_drive() -> None:
    root = Path(r"C:\models\expertflow")

    assert root.drive == "C:"
