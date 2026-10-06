"""Dependency-free CUDA host-to-device transfer measurements."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import statistics
import time
from typing import Any


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_HOST_ALLOC_DEFAULT = 0x00
CUDA_HOST_ALLOC_WRITE_COMBINED = 0x04
CUDA_ERROR_NOT_READY = 600

_STAGING_MODES = frozenset({"pageable", "pinned", "pinned_wc"})


def summarize_latency_samples(
    payload_bytes: int, samples_ms: list[float]
) -> dict[str, float | int]:
    """Summarize positive per-copy latency samples."""

    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    if not samples_ms or any(sample <= 0 for sample in samples_ms):
        raise ValueError("samples_ms must contain positive values")

    ordered = sorted(samples_ms)
    mean_ms = statistics.fmean(ordered)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    bytes_per_second = payload_bytes / (mean_ms / 1000.0)
    return {
        "sample_count": len(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p50_ms": statistics.median(ordered),
        "mean_ms": mean_ms,
        "p95_ms": ordered[p95_index],
        "mean_gib_per_second": bytes_per_second / (1024**3),
    }


class CudaRuntime:
    """Narrow ctypes wrapper around the CUDA Runtime API."""

    def __init__(self, library_path: Path, *, device: int = 0) -> None:
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        self.library_path = library_path.resolve()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_api()
        self._check(self._cuda_set_device(device), "cudaSetDevice")

    def _bind(
        self,
        name: str,
        restype: type[ctypes._SimpleCData] | None,
        argtypes: list[Any],
    ) -> Any:
        function = getattr(self._library, name)
        function.restype = restype
        function.argtypes = argtypes
        return function

    def _configure_api(self) -> None:
        void_pointer = ctypes.c_void_p
        pointer_to_void = ctypes.POINTER(void_pointer)
        self._cuda_set_device = self._bind(
            "cudaSetDevice", ctypes.c_int, [ctypes.c_int]
        )
        self._cuda_runtime_get_version = self._bind(
            "cudaRuntimeGetVersion",
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_int)],
        )
        self._cuda_driver_get_version = self._bind(
            "cudaDriverGetVersion",
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_int)],
        )
        self._cuda_get_error_string = self._bind(
            "cudaGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )
        self._cuda_malloc = self._bind(
            "cudaMalloc", ctypes.c_int, [pointer_to_void, ctypes.c_size_t]
        )
        self._cuda_free = self._bind(
            "cudaFree", ctypes.c_int, [void_pointer]
        )
        self._cuda_malloc_host = self._bind(
            "cudaMallocHost",
            ctypes.c_int,
            [pointer_to_void, ctypes.c_size_t],
        )
        self._cuda_free_host = self._bind(
            "cudaFreeHost", ctypes.c_int, [void_pointer]
        )
        self._cuda_host_alloc = self._bind(
            "cudaHostAlloc",
            ctypes.c_int,
            [pointer_to_void, ctypes.c_size_t, ctypes.c_uint],
        )
        self._cuda_memcpy_async = self._bind(
            "cudaMemcpyAsync",
            ctypes.c_int,
            [
                void_pointer,
                void_pointer,
                ctypes.c_size_t,
                ctypes.c_int,
                void_pointer,
            ],
        )
        self._cuda_event_create = self._bind(
            "cudaEventCreate", ctypes.c_int, [pointer_to_void]
        )
        self._cuda_event_destroy = self._bind(
            "cudaEventDestroy", ctypes.c_int, [void_pointer]
        )
        self._cuda_event_record = self._bind(
            "cudaEventRecord",
            ctypes.c_int,
            [void_pointer, void_pointer],
        )
        self._cuda_event_synchronize = self._bind(
            "cudaEventSynchronize", ctypes.c_int, [void_pointer]
        )
        self._cuda_event_elapsed_time = self._bind(
            "cudaEventElapsedTime",
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_float), void_pointer, void_pointer],
        )
        self._cuda_event_query = self._bind(
            "cudaEventQuery", ctypes.c_int, [void_pointer]
        )
        self._cuda_stream_create = self._bind(
            "cudaStreamCreate", ctypes.c_int, [pointer_to_void]
        )
        self._cuda_stream_destroy = self._bind(
            "cudaStreamDestroy", ctypes.c_int, [void_pointer]
        )
        self._cuda_stream_synchronize = self._bind(
            "cudaStreamSynchronize", ctypes.c_int, [void_pointer]
        )

    def _allocate_host(
        self, payload_bytes: int, source_memory: str
    ) -> tuple[ctypes.c_void_p, Any | None]:
        pointer = ctypes.c_void_p()
        if source_memory == "pinned":
            self._check(
                self._cuda_malloc_host(ctypes.byref(pointer), payload_bytes),
                "cudaMallocHost",
            )
            return pointer, None
        if source_memory == "pinned_wc":
            self._check(
                self._cuda_host_alloc(
                    ctypes.byref(pointer),
                    payload_bytes,
                    CUDA_HOST_ALLOC_WRITE_COMBINED,
                ),
                "cudaHostAlloc(write-combined)",
            )
            return pointer, None
        owner = (ctypes.c_ubyte * payload_bytes)()
        return ctypes.cast(owner, ctypes.c_void_p), owner

    def _release_host(self, pointer: ctypes.c_void_p) -> None:
        if pointer.value:
            self._cuda_free_host(pointer)

    def _fill_source(
        self, pointer: ctypes.c_void_p, payload_bytes: int
    ) -> None:
        ctypes.memset(pointer, 0xA5, payload_bytes)

    def _check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        raw_message = self._cuda_get_error_string(code)
        message = raw_message.decode("utf-8") if raw_message else "unknown error"
        raise RuntimeError(f"{operation} failed with CUDA {code}: {message}")

    def versions(self) -> dict[str, int]:
        runtime = ctypes.c_int()
        driver = ctypes.c_int()
        self._check(
            self._cuda_runtime_get_version(ctypes.byref(runtime)),
            "cudaRuntimeGetVersion",
        )
        self._check(
            self._cuda_driver_get_version(ctypes.byref(driver)),
            "cudaDriverGetVersion",
        )
        return {"runtime": runtime.value, "driver": driver.value}

    def measure(
        self,
        payload_bytes: int,
        *,
        source_memory: str,
        batches: int,
        copies_per_batch: int,
        warmup_copies: int,
    ) -> tuple[list[float], list[float]]:
        """Return CUDA-event and host-wall per-copy milliseconds."""

        if source_memory not in {"pageable", "pinned"}:
            raise ValueError("source_memory must be pageable or pinned")

        device_pointer = ctypes.c_void_p()
        pinned_pointer = ctypes.c_void_p()
        start_event = ctypes.c_void_p()
        end_event = ctypes.c_void_p()
        pageable_owner: Any | None = None
        try:
            self._check(
                self._cuda_malloc(ctypes.byref(device_pointer), payload_bytes),
                "cudaMalloc",
            )
            if source_memory == "pinned":
                self._check(
                    self._cuda_malloc_host(
                        ctypes.byref(pinned_pointer), payload_bytes
                    ),
                    "cudaMallocHost",
                )
                source_pointer = pinned_pointer
            else:
                pageable_owner = (ctypes.c_ubyte * payload_bytes)()
                source_pointer = ctypes.cast(
                    pageable_owner, ctypes.c_void_p
                )
            ctypes.memset(source_pointer, 0xA5, payload_bytes)

            self._check(
                self._cuda_event_create(ctypes.byref(start_event)),
                "cudaEventCreate(start)",
            )
            self._check(
                self._cuda_event_create(ctypes.byref(end_event)),
                "cudaEventCreate(end)",
            )
            for _ in range(warmup_copies):
                self._copy(device_pointer, source_pointer, payload_bytes)
            self._check(
                self._cuda_event_record(end_event, None),
                "cudaEventRecord(warmup)",
            )
            self._check(
                self._cuda_event_synchronize(end_event),
                "cudaEventSynchronize(warmup)",
            )

            event_samples: list[float] = []
            wall_samples: list[float] = []
            for _ in range(batches):
                wall_started = time.perf_counter_ns()
                self._check(
                    self._cuda_event_record(start_event, None),
                    "cudaEventRecord(start)",
                )
                for _ in range(copies_per_batch):
                    self._copy(
                        device_pointer, source_pointer, payload_bytes
                    )
                self._check(
                    self._cuda_event_record(end_event, None),
                    "cudaEventRecord(end)",
                )
                self._check(
                    self._cuda_event_synchronize(end_event),
                    "cudaEventSynchronize(end)",
                )
                wall_elapsed_ms = (
                    time.perf_counter_ns() - wall_started
                ) / 1_000_000
                event_elapsed_ms = ctypes.c_float()
                self._check(
                    self._cuda_event_elapsed_time(
                        ctypes.byref(event_elapsed_ms),
                        start_event,
                        end_event,
                    ),
                    "cudaEventElapsedTime",
                )
                event_samples.append(
                    event_elapsed_ms.value / copies_per_batch
                )
                wall_samples.append(wall_elapsed_ms / copies_per_batch)
            return event_samples, wall_samples
        finally:
            if end_event.value:
                self._cuda_event_destroy(end_event)
            if start_event.value:
                self._cuda_event_destroy(start_event)
            if pinned_pointer.value:
                self._cuda_free_host(pinned_pointer)
            if device_pointer.value:
                self._cuda_free(device_pointer)
            del pageable_owner

    def measure_pageable_to_pinned(
        self,
        payload_bytes: int,
        *,
        batches: int,
        copies_per_batch: int,
        warmup_copies: int,
    ) -> list[float]:
        """Return host-wall per-copy milliseconds for pinned staging."""

        pinned_pointer = ctypes.c_void_p()
        pageable_owner = (ctypes.c_ubyte * payload_bytes)()
        pageable_pointer = ctypes.cast(pageable_owner, ctypes.c_void_p)
        try:
            self._check(
                self._cuda_malloc_host(
                    ctypes.byref(pinned_pointer), payload_bytes
                ),
                "cudaMallocHost",
            )
            ctypes.memset(pageable_pointer, 0x5A, payload_bytes)
            for _ in range(warmup_copies):
                ctypes.memmove(
                    pinned_pointer, pageable_pointer, payload_bytes
                )

            samples: list[float] = []
            for _ in range(batches):
                started = time.perf_counter_ns()
                for _ in range(copies_per_batch):
                    ctypes.memmove(
                        pinned_pointer, pageable_pointer, payload_bytes
                    )
                elapsed_ms = (
                    time.perf_counter_ns() - started
                ) / 1_000_000
                samples.append(elapsed_ms / copies_per_batch)
            return samples
        finally:
            if pinned_pointer.value:
                self._cuda_free_host(pinned_pointer)
            del pageable_owner

    def measure_single_copy(
        self,
        payload_bytes: int,
        *,
        source_memory: str,
        samples: int,
        warmup_copies: int,
    ) -> tuple[list[float], list[float]]:
        """Return idle-stream single-copy event and host API milliseconds."""

        if source_memory not in {"pageable", "pinned"}:
            raise ValueError("source_memory must be pageable or pinned")
        if payload_bytes <= 0 or samples <= 0 or warmup_copies < 0:
            raise ValueError("single-copy measurement contract is invalid")

        device_pointer = ctypes.c_void_p()
        pinned_pointer = ctypes.c_void_p()
        start_event = ctypes.c_void_p()
        end_event = ctypes.c_void_p()
        pageable_owner: Any | None = None
        try:
            self._check(
                self._cuda_malloc(ctypes.byref(device_pointer), payload_bytes),
                "cudaMalloc",
            )
            if source_memory == "pinned":
                self._check(
                    self._cuda_malloc_host(
                        ctypes.byref(pinned_pointer), payload_bytes
                    ),
                    "cudaMallocHost",
                )
                source_pointer = pinned_pointer
            else:
                pageable_owner = (ctypes.c_ubyte * payload_bytes)()
                source_pointer = ctypes.cast(
                    pageable_owner, ctypes.c_void_p
                )
            ctypes.memset(source_pointer, 0x3C, payload_bytes)
            self._check(
                self._cuda_event_create(ctypes.byref(start_event)),
                "cudaEventCreate(start)",
            )
            self._check(
                self._cuda_event_create(ctypes.byref(end_event)),
                "cudaEventCreate(end)",
            )
            for _ in range(warmup_copies):
                self._copy(device_pointer, source_pointer, payload_bytes)
            self._check(
                self._cuda_event_record(end_event, None),
                "cudaEventRecord(warmup)",
            )
            self._check(
                self._cuda_event_synchronize(end_event),
                "cudaEventSynchronize(warmup)",
            )

            event_samples: list[float] = []
            enqueue_samples: list[float] = []
            for _ in range(samples):
                self._check(
                    self._cuda_event_record(start_event, None),
                    "cudaEventRecord(start)",
                )
                enqueue_started = time.perf_counter_ns()
                self._copy(device_pointer, source_pointer, payload_bytes)
                enqueue_samples.append(
                    (time.perf_counter_ns() - enqueue_started) / 1_000_000
                )
                self._check(
                    self._cuda_event_record(end_event, None),
                    "cudaEventRecord(end)",
                )
                self._check(
                    self._cuda_event_synchronize(end_event),
                    "cudaEventSynchronize(end)",
                )
                event_elapsed_ms = ctypes.c_float()
                self._check(
                    self._cuda_event_elapsed_time(
                        ctypes.byref(event_elapsed_ms),
                        start_event,
                        end_event,
                    ),
                    "cudaEventElapsedTime",
                )
                event_samples.append(event_elapsed_ms.value)
            return event_samples, enqueue_samples
        finally:
            if end_event.value:
                self._cuda_event_destroy(end_event)
            if start_event.value:
                self._cuda_event_destroy(start_event)
            if pinned_pointer.value:
                self._cuda_free_host(pinned_pointer)
            if device_pointer.value:
                self._cuda_free(device_pointer)
            del pageable_owner

    def measure_contiguous_batch(
        self,
        slot_bytes: int,
        slot_count: int,
        *,
        source_memory: str,
        batches: int,
        warmup_copies: int,
    ) -> dict[str, list[float]]:
        """Compare one contiguous N-slot copy against N separate copies."""

        if (
            source_memory not in _STAGING_MODES
            or slot_bytes <= 0
            or slot_count <= 0
            or batches <= 0
            or warmup_copies < 0
        ):
            raise ValueError("contiguous-batch measurement contract is invalid")

        total_bytes = slot_bytes * slot_count
        device_pointer = ctypes.c_void_p()
        host_pointer = ctypes.c_void_p()
        start_event = ctypes.c_void_p()
        end_event = ctypes.c_void_p()
        pageable_owner: Any | None = None
        try:
            self._check(
                self._cuda_malloc(ctypes.byref(device_pointer), total_bytes),
                "cudaMalloc",
            )
            host_pointer, pageable_owner = self._allocate_host(
                total_bytes, source_memory
            )
            self._fill_source(host_pointer, total_bytes)
            self._check(
                self._cuda_event_create(ctypes.byref(start_event)),
                "cudaEventCreate(start)",
            )
            self._check(
                self._cuda_event_create(ctypes.byref(end_event)),
                "cudaEventCreate(end)",
            )

            def timed_run(copy_action: Any) -> float:
                self._check(
                    self._cuda_event_record(start_event, None),
                    "cudaEventRecord(start)",
                )
                copy_action()
                self._check(
                    self._cuda_event_record(end_event, None),
                    "cudaEventRecord(end)",
                )
                self._check(
                    self._cuda_event_synchronize(end_event),
                    "cudaEventSynchronize(end)",
                )
                event_elapsed_ms = ctypes.c_float()
                self._check(
                    self._cuda_event_elapsed_time(
                        ctypes.byref(event_elapsed_ms),
                        start_event,
                        end_event,
                    ),
                    "cudaEventElapsedTime",
                )
                return event_elapsed_ms.value

            def destination_at(offset: int) -> ctypes.c_void_p:
                return ctypes.c_void_p(device_pointer.value + offset)

            def source_at(offset: int) -> ctypes.c_void_p:
                return ctypes.c_void_p(host_pointer.value + offset)

            for _ in range(warmup_copies):
                self._copy(destination_at(0), source_at(0), total_bytes)
            self._check(
                self._cuda_event_record(end_event, None),
                "cudaEventRecord(warmup)",
            )
            self._check(
                self._cuda_event_synchronize(end_event),
                "cudaEventSynchronize(warmup)",
            )

            batched_samples: list[float] = []
            individual_samples: list[float] = []
            for _ in range(batches):
                batched_samples.append(
                    timed_run(
                        lambda: self._copy(
                            destination_at(0), source_at(0), total_bytes
                        )
                    )
                )
                individual_samples.append(
                    timed_run(
                        lambda: [
                            self._copy(
                                destination_at(index * slot_bytes),
                                source_at(index * slot_bytes),
                                slot_bytes,
                            )
                            for index in range(slot_count)
                        ]
                    )
                )
            return {
                "batched": batched_samples,
                "individual": individual_samples,
            }
        finally:
            if end_event.value:
                self._cuda_event_destroy(end_event)
            if start_event.value:
                self._cuda_event_destroy(start_event)
            self._release_host(host_pointer)
            if device_pointer.value:
                self._cuda_free(device_pointer)
            del pageable_owner

    def measure_queue_depth(
        self,
        slot_bytes: int,
        depth: int,
        *,
        source_memory: str,
        batches: int,
        warmup_copies: int,
    ) -> tuple[list[float], list[float]]:
        """Enqueue depth copies without syncing and time each with events."""

        if (
            source_memory not in _STAGING_MODES
            or slot_bytes <= 0
            or depth <= 0
            or batches <= 0
            or warmup_copies < 0
        ):
            raise ValueError("queue-depth measurement contract is invalid")

        device_pointer = ctypes.c_void_p()
        host_pointer = ctypes.c_void_p()
        markers: list[ctypes.c_void_p] = []
        pageable_owner: Any | None = None
        try:
            self._check(
                self._cuda_malloc(ctypes.byref(device_pointer), slot_bytes),
                "cudaMalloc",
            )
            host_pointer, pageable_owner = self._allocate_host(
                slot_bytes, source_memory
            )
            self._fill_source(host_pointer, slot_bytes)
            for index in range(depth + 1):
                marker = ctypes.c_void_p()
                self._check(
                    self._cuda_event_create(ctypes.byref(marker)),
                    f"cudaEventCreate(marker-{index})",
                )
                markers.append(marker)
            for _ in range(warmup_copies):
                self._copy(device_pointer, host_pointer, slot_bytes)
            self._check(
                self._cuda_event_record(markers[depth], None),
                "cudaEventRecord(warmup)",
            )
            self._check(
                self._cuda_event_synchronize(markers[depth]),
                "cudaEventSynchronize(warmup)",
            )

            per_copy_samples: list[float] = []
            batch_wall_samples: list[float] = []
            elapsed_ms = ctypes.c_float()
            for _ in range(batches):
                wall_started = time.perf_counter_ns()
                self._check(
                    self._cuda_event_record(markers[0], None),
                    "cudaEventRecord(queue-start)",
                )
                for index in range(depth):
                    self._copy(device_pointer, host_pointer, slot_bytes)
                    self._check(
                        self._cuda_event_record(markers[index + 1], None),
                        f"cudaEventRecord(marker-{index + 1})",
                    )
                self._check(
                    self._cuda_event_synchronize(markers[depth]),
                    "cudaEventSynchronize(queue-end)",
                )
                batch_wall_samples.append(
                    (time.perf_counter_ns() - wall_started) / 1_000_000
                )
                for index in range(depth):
                    self._check(
                        self._cuda_event_elapsed_time(
                            ctypes.byref(elapsed_ms),
                            markers[index],
                            markers[index + 1],
                        ),
                        "cudaEventElapsedTime",
                    )
                    per_copy_samples.append(elapsed_ms.value)
            return per_copy_samples, batch_wall_samples
        finally:
            for marker in reversed(markers):
                if marker.value:
                    self._cuda_event_destroy(marker)
            self._release_host(host_pointer)
            if device_pointer.value:
                self._cuda_free(device_pointer)
            del pageable_owner

    def measure_ready_latency(
        self,
        slot_bytes: int,
        *,
        source_memory: str,
        background_bytes: int,
        background_copies: int,
        samples: int,
        warmup_copies: int,
    ) -> list[float]:
        """Measure enqueue-to-ready latency while a copy queue may be busy."""

        if (
            source_memory not in _STAGING_MODES
            or slot_bytes <= 0
            or background_bytes < 0
            or background_copies < 0
            or samples <= 0
            or warmup_copies < 0
        ):
            raise ValueError("ready-latency measurement contract is invalid")
        if (background_bytes > 0) != (background_copies > 0):
            raise ValueError(
                "background_bytes and background_copies must both be positive or both zero"
            )

        probe_device = ctypes.c_void_p()
        probe_host = ctypes.c_void_p()
        load_device = ctypes.c_void_p()
        load_host = ctypes.c_void_p()
        ready_event = ctypes.c_void_p()
        load_stream = ctypes.c_void_p()
        probe_stream = ctypes.c_void_p()
        probe_pageable: Any | None = None
        load_pageable: Any | None = None
        try:
            self._check(
                self._cuda_malloc(ctypes.byref(probe_device), slot_bytes),
                "cudaMalloc(probe)",
            )
            probe_host, probe_pageable = self._allocate_host(
                slot_bytes, source_memory
            )
            self._fill_source(probe_host, slot_bytes)
            self._check(
                self._cuda_event_create(ctypes.byref(ready_event)),
                "cudaEventCreate(ready)",
            )
            self._check(
                self._cuda_stream_create(ctypes.byref(load_stream)),
                "cudaStreamCreate(load)",
            )
            self._check(
                self._cuda_stream_create(ctypes.byref(probe_stream)),
                "cudaStreamCreate(probe)",
            )
            if background_bytes > 0:
                self._check(
                    self._cuda_malloc(
                        ctypes.byref(load_device), background_bytes
                    ),
                    "cudaMalloc(load)",
                )
                load_host, load_pageable = self._allocate_host(
                    background_bytes, "pinned"
                )
                self._fill_source(load_host, background_bytes)
                for _ in range(warmup_copies):
                    self._check(
                        self._cuda_memcpy_async(
                            load_device,
                            load_host,
                            background_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            load_stream,
                        ),
                        "cudaMemcpyAsync(load-warmup)",
                    )
                self._check(
                    self._cuda_stream_synchronize(load_stream),
                    "cudaStreamSynchronize(load-warmup)",
                )
            else:
                for _ in range(warmup_copies):
                    self._check(
                        self._cuda_memcpy_async(
                            probe_device,
                            probe_host,
                            slot_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            probe_stream,
                        ),
                        "cudaMemcpyAsync(probe-warmup)",
                    )
                self._check(
                    self._cuda_stream_synchronize(probe_stream),
                    "cudaStreamSynchronize(probe-warmup)",
                )

            ready_samples: list[float] = []
            for _ in range(samples):
                for _ in range(background_copies):
                    self._check(
                        self._cuda_memcpy_async(
                            load_device,
                            load_host,
                            background_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            load_stream,
                        ),
                        "cudaMemcpyAsync(load)",
                    )
                enqueue_started = time.perf_counter_ns()
                self._check(
                    self._cuda_memcpy_async(
                        probe_device,
                        probe_host,
                        slot_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        probe_stream,
                    ),
                    "cudaMemcpyAsync(probe)",
                )
                self._check(
                    self._cuda_event_record(ready_event, probe_stream),
                    "cudaEventRecord(ready)",
                )
                while True:
                    code = self._cuda_event_query(ready_event)
                    if code == 0:
                        break
                    if code != CUDA_ERROR_NOT_READY:
                        self._check(code, "cudaEventQuery(ready)")
                ready_samples.append(
                    (time.perf_counter_ns() - enqueue_started) / 1_000_000
                )
                self._check(
                    self._cuda_stream_synchronize(load_stream),
                    "cudaStreamSynchronize(load-drain)",
                )
                self._check(
                    self._cuda_stream_synchronize(probe_stream),
                    "cudaStreamSynchronize(probe-drain)",
                )
            return ready_samples
        finally:
            if probe_stream.value:
                self._cuda_stream_destroy(probe_stream)
            if load_stream.value:
                self._cuda_stream_destroy(load_stream)
            if ready_event.value:
                self._cuda_event_destroy(ready_event)
            if load_device.value:
                self._cuda_free(load_device)
            self._release_host(load_host)
            self._release_host(probe_host)
            if probe_device.value:
                self._cuda_free(probe_device)
            del load_pageable
            del probe_pageable

    def _copy(
        self,
        destination: ctypes.c_void_p,
        source: ctypes.c_void_p,
        payload_bytes: int,
    ) -> None:
        self._check(
            self._cuda_memcpy_async(
                destination,
                source,
                payload_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                None,
            ),
            "cudaMemcpyAsync",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_cuda_transfers(
    runtime: Path,
    *,
    payload_bytes: tuple[int, ...],
    batches: int,
    copies_per_batch: int,
    warmup_copies: int,
    device: int,
    single_copy_samples: int = 200,
) -> dict[str, object]:
    """Measure pageable and pinned H2D copies for each payload size."""

    if not payload_bytes or any(size <= 0 for size in payload_bytes):
        raise ValueError("payload_bytes must contain positive values")
    if (
        batches <= 0
        or copies_per_batch <= 0
        or warmup_copies < 0
        or single_copy_samples <= 0
    ):
        raise ValueError("benchmark iteration counts are invalid")
    if device < 0:
        raise ValueError("device must be non-negative")

    resolved_runtime = runtime.resolve()
    cuda = CudaRuntime(resolved_runtime, device=device)
    runs: list[dict[str, object]] = []
    for size in payload_bytes:
        staging_samples = cuda.measure_pageable_to_pinned(
            size,
            batches=batches,
            copies_per_batch=copies_per_batch,
            warmup_copies=warmup_copies,
        )
        runs.append(
            {
                "payload_bytes": size,
                "source_memory": "pageable",
                "destination_memory": "pinned",
                "direction": "pageable_to_pinned",
                "host_wall_per_copy": summarize_latency_samples(
                    size, staging_samples
                ),
                "raw_host_wall_ms_per_copy": staging_samples,
            }
        )
        for source_memory in ("pageable", "pinned"):
            event_samples, wall_samples = cuda.measure(
                size,
                source_memory=source_memory,
                batches=batches,
                copies_per_batch=copies_per_batch,
                warmup_copies=warmup_copies,
            )
            single_event_samples, enqueue_samples = cuda.measure_single_copy(
                size,
                source_memory=source_memory,
                samples=single_copy_samples,
                warmup_copies=warmup_copies,
            )
            runs.append(
                {
                    "payload_bytes": size,
                    "source_memory": source_memory,
                    "direction": "host_to_device",
                    "cuda_event_per_copy": summarize_latency_samples(
                        size, event_samples
                    ),
                    "host_wall_per_copy": summarize_latency_samples(
                        size, wall_samples
                    ),
                    "single_copy_cuda_event": summarize_latency_samples(
                        size, single_event_samples
                    ),
                    "host_enqueue": summarize_latency_samples(
                        size, enqueue_samples
                    ),
                    "raw_cuda_event_ms_per_copy": event_samples,
                    "raw_host_wall_ms_per_copy": wall_samples,
                    "raw_single_copy_cuda_event_ms": single_event_samples,
                    "raw_host_enqueue_ms": enqueue_samples,
                }
            )

    return {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "path": str(resolved_runtime),
            "sha256": _sha256(resolved_runtime),
            "versions": cuda.versions(),
        },
        "device_index": device,
        "contract": {
            "batches": batches,
            "copies_per_batch": copies_per_batch,
            "warmup_copies": warmup_copies,
            "single_copy_samples": single_copy_samples,
            "batch_sample_unit": "per-copy average within each batch",
            "single_copy_sample_unit": (
                "one idle-stream copy bracketed by CUDA events"
            ),
            "host_enqueue_sample_unit": (
                "host duration of one cudaMemcpyAsync API call after prior synchronization"
            ),
        },
        "runs": runs,
    }


def _validated_mover_contract(
    *,
    slot_bytes_values: tuple[int, ...],
    slot_counts: tuple[int, ...],
    queue_depths: tuple[int, ...],
    batches: int,
    warmup_copies: int,
    ready_samples: int,
    background_bytes: int,
    background_copies: int,
    staging_mode: str,
    device: int,
) -> None:
    if (
        not slot_bytes_values
        or any(size <= 0 for size in slot_bytes_values)
        or not slot_counts
        or any(count <= 0 for count in slot_counts)
        or not queue_depths
        or any(depth <= 0 for depth in queue_depths)
        or batches <= 0
        or warmup_copies < 0
        or ready_samples <= 0
        or background_bytes < 0
        or background_copies < 0
        or device < 0
    ):
        raise ValueError("mover benchmark iteration contract is invalid")
    if staging_mode not in _STAGING_MODES:
        raise ValueError("staging_mode must be pageable, pinned, or pinned_wc")
    if (background_bytes > 0) != (background_copies > 0):
        raise ValueError(
            "background_bytes and background_copies must both be positive or both zero"
        )


def benchmark_mover(
    runtime: Path,
    *,
    slot_bytes_values: tuple[int, ...],
    slot_counts: tuple[int, ...],
    queue_depths: tuple[int, ...],
    batches: int,
    warmup_copies: int,
    ready_samples: int,
    background_bytes: int,
    background_copies: int,
    staging_mode: str = "pinned",
    device: int = 0,
) -> dict[str, object]:
    """Measure contiguous batching, queue depth, and ready-latency modes."""

    _validated_mover_contract(
        slot_bytes_values=slot_bytes_values,
        slot_counts=slot_counts,
        queue_depths=queue_depths,
        batches=batches,
        warmup_copies=warmup_copies,
        ready_samples=ready_samples,
        background_bytes=background_bytes,
        background_copies=background_copies,
        staging_mode=staging_mode,
        device=device,
    )

    resolved_runtime = runtime.resolve()
    cuda = CudaRuntime(resolved_runtime, device=device)

    contiguous_runs: list[dict[str, object]] = []
    for size in slot_bytes_values:
        for count in slot_counts:
            samples = cuda.measure_contiguous_batch(
                size,
                count,
                source_memory=staging_mode,
                batches=batches,
                warmup_copies=warmup_copies,
            )
            batched_summary = summarize_latency_samples(
                size * count, samples["batched"]
            )
            individual_summary = summarize_latency_samples(
                size * count, samples["individual"]
            )
            contiguous_runs.append(
                {
                    "slot_bytes": size,
                    "slot_count": count,
                    "payload_bytes": size * count,
                    "source_memory": staging_mode,
                    "batched": {
                        **batched_summary,
                        "raw_cuda_event_ms_per_copy": samples["batched"],
                    },
                    "individual": {
                        **individual_summary,
                        "raw_cuda_event_ms_per_copy": samples["individual"],
                    },
                    "speedup_ratio": (
                        individual_summary["mean_ms"]
                        / batched_summary["mean_ms"]
                    ),
                }
            )

    queue_runs: list[dict[str, object]] = []
    for size in slot_bytes_values:
        for depth in queue_depths:
            per_copy, batch_wall = cuda.measure_queue_depth(
                size,
                depth,
                source_memory=staging_mode,
                batches=batches,
                warmup_copies=warmup_copies,
            )
            queue_runs.append(
                {
                    "slot_bytes": size,
                    "queue_depth": depth,
                    "source_memory": staging_mode,
                    "per_copy_event": summarize_latency_samples(size, per_copy),
                    "batch_wall_total": summarize_latency_samples(
                        size * depth, batch_wall
                    ),
                    "raw_per_copy_event_ms": per_copy,
                    "raw_batch_wall_ms": batch_wall,
                }
            )

    ready_runs: list[dict[str, object]] = []
    for size in slot_bytes_values:
        idle = cuda.measure_ready_latency(
            size,
            source_memory=staging_mode,
            background_bytes=0,
            background_copies=0,
            samples=ready_samples,
            warmup_copies=warmup_copies,
        )
        ready_runs.append(
            {
                "slot_bytes": size,
                "load": "idle",
                "host_ready_latency": summarize_latency_samples(1, [
                    max(sample, 1e-9) for sample in idle
                ]),
                "raw_host_ready_ms": idle,
            }
        )
        if background_bytes > 0:
            loaded = cuda.measure_ready_latency(
                size,
                source_memory=staging_mode,
                background_bytes=background_bytes,
                background_copies=background_copies,
                samples=ready_samples,
                warmup_copies=warmup_copies,
            )
            ready_runs.append(
                {
                    "slot_bytes": size,
                    "load": "copy_engine_busy",
                    "host_ready_latency": summarize_latency_samples(1, [
                        max(sample, 1e-9) for sample in loaded
                    ]),
                    "raw_host_ready_ms": loaded,
                }
            )

    return {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "path": str(resolved_runtime),
            "sha256": _sha256(resolved_runtime),
            "versions": cuda.versions(),
        },
        "device_index": device,
        "contract": {
            "slot_bytes_values": list(slot_bytes_values),
            "slot_counts": list(slot_counts),
            "queue_depths": list(queue_depths),
            "batches": batches,
            "warmup_copies": warmup_copies,
            "ready_samples": ready_samples,
            "background_bytes": background_bytes,
            "background_copies": background_copies,
            "staging_mode": staging_mode,
            "contiguous_sample_unit": (
                "one CUDA-event-bracketed copy round; batched moves all slots "
                "in one memcpy, individual issues one memcpy per slot"
            ),
            "queue_sample_unit": (
                "per-copy CUDA-event delta between consecutive markers inside "
                "one unsynchronized enqueue burst"
            ),
            "ready_sample_unit": (
                "host wall time from cudaMemcpyAsync enqueue until "
                "cudaEventQuery first reports completion"
            ),
        },
        "contiguous_batch": contiguous_runs,
        "queue_depth": queue_runs,
        "ready_latency": ready_runs,
    }


def aggregate_cuda_transfer_trials(
    reports: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    source_paths: tuple[str, ...],
) -> dict[str, object]:
    """Pool raw latency samples from compatible independent trials."""

    trials = tuple(reports)
    if not trials or len(trials) != len(source_paths):
        raise ValueError("reports and source_paths must have equal nonzero length")
    first = trials[0]
    for field in ("runtime", "device_index", "contract"):
        if any(report.get(field) != first.get(field) for report in trials[1:]):
            raise ValueError(f"transfer trial {field} values must match")
    if any(report.get("measurement_kind") != "measured" for report in trials):
        raise ValueError("transfer trials must be measured")

    descriptor_fields = (
        "payload_bytes",
        "source_memory",
        "destination_memory",
        "direction",
    )
    raw_to_summary = {
        "raw_host_wall_ms_per_copy": "host_wall_per_copy",
        "raw_cuda_event_ms_per_copy": "cuda_event_per_copy",
        "raw_single_copy_cuda_event_ms": "single_copy_cuda_event",
        "raw_host_enqueue_ms": "host_enqueue",
    }

    def descriptor(run: dict[str, object]) -> tuple[object, ...]:
        return tuple(run.get(field) for field in descriptor_fields)

    indexed_trials: list[dict[tuple[object, ...], dict[str, object]]] = []
    for report in trials:
        raw_runs = report.get("runs")
        if not isinstance(raw_runs, list) or not raw_runs:
            raise ValueError("transfer trial runs must be a non-empty array")
        index = {
            descriptor(run): run
            for run in raw_runs
            if isinstance(run, dict)
        }
        if len(index) != len(raw_runs):
            raise ValueError("transfer run descriptors must be unique objects")
        indexed_trials.append(index)
    run_keys = set(indexed_trials[0])
    if any(set(index) != run_keys for index in indexed_trials[1:]):
        raise ValueError("transfer trial run sets must match")

    pooled_runs: list[dict[str, object]] = []
    for key in sorted(run_keys, key=lambda item: tuple(str(value) for value in item)):
        runs = [index[key] for index in indexed_trials]
        payload_bytes = key[0]
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
            raise ValueError("transfer payload_bytes must be an integer")
        pooled: dict[str, object] = {
            field: value
            for field, value in zip(descriptor_fields, key)
            if value is not None
        }
        for raw_field, summary_field in raw_to_summary.items():
            if not any(raw_field in run for run in runs):
                continue
            if not all(isinstance(run.get(raw_field), list) for run in runs):
                raise ValueError(f"transfer trials must all contain {raw_field}")
            samples = [
                float(sample)
                for run in runs
                for sample in run[raw_field]
            ]
            pooled[raw_field] = samples
            pooled[summary_field] = summarize_latency_samples(
                payload_bytes, samples
            )
        pooled_runs.append(pooled)

    return {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregation": "raw_samples_pooled_across_independent_trials",
        "trial_count": len(trials),
        "source_trials": list(source_paths),
        "runtime": first["runtime"],
        "device_index": first["device_index"],
        "contract": first["contract"],
        "runs": pooled_runs,
    }
