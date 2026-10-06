"""Bounded metadata-only GGUF inspection (GGML GGUF v2/v3 specification)."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from .runtime import file_digest

# GGML block width / stored bytes; pinned format constants from llama.cpp a7312ae9.
QUANT_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
               6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
               10: (256, 84), 11: (256, 110), 12: (256, 144),
               13: (256, 176), 14: (256, 210), 15: (256, 292),
               16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
               20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
               24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
               29: (256, 56), 30: (1, 2)}
VALUE_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _read_metadata(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        def read(count):
            if count < 0 or stream.tell() + count > 64 * 1024**2 or stream.tell() + count > size:
                raise ValueError("GGUF metadata or tensor header is truncated/oversized.")
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("GGUF file is truncated.")
            return data
        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]
        def string(*, keep=True, maximum=1024 * 1024):
            length = number("Q")
            if length > maximum:
                raise ValueError("GGUF metadata string exceeds inspection limit.")
            if keep:
                return read(length).decode("utf-8", errors="strict")
            read(length)
            return None
        def value(kind, keep=True):
            if kind == 8:
                return string(keep=keep)
            if kind in VALUE_FORMATS:
                val = number(VALUE_FORMATS[kind])
                return val if keep else None
            if kind == 9:
                item_kind, count = number("I"), number("Q")
                if count > 2_000_000 or item_kind == 9:
                    raise ValueError("GGUF metadata array exceeds supported bounds.")
                if item_kind in VALUE_FORMATS and (not keep or count > 1024):
                    read(count * struct.calcsize(VALUE_FORMATS[item_kind]))
                    return None
                if count > 1024:
                    for _ in range(count):
                        value(item_kind, False)
                    return None
                return [value(item_kind, keep) for _ in range(count)] if keep else [value(item_kind, False) for _ in range(count)] and None
            raise ValueError(f"Unsupported GGUF metadata type {kind}.")
        if read(4) != b"GGUF" or number("I") not in {2, 3}:
            raise ValueError("Only little-endian GGUF v2/v3 models are supported.")
        tensor_count, count = number("Q"), number("Q")
        if not 0 < tensor_count <= 1_000_000 or not 0 < count <= 100_000:
            raise ValueError("GGUF tensor/metadata count is invalid.")
        metadata = {}
        for _ in range(count):
            key = string(maximum=4096)
            if key in metadata:
                raise ValueError("Duplicate GGUF metadata key.")
            metadata[key] = value(number("I"), not key.startswith(("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores", "tokenizer.ggml.token_type")))
            if stream.tell() > 64 * 1024**2:
                raise ValueError("GGUF metadata exceeds the 64 MiB inspection limit.")
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a power of two up to 4096.")
        tensors, names = [], set()
        for _ in range(tensor_count):
            name = string(maximum=4096)
            dims = number("I")
            if name in names or not 1 <= dims <= 4:
                raise ValueError("Invalid GGUF tensor name/dimensions.")
            names.add(name)
            shape = [number("Q") for _ in range(dims)]
            kind, offset = number("I"), number("Q")
            if kind not in QUANT_SIZES:
                raise ValueError(f"Unsupported GGUF tensor type {kind}; update the format adapter.")
            block, width = QUANT_SIZES[kind]
            elements = math.prod(shape)
            if any(n == 0 for n in shape) or shape[0] % block or offset % alignment:
                raise ValueError("Invalid GGUF tensor shape/offset alignment.")
            tensors.append((offset, elements // block * width))
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        previous_end = 0
        for offset, length in sorted(tensors):
            if offset < previous_end or data_start + offset + length > size:
                raise ValueError("GGUF tensor data overlaps or is truncated/outside file.")
            previous_end = offset + length
        architecture = metadata.get("general.architecture")
        if not isinstance(architecture, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", architecture):
            raise ValueError("GGUF architecture metadata is missing or invalid.")
        return {"metadata": metadata, "architecture": architecture, "tensor_count": tensor_count}


def inspect_model(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError("GGUF model file is missing.")
    try:
        result = _read_metadata(path)
        metadata = result["metadata"]
        count = metadata.get("split.count", 1)
        paths = [path]
        if type(count) is not int or not 1 <= count <= 1000:
            raise ValueError("Invalid GGUF split shard count.")
        if count > 1:
            match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
            if not match or int(match[3]) != count:
                raise ValueError("Split GGUF shard naming does not match metadata.")
            paths = [path.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
            for index, shard in enumerate(paths):
                if not shard.is_file():
                    raise ValueError(f"Missing GGUF shard {shard.name}.")
                info = _read_metadata(shard)
                if info["architecture"] != result["architecture"] or info["metadata"].get("split.count") != count or info["metadata"].get("split.no") != index:
                    raise ValueError("GGUF shard metadata does not match the split set.")
            path = paths[0]
        files = [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_digest(p)} for p in paths]
        result.update(path=str(path), files=files, bytes=sum(f["bytes"] for f in files),
                      sha256=files[0]["sha256"] if len(files) == 1 else hashlib.sha256(json.dumps([(f["bytes"], f["sha256"]) for f in files]).encode()).hexdigest(),
                      name=metadata.get("general.name", path.stem),
                      context_limit=metadata.get(result["architecture"] + ".context_length"),
                      chat_template=metadata.get("tokenizer.chat_template"),
                      qualification="RUNTIME-LOAD-REQUIRED")
        return result
    except (UnicodeError, struct.error, OverflowError) as error:
        raise ValueError(f"Malformed GGUF metadata: {error}") from error


def estimate_memory(model: dict, context: int) -> dict:
    if type(context) is not int or not 1 <= context <= 1_048_576:
        raise ValueError("Context must be between 1 and 1048576 tokens.")
    metadata, prefix = model["metadata"], model["architecture"] + "."
    def positive(key):
        val = metadata.get(prefix + key)
        return val if type(val) is int and val > 0 else None
    layers, embed, heads, kv_heads = [positive(k) for k in ["block_count", "embedding_length", "attention.head_count", "attention.head_count_kv"]]
    kv = None
    if all([layers, embed, heads, kv_heads]):
        key = positive("attention.key_length") or embed // heads
        val = positive("attention.value_length") or embed // heads
        kv = layers * kv_heads * (key + val) * context * 2
    return {"kind": "ESTIMATE", "weights_bytes": model["bytes"], "kv_bytes": kv,
            "runtime_reserve_bytes": 1024**3,
            "resident_total_bytes": model["bytes"] + kv + 1024**3 if kv is not None else None,
            "uncertainty": "F16 KV estimate; architecture-specific/sliding/recurrent caches and runtime buffers may differ. A real load probe is required."}
