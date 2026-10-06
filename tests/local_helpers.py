"""Small real GGUF fixtures, independent of model/runtime implementations."""
import struct


def write_gguf(path, *, architecture="llama", context=8192, extra=None):
    metadata = {"general.architecture": architecture, "general.name": "Tiny fixture",
                f"{architecture}.context_length": context,
                f"{architecture}.embedding_length": 64,
                f"{architecture}.block_count": 2,
                f"{architecture}.attention.head_count": 4,
                f"{architecture}.attention.head_count_kv": 2,
                "tokenizer.chat_template": "{{ messages }}"}
    metadata.update(extra or {})
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value in metadata.items():
        data += string(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    data += string("weight") + struct.pack("<IQQIQ", 2, 32, 32, 0, 0)
    data += b"\0" * ((-len(data)) % 32)
    path.write_bytes(data + b"\0" * 4096)
    return path
