from pathlib import Path
import struct
import pytest
from local_helpers import write_gguf


def test_model_reads_architecture_template_and_context_without_loading_weights(tmp_path):
    from expertflow.product.models import inspect_model, estimate_memory
    path = write_gguf(tmp_path / "tiny ü.gguf")
    model = inspect_model(path)
    assert model["architecture"] == "llama"
    assert model["context_limit"] == 8192
    assert model["chat_template"] == "{{ messages }}"
    estimate = estimate_memory(model, 4096)
    # 2 layers * 2 KV heads * 16 head dims * 4096 * K/V * F16 bytes.
    assert estimate["kv_bytes"] == 1048576
    assert estimate["kind"] == "ESTIMATE"


@pytest.mark.parametrize("payload", [b"junk", b"GGUF" + struct.pack("<IQQ", 3, 0, 999999999)])
def test_malformed_gguf_rejected_before_runtime_launch(tmp_path, payload):
    from expertflow.product.models import inspect_model
    path = tmp_path / "bad.gguf"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="GGUF|metadata"):
        inspect_model(path)


def test_truncated_tensor_payload_rejected(tmp_path):
    from expertflow.product.models import inspect_model
    path = write_gguf(tmp_path / "tiny.gguf")
    path.write_bytes(path.read_bytes()[:-4000])
    with pytest.raises(ValueError, match="tensor|truncated"):
        inspect_model(path)


def test_missing_split_shard_rejected(tmp_path):
    from expertflow.product.models import inspect_model
    path = write_gguf(tmp_path / "tiny-00001-of-00002.gguf", extra={"split.count": 2, "split.no": 0})
    with pytest.raises(ValueError, match="shard"):
        inspect_model(path)


def test_tensor_offset_outside_file_is_rejected(tmp_path):
    from expertflow.product.models import inspect_model
    path = write_gguf(tmp_path / "tiny.gguf")
    payload = path.read_bytes()
    marker = struct.pack("<IQQIQ", 2, 32, 32, 0, 0)
    path.write_bytes(payload.replace(marker, struct.pack("<IQQIQ", 2, 32, 32, 0, 999999999), 1))
    with pytest.raises(ValueError, match="tensor"):
        inspect_model(path)


def test_nested_metadata_cannot_read_past_global_header_budget(tmp_path,monkeypatch):
    from expertflow.product.models import inspect_model
    path=tmp_path / "oversized-metadata.gguf"
    key=b"tokenizer.ggml.tokens"
    with path.open("wb") as stream:
        stream.write(b"GGUF"+struct.pack("<IQQ",3,1,1)+struct.pack("<Q",len(key))+key+struct.pack("<IIQ",9,8,70))
        for _ in range(70):
            stream.write(struct.pack("<Q",1024**2));stream.seek(1024**2,1)
        stream.write(b"\0")
    original=Path.open
    class Guard:
        def __init__(self,stream):self.stream,self.read_bytes=stream,0
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def __getattr__(self,name):return getattr(self.stream,name)
        def read(self,count):
            self.read_bytes+=count
            assert self.read_bytes<=64*1024**2,"Parser read past the declared global header budget"
            return self.stream.read(count)
    monkeypatch.setattr(Path,"open",lambda p,*args,**kwargs:Guard(original(p,*args,**kwargs)) if p==path and args==("rb",) else original(p,*args,**kwargs))
    with pytest.raises(ValueError,match="metadata|limit"):
        inspect_model(path)
