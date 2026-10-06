"""Real file-backed bindings must match the manifest, not just themselves."""
from dataclasses import replace
import json

import pytest

from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.runner import RuntimeBinding
from expertflow.compiler.schema import ArtifactIdentity


def runtime(tmp_path):
    def artifact(name):
        p = tmp_path / name
        p.write_bytes(name.encode())
        return ArtifactIdentity(str(p), p.stat().st_size, file_sha256(p))
    server, cli, dep, cuda = [artifact(n) for n in
        ('llama-server.exe', 'llama-cli.exe', 'ggml.dll', 'cudart64_12.dll')]
    # CUDA lives outside the binary DLL inventory.
    cuda_dir = tmp_path / 'cuda'
    cuda_dir.mkdir()
    moved = cuda_dir / 'cudart64_12.dll'
    (tmp_path / 'cudart64_12.dll').rename(moved)
    cuda = replace(cuda, path=str(moved))
    manifest = {'binaries': {'llama-server.exe': server.sha256, 'llama-cli.exe': cli.sha256},
        'dependencies': {'ggml.dll': dep.sha256}, 'cuda_runtime_sha256': cuda.sha256}
    return RuntimeBinding(server, json.dumps(manifest), (dep,), cuda)


@pytest.mark.parametrize('mutation', ['none', 'server', 'missing-dependency',
    'duplicate-dependency', 'dependency-hash', 'dependency-directory', 'cuda',
    'missing-cuda', 'extra-dll', 'cli'])
def test_real_runtime_manifest_binding(tmp_path, mutation):
    binding = runtime(tmp_path)
    if mutation == 'server':
        binding = replace(binding, server=replace(binding.server, sha256='0' * 64))
    elif mutation == 'missing-dependency':
        binding = replace(binding, dependencies=())
    elif mutation == 'duplicate-dependency':
        binding = replace(binding, dependencies=binding.dependencies * 2)
    elif mutation == 'dependency-hash':
        binding = replace(binding, dependencies=(replace(binding.dependencies[0], sha256='0' * 64),))
    elif mutation == 'dependency-directory':
        dep = binding.dependencies[0]
        outside = tmp_path / 'cuda' / 'ggml.dll'
        outside.write_bytes((tmp_path / 'ggml.dll').read_bytes())
        binding = replace(binding, dependencies=(replace(dep, path=str(outside)),))
    elif mutation == 'cuda':
        binding = replace(binding, cuda_runtime=replace(binding.cuda_runtime, sha256='0' * 64))
    elif mutation == 'missing-cuda':
        binding = replace(binding, cuda_runtime=None)
    elif mutation == 'extra-dll':
        (tmp_path / 'unreviewed.dll').write_bytes(b'unreviewed')
    elif mutation == 'cli':
        (tmp_path / 'llama-cli.exe').write_bytes(b'unreviewed')
    if mutation == 'none':
        binding.verify_manifest_bindings()
    else:
        with pytest.raises(ValueError, match='pinned runtime'):
            binding.verify_manifest_bindings()
