"""Separate Q4 provider scope contracts; no real weights or native launch here."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from expertflow.compiler.schema import ArtifactIdentity, MoELayerIR, canonical_sha256
from test_compiler_schema import model_fixture


AUDIT = json.loads(Path('docs/evidence/stock-discovery-20261004/q4-scheduling-source-audit.json').read_text())
MODEL_SHA = '4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5'


def api():
    from expertflow.compiler import stock_eligibility
    return stock_eligibility


def fixtures():
    components = (512, 142737408, 285474816)
    bank = sum(components)
    model = replace(model_fixture(), quantization='Q4_0',
        identity=ArtifactIdentity('fixture-q4.gguf', 14439361440, MODEL_SHA),
        moe_layers=tuple(MoELayerIR(i, 128, 8, bank // 128, bank, components) for i in range(30)))
    binding = SimpleNamespace(manifest_json=Path('configs/compiler/runtime-stock.json').read_text(),
        sha256='a' * 64, verify=lambda: None, verify_manifest_bindings=lambda: None)
    return SimpleNamespace(model=model, stock=binding, hardware={'supported_features': ['cuda_graphs']}), {'architecture': 'AMD64'}


def reader(repository, revision, path):
    assert revision == AUDIT['upstream_commit']
    return AUDIT['source_object_ids'][path]


def test_q4_provider_requires_its_own_repack_and_conversion_source_objects(tmp_path):
    inp, host = fixtures()
    result = api().EligibilityRegistry.with_builtins().attest(inp, host, tmp_path, source_reader=reader)
    assert result['provider_id'] == AUDIT['proposed_provider_id']
    assert result['source_object_ids'] == AUDIT['source_object_ids']
    assert result['model_ir_sha256'] == canonical_sha256(inp.model)
    assert result['allowed_controls'] == ['threads', 'cuda_graphs']
    assert result['host_environment_sha256'] == canonical_sha256(host)


@pytest.mark.parametrize('corruption', ['weights', 'size', 'quantization', 'architecture', 'layers', 'bundle', 'components', 'runtime', 'host', 'source', 'capability'])
def test_q4_scope_rejects_changed_or_unsupported_inputs(tmp_path, corruption):
    inp, host = fixtures()
    source_reader = reader
    if corruption == 'weights':
        inp.model = replace(inp.model, identity=replace(inp.model.identity, sha256='f' * 64))
    elif corruption == 'size':
        inp.model = replace(inp.model, identity=replace(inp.model.identity, size_bytes=1))
    elif corruption == 'quantization':
        inp.model = replace(inp.model, quantization='Q6_K')
    elif corruption == 'architecture':
        inp.model = replace(inp.model, architecture='unreviewed')
    elif corruption == 'layers':
        inp.model = replace(inp.model, moe_layers=inp.model.moe_layers[:-1])
    elif corruption == 'bundle':
        layers = list(inp.model.moe_layers)
        layers[0] = MoELayerIR(0, 128, 8, 1, 128, (128,))
        inp.model = replace(inp.model, moe_layers=tuple(layers))
    elif corruption == 'components':
        layers = list(inp.model.moe_layers)
        old = layers[0]
        layers[0] = replace(old, component_bank_bytes=(old.routed_expert_bank_bytes,))
        inp.model = replace(inp.model, moe_layers=tuple(layers))
    elif corruption == 'runtime':
        manifest = json.loads(inp.stock.manifest_json)
        manifest['build']['flags'].append('GGML_NATIVE=ON')
        inp.stock.manifest_json = json.dumps(manifest)
    elif corruption == 'host':
        host['architecture'] = 'ARM64'
    elif corruption == 'source':
        source_reader = lambda *args: '0' * 40
    else:
        inp.hardware = {'supported_features': []}
    with pytest.raises(ValueError):
        api().EligibilityRegistry.with_builtins().attest(inp, host, tmp_path, source_reader=source_reader)


def test_q4_provider_does_not_admit_q6_model_bytes_under_q4_label(tmp_path):
    inp, host = fixtures()
    inp.model = replace(inp.model, identity=ArtifactIdentity('fixture.gguf', 22862575520,
        '089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba'))
    with pytest.raises(ValueError):
        api().EligibilityRegistry.with_builtins().attest(inp, host, tmp_path, source_reader=reader)


def test_q4_provider_rejects_self_consistent_unpinned_runtime(tmp_path):
    from expertflow.compiler.runner import RuntimeBinding
    from expertflow.compiler.preflight import file_sha256
    inp, host = fixtures()
    server = tmp_path / 'llama-server.exe'
    server.write_bytes(b'not the audited server')
    inp.stock = RuntimeBinding(
        ArtifactIdentity(str(server), server.stat().st_size, file_sha256(server)),
        Path('configs/compiler/runtime-stock.json').read_text(), (), None)
    inp.stock.verify()  # Self-consistency alone must not establish eligibility.
    with pytest.raises(ValueError, match='pinned runtime'):
        api().EligibilityRegistry.with_builtins().attest(inp, host, tmp_path, source_reader=reader)
