from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_compiler_granite_adapter import normalize

AUDIT = json.loads(Path('docs/evidence/compiler-granite-20261004/scheduling-source-audit.json').read_text())


def inputs():
    stock = SimpleNamespace(manifest_json=Path('configs/compiler/runtime-stock.json').read_text(),
        sha256='a'*64, verify_manifest_bindings=lambda: None)
    return SimpleNamespace(model=normalize(), stock=stock, hardware={'supported_features':['cuda_graphs']})


def reader(repository, revision, path):
    assert revision == AUDIT['upstream_commit']
    return AUDIT['source_object_ids'][path]


def test_real_family_has_separate_gpu_baseline_proof(tmp_path):
    from expertflow.compiler.stock_eligibility import EligibilityRegistry
    proof = EligibilityRegistry.with_builtins().attest(inputs(), {'architecture':'AMD64'}, tmp_path, source_reader=reader)
    assert proof['provider_id'] == AUDIT['proposed_provider_id']
    assert proof['source_object_ids'] == AUDIT['source_object_ids']
    assert proof['baseline_cpu_moe'] is False
    assert proof['allowed_controls'] == ['threads','cuda_graphs']


@pytest.mark.parametrize('mutation', ['weights','size','architecture','quantization','inventory','layers','kv','runtime','host','source','capability','binary'])
def test_granite_provider_fails_closed(tmp_path, mutation):
    from expertflow.compiler.stock_eligibility import EligibilityRegistry
    inp, host, read = inputs(), {'architecture':'AMD64'}, reader
    if mutation == 'weights': inp.model=replace(inp.model,identity=replace(inp.model.identity,sha256='f'*64))
    if mutation == 'size': inp.model=replace(inp.model,identity=replace(inp.model.identity,size_bytes=1))
    if mutation == 'architecture': inp.model=replace(inp.model,architecture='gemma4-moe')
    if mutation == 'quantization': inp.model=replace(inp.model,quantization='Q4_0')
    if mutation == 'inventory': inp.model=replace(inp.model,inventory_sha256='f'*64)
    if mutation == 'layers': inp.model=replace(inp.model,moe_layers=inp.model.moe_layers[:-1])
    if mutation == 'kv': inp.model=replace(inp.model,kv_kind='approximate')
    if mutation == 'runtime': inp.stock.manifest_json='{}'
    if mutation == 'host': host['architecture']='ARM64'
    if mutation == 'source': read=lambda *args:'0'*40
    if mutation == 'capability': inp.hardware={'supported_features':[]}
    if mutation == 'binary':
        def invalid(): raise ValueError('binary identity mismatch')
        inp.stock.verify_manifest_bindings=invalid
    with pytest.raises(ValueError):
        EligibilityRegistry.with_builtins().attest(inp, host, tmp_path, source_reader=read)
