"""Eligibility fixtures exercise scope checks, not native arithmetic evidence."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from expertflow.compiler.schema import ArtifactIdentity, canonical_sha256
from test_compiler_schema import model_fixture


PROOF = json.loads(Path('docs/evidence/stock-discovery-20261004/scheduling-source-proof.json').read_text())


def api():
    from expertflow.compiler import stock_eligibility
    return stock_eligibility


def fixtures():
    model = replace(model_fixture(),identity=ArtifactIdentity('fixture.gguf',
        PROOF['model_size_bytes'],PROOF['model_artifact_sha256']))
    manifest = Path('configs/compiler/runtime-stock.json').read_text()
    binding = SimpleNamespace(manifest_json=manifest, sha256='a'*64, verify=lambda: None,
        verify_manifest_bindings=lambda: None)
    hardware = {'supported_features':['cuda_graphs']}
    inputs = SimpleNamespace(model=model,stock=binding,hardware=hardware)
    host = {'architecture':'AMD64'}
    return inputs,host


def source_reader(repository, revision, path):
    assert revision == PROOF['upstream_commit']
    return PROOF['source_object_ids'][path]


def test_audited_scope_attests_exact_model_runtime_and_source_objects(tmp_path):
    inputs,host = fixtures()
    result = api().EligibilityRegistry.with_builtins().attest(inputs,host,tmp_path,
        source_reader=source_reader)
    assert result['provider_id'] == PROOF['provider_id']
    assert result['model_ir_sha256'] == canonical_sha256(inputs.model)
    assert result['runtime_sha256'] == inputs.stock.sha256
    assert result['source_object_ids'] == PROOF['source_object_ids']
    assert result['allowed_controls'] == ['threads','cuda_graphs']


@pytest.mark.parametrize('corruption',['family','quantization','weights','runtime','architecture','source'])
def test_unreviewed_scope_is_rejected_before_native_launch(tmp_path,corruption):
    inputs,host = fixtures()
    reader = source_reader
    if corruption == 'family':
        inputs.model = replace(inputs.model,family='mixtral',architecture='mixtral')
    elif corruption == 'quantization':
        inputs.model = replace(inputs.model,quantization='Q4_0')
    elif corruption == 'weights':
        inputs.model = replace(inputs.model,identity=replace(inputs.model.identity,sha256='f'*64))
    elif corruption == 'runtime':
        manifest = json.loads(inputs.stock.manifest_json)
        manifest['build']['flags'].append('GGML_NATIVE=ON')
        inputs.stock.manifest_json = json.dumps(manifest)
    elif corruption == 'architecture':
        host['architecture'] = 'ARM64'
    else:
        reader = lambda *args: '0'*40
    with pytest.raises(ValueError):
        api().EligibilityRegistry.with_builtins().attest(inputs,host,tmp_path,source_reader=reader)


def test_second_family_provider_contract_is_explicit_and_identity_bound(tmp_path):
    inputs,host = fixtures()
    inputs.model = replace(inputs.model,family='synthetic-moe',architecture='synthetic')
    class SyntheticProvider:
        family = 'synthetic-moe'
        quantization = 'Q6_K'
        def attest(self, inputs, host, repository, *, source_reader=None):
            return {'provider_id':'synthetic-contract-only',
                'model_ir_sha256':canonical_sha256(inputs.model),
                'runtime_sha256':inputs.stock.sha256,'hardware_sha256':canonical_sha256(inputs.hardware),
                'allowed_controls':['threads','cuda_graphs']}
    registry = api().EligibilityRegistry.with_builtins()
    registry.register(SyntheticProvider())
    result = registry.attest(inputs,host,tmp_path)
    assert result['provider_id'] == 'synthetic-contract-only'
    with pytest.raises(ValueError,match='duplicate'):
        registry.register(SyntheticProvider())


def test_provider_cannot_attest_different_model_identity(tmp_path):
    inputs,host = fixtures()
    class WrongProvider:
        family = 'gemma4'
        quantization = 'Q6_K'
        def attest(self,*args,**kwargs):
            return {'model_ir_sha256':'wrong','runtime_sha256':'wrong','hardware_sha256':'wrong',
                    'allowed_controls':['threads','cuda_graphs']}
    registry = api().EligibilityRegistry()
    registry.register(WrongProvider())
    with pytest.raises(ValueError,match='identity'):
        registry.attest(inputs,host,tmp_path)
