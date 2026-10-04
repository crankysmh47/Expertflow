"""Reviewed scheduling eligibility providers, separate from model adapters."""

import json
from pathlib import Path
import subprocess
from typing import Protocol

from .schema import canonical_payload, canonical_sha256


UPSTREAM = 'a7312ae94f801fc9c6786dc56e38df57b964f697'
SOURCE_OBJECTS = {
    'ggml/src/ggml-cpu/ggml-cpu.c': '2745a7dbb29ea12f45c1722f55bc913141669419',
    'ggml/src/ggml-cpu/ops.cpp': '7c54cb6f469da40913c7a9bd4beb51359a4b7e72',
    'ggml/src/ggml-cpu/repack.cpp': 'f18758f16bb62040333418bab06ce22efe826b20',
    'ggml/src/ggml-cuda/common.cuh': '290dc4aff259cb78a9a477dea27920fecf398c4a',
    'ggml/src/ggml-cuda/ggml-cuda.cu': '0878ab9c08afb783c6b5d459d4f9654617807d1d',
}

Q4_SOURCE_OBJECTS = {**SOURCE_OBJECTS,
    'ggml/src/ggml-cpu/quants.c': '5e36459f8cbc5900b375d2189414307393471a6b',
    'ggml/src/ggml-cpu/arch/x86/repack.cpp': 'af1cebad131d17118c26634faccb9a6e0c08a6f9',
    'ggml/src/ggml-cpu/arch/x86/quants.c': 'ea54cfe44ce403fe06c8b4aa10b5a882779e7e71',
}


def read_source_object(repository, revision, path):
    return subprocess.check_output(['git','-C',str(Path(repository).resolve()),
        'rev-parse',f'{revision}:{path}'],text=True,timeout=10).strip()


class SchedulingEligibilityProvider(Protocol):
    family: str
    quantization: str

    def attest(self, inputs, host, repository, *, source_reader=None) -> dict: ...


class Gemma4Q6SchedulingProvider:
    family = 'gemma4'
    quantization = 'Q6_K'

    def attest(self, inputs, host, repository, *, source_reader=None):
        model = inputs.model
        if (model.family,model.architecture,model.quantization,model.identity.sha256,model.identity.size_bytes) != (
                'gemma4','gemma4-moe','Q6_K',
                '089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba',22862575520):
            raise ValueError('model is outside audited scheduling scope')
        if host.get('architecture') not in ('AMD64','x86_64'):
            raise ValueError('host architecture is outside audited scheduling scope')
        manifest = json.loads(inputs.stock.manifest_json)
        if canonical_sha256(manifest) != '326daa6e17e1293f6b7c5c9f23e85868961241f4dd00a94a3eee54e25071f629':
            raise ValueError('runtime build is outside audited scheduling scope')
        if 'cuda_graphs' not in canonical_payload(inputs.hardware).get('supported_features',[]):
            raise ValueError('runtime/hardware graph capability unavailable')
        inputs.stock.verify()
        reader = source_reader or read_source_object
        actual = {path:reader(repository,UPSTREAM,path) for path in SOURCE_OBJECTS}
        if actual != SOURCE_OBJECTS:
            raise ValueError('audited scheduling source object mismatch')
        return {'provider_id':'gemma4-q6-x86-scheduling-v1',
            'model_ir_sha256':canonical_sha256(model),'runtime_sha256':inputs.stock.sha256,
            'hardware_sha256':canonical_sha256(inputs.hardware),'upstream_commit':UPSTREAM,
            'source_object_ids':actual,'allowed_controls':['threads','cuda_graphs'],
            'scope':'pinned Gemma4 Q6, unchanged arithmetic/placement/KV; exact native token guards required'}


class Gemma4Q4SchedulingProvider:
    family = 'gemma4'
    quantization = 'Q4_0'

    def attest(self, inputs, host, repository, *, source_reader=None):
        model = inputs.model
        if (model.family, model.architecture, model.quantization, model.identity.sha256,
                model.identity.size_bytes, model.expert_count, model.expert_top_k) != (
                'gemma4', 'gemma4-moe', 'Q4_0',
                '4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5',
                14439361440, 128, 8):
            raise ValueError('model is outside audited Q4 scheduling scope')
        components = (512, 142737408, 285474816)
        if tuple(layer.layer_id for layer in model.moe_layers) != tuple(range(30)) or any(
                (layer.expert_bundle_bytes, layer.routed_expert_bank_bytes,
                 layer.component_bank_bytes) != (3345412, 428212736, components)
                for layer in model.moe_layers):
            raise ValueError('Q4 inventory is outside audited scheduling scope')
        if host.get('architecture') not in ('AMD64', 'x86_64'):
            raise ValueError('host architecture is outside audited scheduling scope')
        if canonical_sha256(json.loads(inputs.stock.manifest_json)) != '326daa6e17e1293f6b7c5c9f23e85868961241f4dd00a94a3eee54e25071f629':
            raise ValueError('runtime build is outside audited scheduling scope')
        if 'cuda_graphs' not in canonical_payload(inputs.hardware).get('supported_features', []):
            raise ValueError('runtime/hardware graph capability unavailable')
        inputs.stock.verify()
        reader = source_reader or read_source_object
        actual = {path: reader(repository, UPSTREAM, path) for path in Q4_SOURCE_OBJECTS}
        if actual != Q4_SOURCE_OBJECTS:
            raise ValueError('audited Q4 scheduling source object mismatch')
        return {'provider_id': 'gemma4-q4-x86-scheduling-v1',
            'model_ir_sha256': canonical_sha256(model), 'runtime_sha256': inputs.stock.sha256,
            'hardware_sha256': canonical_sha256(inputs.hardware), 'upstream_commit': UPSTREAM,
            'source_object_ids': actual, 'allowed_controls': ['threads', 'cuda_graphs'],
            'scope': 'pinned Gemma4 Q4, unchanged AVX2 repack/arithmetic/placement/KV; exact own-reference native token guards required'}


class EligibilityRegistry:
    """Providers are trusted reviewed code; arbitrary receipt claims are not providers."""

    def __init__(self):
        self._providers = {}

    def register(self, provider: SchedulingEligibilityProvider):
        key = (provider.family,provider.quantization)
        if key in self._providers:
            raise ValueError(f'duplicate eligibility provider: {key}')
        self._providers[key] = provider

    def attest(self, inputs, host, repository, *, source_reader=None):
        key = (inputs.model.family,inputs.model.quantization)
        if key not in self._providers:
            raise ValueError(f'unsupported scheduling eligibility scope: {key}')
        proof = self._providers[key].attest(inputs,host,repository,source_reader=source_reader)
        expected = {'model_ir_sha256':canonical_sha256(inputs.model),'runtime_sha256':inputs.stock.sha256,
                    'hardware_sha256':canonical_sha256(inputs.hardware)}
        if any(proof.get(name) != value for name,value in expected.items()):
            raise ValueError('eligibility provider identity mismatch')
        if not proof.get('provider_id') or proof.get('allowed_controls') != ['threads','cuda_graphs']:
            raise ValueError('eligibility provider scope mismatch')
        return {**proof,'host_environment_sha256':canonical_sha256(host),
                'protocol_version':'stock-scheduling-eligibility-v1'}

    @classmethod
    def with_builtins(cls):
        registry = cls()
        registry.register(Gemma4Q6SchedulingProvider())
        registry.register(Gemma4Q4SchedulingProvider())
        return registry
