"""Immutable normalized compiler inputs and deterministic identities."""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import math
import re

from .reference import SCHEMA_VERSION, canonical_json


class ExactnessPolicy(str, Enum):
    EXACT = 'exact'
    APPROXIMATE = 'approximate'


class Objective(str, Enum):
    DECODE_TPS = 'decode_tps'


def require_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def require_number(value, name, minimum=0):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite and >= {minimum}')


def require_hash(value, name='sha256'):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA-256')


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')


def require_schema(value):
    if value != SCHEMA_VERSION:
        raise ValueError(f'unsupported schema: {value}')


def canonical_payload(value):
    """Copy to JSON primitives, rejecting hidden mutable/nonfinite inputs."""
    if isinstance(value, Enum):
        return canonical_payload(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: canonical_payload(getattr(value, f.name)) for f in fields(value)
                if not (f.metadata.get('omit_if_none') and getattr(value, f.name) is None)}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        require_number(abs(value), 'JSON number')
        return value
    if isinstance(value, (tuple, list)):
        return [canonical_payload(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: canonical_payload(item) for key, item in value.items()}
    raise ValueError(f'unsupported canonical type: {type(value).__name__}')


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(canonical_payload(value)).encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self):
        require_text(self.path, 'artifact path')
        require_int(self.size_bytes, 'artifact size')
        require_hash(self.sha256)


@dataclass(frozen=True, slots=True)
class MoELayerIR:
    layer_id: int
    expert_count: int
    expert_top_k: int
    expert_bundle_bytes: int
    routed_expert_bank_bytes: int
    component_bank_bytes: tuple[int, ...] = ()

    def __post_init__(self):
        require_int(self.layer_id, 'layer_id', 0)
        for name in ('expert_count', 'expert_top_k', 'expert_bundle_bytes', 'routed_expert_bank_bytes'):
            require_int(getattr(self, name), name)
        if self.expert_top_k > self.expert_count:
            raise ValueError('top_k exceeds expert_count')
        components = tuple(self.component_bank_bytes) or (self.routed_expert_bank_bytes,)
        for value in components:
            require_int(value, 'component bytes')
        if sum(components) != self.routed_expert_bank_bytes:
            raise ValueError('component bytes do not sum to expert bank')
        object.__setattr__(self, 'component_bank_bytes', components)


@dataclass(frozen=True, slots=True)
class ModelIR:
    schema_version: str
    identity: ArtifactIdentity
    family: str
    architecture: str
    quantization: str
    expert_count: int
    expert_top_k: int
    moe_layers: tuple[MoELayerIR, ...]
    kv_kind: str
    mtp_kind: str
    alignment_bytes: int = 256
    inventory_sha256: str | None = None

    def __post_init__(self):
        require_schema(self.schema_version)
        if not isinstance(self.identity, ArtifactIdentity):
            raise ValueError('invalid model identity')
        for name in ('family', 'architecture', 'quantization', 'kv_kind', 'mtp_kind'):
            require_text(getattr(self, name), name)
        for name in ('expert_count', 'expert_top_k', 'alignment_bytes'):
            require_int(getattr(self, name), name)
        if self.expert_top_k > self.expert_count:
            raise ValueError('top_k exceeds expert_count')
        if self.inventory_sha256 is not None:
            require_hash(self.inventory_sha256)
        if not self.moe_layers or not all(isinstance(x, MoELayerIR) for x in self.moe_layers):
            raise ValueError('model requires normalized MoE layers')
        layers = tuple(sorted(self.moe_layers, key=lambda x: x.layer_id))
        if len({x.layer_id for x in layers}) != len(layers):
            raise ValueError('duplicate layer IDs')
        if any((x.expert_count, x.expert_top_k) != (self.expert_count, self.expert_top_k) for x in layers):
            raise ValueError('inconsistent layer expert counts/top_k')
        object.__setattr__(self, 'moe_layers', layers)


@dataclass(frozen=True, slots=True)
class HardwareIR:
    gpu_uuid: str
    gpu_name: str
    compute_capability: str
    total_vram_bytes: int
    usable_vram_bytes: int
    driver_version: str
    cuda_version: str
    cuda_runtime_sha256: str
    minimum_reserve_bytes: int = 256 << 20
    supported_features: tuple[str, ...] = ('cuda_graphs',)
    calibration_sha256: str | None = None

    def __post_init__(self):
        for name in ('gpu_uuid', 'gpu_name', 'compute_capability', 'driver_version', 'cuda_version'):
            require_text(getattr(self, name), name)
        for name in ('total_vram_bytes', 'usable_vram_bytes', 'minimum_reserve_bytes'):
            require_int(getattr(self, name), name)
        if self.usable_vram_bytes + self.minimum_reserve_bytes > self.total_vram_bytes:
            raise ValueError('usable VRAM exceeds reserved allocation frontier')
        require_hash(self.cuda_runtime_sha256)
        if self.calibration_sha256 is not None:
            require_hash(self.calibration_sha256)
        if not all(isinstance(x, str) and x for x in self.supported_features):
            raise ValueError('invalid hardware features')
        object.__setattr__(self, 'supported_features', tuple(sorted(set(self.supported_features))))


@dataclass(frozen=True, slots=True)
class WorkloadIR:
    prompt: str
    context_size: int
    predict_tokens: int
    schema_version: str = SCHEMA_VERSION
    objective: Objective = Objective.DECODE_TPS
    policy: ExactnessPolicy = ExactnessPolicy.EXACT
    runtime_interface: str = 'server_completion'
    concurrency: int = 1
    threads: int = 12
    seed: int = 42
    temperature: float = 0.0
    ignore_eos: bool = True
    cache_prompt: bool = False
    kv_type_k: str = 'f16'
    kv_type_v: str = 'f16'
    cuda_graphs: str = 'on'
    batch_size: int = 2048
    microbatch_size: int = 512
    warmup_runs: int = 1
    measured_runs: int = 3
    minimum_vram_reserve_mib: int = 256
    maximum_cv_pct: float = 10.0
    confirmation_pairs: int = 10
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20261003
    replay_tolerance_pct: float = 2.0
    health_timeout_seconds: int = 180
    completion_timeout_seconds: int = 300
    approximate_quality_budget: float | None = None

    def __post_init__(self):
        require_schema(self.schema_version)
        try:
            object.__setattr__(self, 'objective', Objective(self.objective))
            object.__setattr__(self, 'policy', ExactnessPolicy(self.policy))
        except (ValueError, TypeError) as error:
            raise ValueError('unsupported objective/policy') from error
        require_text(self.prompt, 'prompt')
        for name in ('context_size', 'predict_tokens', 'threads', 'batch_size', 'microbatch_size',
                     'warmup_runs', 'measured_runs', 'minimum_vram_reserve_mib', 'confirmation_pairs',
                     'bootstrap_samples', 'health_timeout_seconds', 'completion_timeout_seconds'):
            require_int(getattr(self, name), name)
        for name in ('seed', 'bootstrap_seed'):
            require_int(getattr(self, name), name, 0)
        for name in ('temperature', 'maximum_cv_pct', 'replay_tolerance_pct'):
            require_number(getattr(self, name), name)
        if type(self.concurrency) is not int or self.concurrency != 1:
            raise ValueError('only concurrency one is supported')
        if self.runtime_interface != 'server_completion':
            raise ValueError('product requires server_completion')
        if self.microbatch_size > self.batch_size:
            raise ValueError('microbatch exceeds batch')
        if self.cuda_graphs not in {'on', 'off'}:
            raise ValueError('invalid CUDA graph setting')
        if type(self.ignore_eos) is not bool or type(self.cache_prompt) is not bool:
            raise ValueError('invalid boolean workload control')
        if self.policy is ExactnessPolicy.EXACT and (
            self.temperature != 0 or not self.ignore_eos or self.cache_prompt
            or self.kv_type_k != 'f16' or self.kv_type_v != 'f16'
            or self.approximate_quality_budget is not None
        ):
            raise ValueError('approximate settings in exact workload')
        if self.policy is ExactnessPolicy.APPROXIMATE:
            require_number(self.approximate_quality_budget, 'approximate_quality_budget', 0.000001)

    @classmethod
    def from_reference(cls, reference):
        payload = reference.payload
        payload.pop('prompt_file', None)
        return cls(**payload)
