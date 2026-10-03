"""Candidate contracts and evidence-backed, hashed execution plans."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from .reference import SCHEMA_VERSION, read_json
from .schema import (
    ExactnessPolicy, WorkloadIR, canonical_payload, canonical_sha256,
    require_hash, require_int, require_number, require_schema, require_text,
)


class CandidateStatus(str, Enum):
    ESTIMATED = 'estimated'
    UNMEASURED = 'unmeasured'
    INVALID = 'invalid'
    REJECTED = 'rejected'
    MEASURED = 'measured'


@dataclass(frozen=True, slots=True)
class StaticPlacement:
    layer_ids: tuple[int, ...]
    precompute: bool = True

    def __post_init__(self):
        if not self.layer_ids or len(self.layer_ids) > 12:
            raise ValueError('static placement requires 1–12 layers')
        for layer in self.layer_ids:
            require_int(layer, 'static layer', 0)
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('duplicate static layers')
        if self.precompute is not True:
            raise ValueError('static precompute must be enabled')
        object.__setattr__(self, 'layer_ids', tuple(sorted(self.layer_ids)))


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    gpu_layers: str | int
    cpu_moe: bool
    cuda_graphs: str = 'on'
    kv_type_k: str = 'f16'
    kv_type_v: str = 'f16'
    batch_size: int = 2048
    microbatch_size: int = 512
    static: StaticPlacement | None = None

    def __post_init__(self):
        if not ((type(self.gpu_layers) is int and self.gpu_layers >= 0)
                or self.gpu_layers in ('auto', 'all')):
            raise ValueError('invalid GPU layers')
        if type(self.cpu_moe) is not bool or self.cuda_graphs not in {'on', 'off'}:
            raise ValueError('invalid runtime controls')
        if self.kv_type_k != 'f16' or self.kv_type_v != 'f16':
            raise ValueError('Phase 3 requires exact F16 KV')
        require_int(self.batch_size, 'batch_size')
        require_int(self.microbatch_size, 'microbatch_size')
        if self.microbatch_size > self.batch_size:
            raise ValueError('microbatch exceeds batch')
        if self.static is not None and not isinstance(self.static, StaticPlacement):
            raise ValueError('invalid static placement')
        if self.static is not None and not self.cpu_moe:
            raise ValueError('static placement requires CPU-MoE baseline')


@dataclass(frozen=True, slots=True)
class PlanIdentities:
    model_sha256: str
    hardware_sha256: str
    workload_sha256: str
    runtime_sha256: str
    workload: WorkloadIR

    def __post_init__(self):
        for name in ('model_sha256', 'hardware_sha256', 'workload_sha256', 'runtime_sha256'):
            require_hash(getattr(self, name), name)
        if not isinstance(self.workload, WorkloadIR) or canonical_sha256(self.workload) != self.workload_sha256:
            raise ValueError('workload identity mismatch')


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    identities: PlanIdentities
    settings: RuntimeSettings
    status: CandidateStatus = CandidateStatus.UNMEASURED
    estimated_decode_tps: float | None = None
    measurement_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    validation: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self):
        if not isinstance(self.identities, PlanIdentities) or not isinstance(self.settings, RuntimeSettings):
            raise ValueError('invalid candidate inputs')
        object.__setattr__(self, 'status', CandidateStatus(self.status))
        if self.estimated_decode_tps is not None:
            require_number(self.estimated_decode_tps, 'estimated TPS', 0.000001)
        for name in ('measurement_ids', 'rejection_reasons'):
            items = tuple(getattr(self, name))
            for item in items:
                require_text(item, name)
            object.__setattr__(self, name, items)
        validation = tuple(sorted(tuple(x) for x in self.validation))
        if any(len(x) != 2 or not isinstance(x[0], str) or type(x[1]) is not bool for x in validation):
            raise ValueError('invalid candidate validation')
        object.__setattr__(self, 'validation', validation)
        w, s = self.identities.workload, self.settings
        if (s.cuda_graphs, s.kv_type_k, s.kv_type_v, s.batch_size, s.microbatch_size) != (
            w.cuda_graphs, w.kv_type_k, w.kv_type_v, w.batch_size, w.microbatch_size
        ):
            raise ValueError('settings/workload mismatch')

    @property
    def candidate_id(self):
        return canonical_sha256({'identities': self.identities, 'settings': self.settings})


class VerifiedEvidence(Protocol):
    def verify_measurement(self, measurement_id: str) -> dict: ...


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    schema_version: str
    compiler_version: str
    launcher_abi: str
    candidate: CandidatePlan
    fallback: 'ExecutionPlan | None'
    sealed_at: str
    plan_sha256: str

    def __post_init__(self):
        require_schema(self.schema_version)
        require_schema(self.launcher_abi)
        require_text(self.compiler_version, 'compiler_version')
        require_text(self.sealed_at, 'sealed_at')
        require_hash(self.plan_sha256, 'plan hash')

    def without_hash(self):
        payload = canonical_payload(self)
        payload.pop('plan_sha256')
        return payload


def _validate_fallback(candidate, fallback):
    if fallback is None:
        if candidate.settings.static is not None:
            raise ValueError('static candidate requires measured fallback')
        return
    if not isinstance(fallback, ExecutionPlan):
        raise ValueError('unresolvable fallback')
    validate_execution_plan(fallback)
    a, b = candidate.identities, fallback.candidate.identities
    if (a.model_sha256, a.hardware_sha256, a.workload_sha256) != (
        b.model_sha256, b.hardware_sha256, b.workload_sha256
    ) or fallback.candidate.settings.static is not None:
        raise ValueError('incompatible fallback')


def _validate_candidate(candidate):
    if candidate.status is not CandidateStatus.MEASURED or candidate.rejection_reasons:
        raise ValueError('requires a measured passing candidate')
    if len(candidate.measurement_ids) < candidate.identities.workload.measured_runs:
        raise ValueError('missing measured evidence')
    if len(set(candidate.measurement_ids)) != len(candidate.measurement_ids):
        raise ValueError('duplicate measurement IDs')
    if candidate.identities.workload.policy is not ExactnessPolicy.EXACT:
        raise ValueError('Phase 3 only seals exact candidates')
    if candidate.settings.static is not None:
        # This runtime has no bitwise-preservation contract for CPU -> CUDA MoE.
        raise ValueError('numerical_path_change')


def seal_candidate(candidate, store: VerifiedEvidence, identities, fallback):
    _validate_candidate(candidate)
    if candidate.identities != identities:
        raise ValueError('candidate identity mismatch')
    _validate_fallback(candidate, fallback)
    token_hashes = set()
    for mid in candidate.measurement_ids:
        row = store.verify_measurement(mid)
        if row.get('candidate_id') != candidate.candidate_id or row.get('identities') != canonical_payload(identities):
            raise ValueError('foreign candidate evidence or identity mismatch')
        if row.get('settings_sha256') != canonical_sha256(candidate.settings):
            raise ValueError('measurement settings mismatch')
        if row.get('exit_code') != 0 or row.get('measured') is not True:
            raise ValueError('measurement failed or is warmup')
        if any(row.get('validations', {}).get(name) is not True for name in ('exact_tokens', 'memory', 'cleanup')):
            raise ValueError('measurement validation failed')
        require_hash(row.get('generated_tokens_sha256'), 'generated token hash')
        require_number(row.get('decode_tps'), 'measured TPS', 0.000001)
        token_hashes.add(row['generated_tokens_sha256'])
    if len(token_hashes) != 1:
        raise ValueError('unstable generated tokens')
    plan = ExecutionPlan(SCHEMA_VERSION, '0.1.0', SCHEMA_VERSION, candidate, fallback,
                         datetime.now(timezone.utc).isoformat(), '0' * 64)
    return replace(plan, plan_sha256=canonical_sha256(plan.without_hash()))


def validate_execution_plan(plan, identities=None, store=None):
    if not isinstance(plan, ExecutionPlan):
        raise ValueError('expected ExecutionPlan')
    require_schema(plan.schema_version)
    require_schema(plan.launcher_abi)
    if canonical_sha256(plan.without_hash()) != plan.plan_sha256:
        raise ValueError('plan hash mismatch')
    _validate_candidate(plan.candidate)
    if identities is not None and plan.candidate.identities != identities:
        raise ValueError('execution identity mismatch')
    _validate_fallback(plan.candidate, plan.fallback)
    if store is not None:
        seal_candidate(plan.candidate, store, plan.candidate.identities, plan.fallback)


def _decode_plan(payload):
    try:
        value = dict(payload)
        c = dict(value['candidate'])
        i = dict(c['identities'])
        i['workload'] = WorkloadIR(**i['workload'])
        c['identities'] = PlanIdentities(**i)
        s = dict(c['settings'])
        if s['static'] is not None:
            s['static'] = StaticPlacement(**s['static'])
        c['settings'] = RuntimeSettings(**s)
        value['candidate'] = CandidatePlan(**c)
        if value['fallback'] is not None:
            value['fallback'] = _decode_plan(value['fallback'])
        return ExecutionPlan(**value)
    except (TypeError, KeyError, RecursionError) as error:
        raise ValueError(f'invalid execution plan: {error}') from error


def load_execution_plan(path: Path, *, identities=None, store=None):
    plan = _decode_plan(read_json(Path(path)))
    validate_execution_plan(plan, identities, store)
    return plan
