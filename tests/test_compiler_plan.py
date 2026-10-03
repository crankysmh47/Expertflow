from dataclasses import replace
import json

import pytest

from expertflow.compiler.plan import (
    CandidatePlan, CandidateStatus, ExecutionPlan, PlanIdentities, RuntimeSettings,
    StaticPlacement, load_execution_plan, seal_candidate, validate_execution_plan,
)
from expertflow.compiler.schema import WorkloadIR, canonical_payload, canonical_sha256


def identities_fixture():
    workload = WorkloadIR('hello', 4096, 512)
    return PlanIdentities('a' * 64, 'b' * 64, canonical_sha256(workload), 'c' * 64, workload)


def candidate_fixture(**changes):
    return replace(CandidatePlan(identities_fixture(), RuntimeSettings('auto', True),
                                 status=CandidateStatus.MEASURED,
                                 measurement_ids=('m-1', 'm-2', 'm-3')), **changes)


class VerifiedStore:
    """Protocol fixture; the real store must verify bytes, not caller flags."""
    def __init__(self, candidate):
        self.candidate = candidate

    def verify_measurement(self, mid):
        if mid not in self.candidate.measurement_ids:
            raise ValueError('missing evidence')
        return {'candidate_id': self.candidate.candidate_id,
                'identities': canonical_payload(self.candidate.identities),
                'settings_sha256': canonical_sha256(self.candidate.settings),
                'validations': {'exact_tokens': True, 'memory': True, 'cleanup': True},
                'exit_code': 0, 'generated_tokens_sha256': 'd' * 64,
                'measured': True, 'decode_tps': 23.0}


def test_seal_roundtrip_hash_and_full_workload(tmp_path):
    candidate = candidate_fixture()
    sealed = seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, None)
    assert sealed.plan_sha256 == canonical_sha256(sealed.without_hash())
    assert sealed.candidate.identities.workload.prompt == 'hello'
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(canonical_payload(sealed)), encoding='utf-8')
    assert load_execution_plan(path) == sealed
    assert validate_execution_plan(sealed, candidate.identities) is None


@pytest.mark.parametrize('status', [CandidateStatus.ESTIMATED, CandidateStatus.REJECTED,
                                   CandidateStatus.UNMEASURED, CandidateStatus.INVALID])
def test_unmeasured_cannot_seal(status):
    candidate = candidate_fixture(status=status)
    with pytest.raises(ValueError, match='measured passing candidate'):
        seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, None)


@pytest.mark.parametrize('change', [
    {'measurement_ids': ()}, {'measurement_ids': ('m-1',) * 3},
    {'rejection_reasons': ('numerical_path_change',)},
])
def test_missing_duplicate_or_rejected_evidence(change):
    candidate = candidate_fixture(**change)
    with pytest.raises(ValueError):
        seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, None)


@pytest.mark.parametrize('field,value', [('candidate_id', 'foreign'), ('exit_code', 1),
    ('generated_tokens_sha256', 'bad'), ('measured', False),
    ('validations', {'exact_tokens': True, 'memory': False, 'cleanup': True})])
def test_foreign_or_failed_store_evidence(field, value):
    candidate = candidate_fixture()
    store = VerifiedStore(candidate)
    original = store.verify_measurement
    store.verify_measurement = lambda mid: {**original(mid), field: value}
    with pytest.raises(ValueError):
        seal_candidate(candidate, store, candidate.identities, None)


def test_fabricated_ids_and_identity_mismatch():
    candidate = candidate_fixture()
    with pytest.raises(ValueError, match='missing evidence'):
        seal_candidate(replace(candidate, measurement_ids=('fake1', 'fake2', 'fake3')),
                       VerifiedStore(candidate), candidate.identities, None)
    with pytest.raises(ValueError, match='identity'):
        seal_candidate(candidate, VerifiedStore(candidate),
                       replace(candidate.identities, runtime_sha256='e' * 64), None)


def test_exact_static_numerical_path_change_cannot_seal_even_with_token_parity():
    candidate = candidate_fixture(settings=RuntimeSettings('auto', True, static=StaticPlacement((0, 1))))
    with pytest.raises(ValueError, match='numerical_path_change'):
        seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, None)


def test_tampered_hash_schema_and_fallback():
    candidate = candidate_fixture()
    sealed = seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, None)
    with pytest.raises(ValueError, match='hash'):
        validate_execution_plan(replace(sealed, plan_sha256='0' * 64))
    with pytest.raises(ValueError, match='schema'):
        replace(sealed, schema_version='2.0.0')
    foreign = candidate_fixture(identities=replace(candidate.identities, model_sha256='f' * 64))
    fallback = seal_candidate(foreign, VerifiedStore(foreign), foreign.identities, None)
    with pytest.raises(ValueError, match='fallback'):
        seal_candidate(candidate, VerifiedStore(candidate), candidate.identities, fallback)


def test_settings_and_immutable_identity_controls():
    with pytest.raises(ValueError):
        RuntimeSettings('auto', True, kv_type_k='q8_0')
    with pytest.raises(ValueError):
        StaticPlacement((0, 0))
    with pytest.raises(ValueError):
        StaticPlacement(tuple(range(13)))
    with pytest.raises(ValueError):
        replace(identities_fixture(), workload_sha256='e' * 64)
    candidate = candidate_fixture()
    assert candidate.candidate_id == replace(candidate, status=CandidateStatus.UNMEASURED,
                                             measurement_ids=()).candidate_id
    assert candidate.candidate_id != replace(candidate, settings=RuntimeSettings('auto', False)).candidate_id
