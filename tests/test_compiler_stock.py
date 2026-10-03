from dataclasses import replace

import pytest

from expertflow.compiler.stock import StockMeasurement, select_strongest_stock, stock_candidate_matrix
from expertflow.compiler.plan import PlanIdentities
from expertflow.compiler.schema import WorkloadIR, canonical_sha256


def identities():
    w = WorkloadIR('hello', 4096, 3)
    return PlanIdentities('a' * 64, 'b' * 64, canonical_sha256(w), 'c' * 64, w)


def result(candidate, tps=(23, 23, 23), **changes):
    return replace(StockMeasurement(candidate, ('a', 'b', 'c'), tps, True, 3000 << 20,
                                    True, 0, 'measured', canonical_sha256(candidate.identities)), **changes)


def test_bounded_matrix_and_measured_strongest_floor():
    candidates = stock_candidate_matrix(identities())
    assert {(c.settings.gpu_layers, c.settings.cpu_moe) for c in candidates} == {
        (g, cpu) for g in ('auto', 'all', 99) for cpu in (False, True)}
    assert len({c.candidate_id for c in candidates}) == 6
    measurements = [result(candidates[0]), result(candidates[1], (99, 99, 99), exact=False),
                    result(candidates[2], (100, 100, 100), peak_bytes=17000 << 20)]
    winner = select_strongest_stock(measurements, total_vram_mib=16311, reserve_mib=256)
    assert winner.candidate.candidate_id == candidates[0].candidate_id


@pytest.mark.parametrize('change', [
    {'measurement_ids': ('one',)}, {'decode_tps': (1, 20, 50)}, {'cleanup': False},
    {'exit_code': 1}, {'key_sha256': 'e' * 64}, {'status': 'estimated'},
])
def test_failures_missing_repeats_variance_or_wrong_key_cannot_win(change):
    candidate = stock_candidate_matrix(identities())[0]
    with pytest.raises(ValueError):
        select_strongest_stock([result(candidate, **change)], total_vram_mib=16311, reserve_mib=256)


def test_ties_stable_and_different_workloads_do_not_compete():
    a, b = stock_candidate_matrix(identities())[:2]
    assert select_strongest_stock([result(b), result(a)], total_vram_mib=16311, reserve_mib=256).candidate.candidate_id == min(a.candidate_id, b.candidate_id)
    other = replace(b, identities=replace(b.identities, runtime_sha256='f' * 64))
    with pytest.raises(ValueError, match='identity'):
        select_strongest_stock([result(a), result(other)], total_vram_mib=16311, reserve_mib=256)
