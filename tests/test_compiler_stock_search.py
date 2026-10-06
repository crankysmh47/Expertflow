"""Generic scheduling contracts; these fixtures are not live model evidence."""

from dataclasses import replace

import pytest

from expertflow.compiler.plan import CandidatePlan, PlanIdentities, RuntimeSettings
from expertflow.compiler.schema import WorkloadIR, canonical_sha256


def api():
    from expertflow.compiler import stock_search
    return stock_search


def candidate(threads=12):
    workload = WorkloadIR('hello', 4096, 512, threads=threads)
    identities = PlanIdentities('a'*64, 'b'*64, canonical_sha256(workload), 'c'*64, workload)
    return CandidatePlan(identities, RuntimeSettings(99, True))


def host(cores=8, logical=16):
    return {'cpu': [{'name': 'synthetic', 'cores': cores, 'logical_processors': logical}],
            'process_affinity_mask': (1 << logical)-1, 'system_affinity_mask': (1 << logical)-1,
            'power_policy': {'settings_sha256': 'd'*64}}


@pytest.mark.parametrize('cores,logical,threads,expected', [
    (8,16,12,(8,12,16)), (4,8,6,(4,6,8)), (16,32,24,(16,24,32)),
    (6,6,6,(6,)), (3,6,5,(3,4,5,6)),
])
def test_topology_anchors_are_deterministic_and_not_machine_specific(cores, logical, threads, expected):
    assert api().topology_anchors(host(cores,logical), threads) == expected


@pytest.mark.parametrize('changed', [
    {'cpu': []}, {'cpu': [{'cores':16,'logical_processors':8}]},
    {'process_affinity_mask': 255}, {'cpu': [{'cores':64,'logical_processors':128}]},
])
def test_ambiguous_or_unsupported_topology_fails_closed(changed):
    with pytest.raises(ValueError):
        api().topology_anchors({**host(), **changed}, 12)


def test_candidates_change_only_declared_controls_and_report_exclusions():
    base = candidate()
    space = api().scheduling_space(base, host(), excluded_threads={8:'prior rejected hypothesis'})
    assert {(c.identities.workload.threads,c.settings.cuda_graphs) for c in space.candidates} == {
        (12,'on'),(12,'off'),(16,'on'),(16,'off')}
    assert base.candidate_id in {c.candidate_id for c in space.candidates}
    assert space.excluded_threads == ((8,'prior rejected hypothesis'),)
    assert space.untested_threads == tuple(i for i in range(1,17) if i not in (12,16))
    fingerprints = {api().semantic_fingerprint(c) for c in space.candidates}
    assert fingerprints == {api().semantic_fingerprint(base)}
    assert len({c.candidate_id for c in space.candidates}) == 4


@pytest.mark.parametrize('field,value', [('prompt','different'),('predict_tokens',256),('context_size',8192)])
def test_semantic_changes_cannot_be_compared_as_tuning(field,value):
    base = candidate()
    workload = replace(base.identities.workload, **{field:value})
    changed = replace(base, identities=replace(base.identities, workload=workload,
        workload_sha256=canonical_sha256(workload)))
    assert api().semantic_fingerprint(changed) != api().semantic_fingerprint(base)


@pytest.mark.parametrize('field', ['model_sha256','hardware_sha256','runtime_sha256'])
def test_model_hardware_runtime_identity_invalidates_comparison(field):
    base = candidate()
    changed = replace(base, identities=replace(base.identities, **{field:'e'*64}))
    assert api().semantic_fingerprint(changed) != api().semantic_fingerprint(base)


def test_incumbent_cannot_be_excluded_or_static_candidate_admitted():
    from expertflow.compiler.plan import StaticPlacement
    with pytest.raises(ValueError):
        api().scheduling_space(candidate(), host(), excluded_threads={12:'invalid'})
    base = candidate()
    with pytest.raises(ValueError):
        api().scheduling_space(replace(base, settings=replace(base.settings,
            static=StaticPlacement((1,)))), host())


def test_screening_schedule_covers_each_candidate_in_each_fixed_block():
    schedule = api().screening_schedule(('a','b','c','d'))
    assert len(schedule) == 3
    assert all(set(block) == {'a','b','c','d'} and len(block) == 4 for block in schedule)
    assert schedule == api().screening_schedule(('d','c','b','a'))


def screen_rows(schedule, rates):
    return [{'block':block,'candidate_id':cid,'decode_tps':rates[block][cid]}
            for block,order in enumerate(schedule) for cid in order]


def test_screening_ranking_normalizes_temporal_block_changes():
    schedule = api().screening_schedule(('incumbent','challenger'))
    rows = screen_rows(schedule,[{'incumbent':20,'challenger':22},
                                 {'incumbent':25,'challenger':27.5},
                                 {'incumbent':15,'challenger':16.5}])
    ranked = api().rank_screening(rows,'incumbent',schedule)
    assert ranked[0]['candidate_id'] == 'challenger'
    assert ranked[0]['geometric_ratio'] == pytest.approx(1.1)


def test_screening_tie_prefers_incumbent_not_candidate_order():
    schedule = api().screening_schedule(('a','z'))
    rows = screen_rows(schedule,[{'a':20,'z':20}]*3)
    assert api().rank_screening(rows,'z',schedule)[0]['candidate_id'] == 'z'


def test_screening_reports_three_block_variation_as_descriptive_only():
    schedule = api().screening_schedule(('a','b'))
    rows = screen_rows(schedule,[{'a':20,'b':21},{'a':20,'b':22},{'a':20,'b':23}])
    winner = api().rank_screening(rows,'a',schedule)[0]
    assert winner['block_ratios'] == pytest.approx([1.05,1.1,1.15])
    assert winner['block_ratio_range'] == pytest.approx([1.05,1.15])
    assert winner['block_ratio_cv_pct'] > 0
    assert 'descriptive' in winner['uncertainty_scope'] and 'not confirmation' in winner['uncertainty_scope']


@pytest.mark.parametrize('corruption',['missing','duplicate','order','nonfinite','unknown'])
def test_partial_or_corrupt_screening_cannot_select_a_finalist(corruption):
    schedule = api().screening_schedule(('a','b'))
    rows = screen_rows(schedule,[{'a':20,'b':22}]*3)
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows[1] = dict(rows[0])
    elif corruption == 'order':
        rows[0],rows[1] = rows[1],rows[0]
    elif corruption == 'nonfinite':
        rows[0]['decode_tps'] = float('nan')
    else:
        rows[0]['candidate_id'] = 'unknown'
    with pytest.raises(ValueError):
        api().rank_screening(rows,'a',schedule)
