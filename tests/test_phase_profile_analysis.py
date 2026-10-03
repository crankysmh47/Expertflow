import copy
import pytest


def fixture():
    return {'schema_version':'2.0.0','diagnostic_synchronization':True,
            'phases':[{'phase':'prefill','tokens':39,'graph_calls':1,'split_count':1},
                      {'phase':'decode','tokens':511,'graph_calls':511,'split_count':2}],
            'records':[{'phase':'decode','split_id':0,'backend':'CPU','first_node':'ffn_moe_gate_up-0',
                        'calls':511,'input_boundary_us':10,'compute_submit_us':80,'completion_us':0,'total_us':90},
                       {'phase':'decode','split_id':1,'backend':'CUDA0','first_node':'other',
                        'calls':511,'input_boundary_us':2,'compute_submit_us':1,'completion_us':7,'total_us':10},
                       {'phase':'prefill','split_id':0,'backend':'CPU','first_node':'ffn_moe_gate_up-0',
                        'calls':1,'input_boundary_us':1000,'compute_submit_us':8000,'completion_us':0,'total_us':9000}]}


def analyze(value):
    from expertflow.analysis.phase_profile import analyze_phase_profile
    return analyze_phase_profile(value,prompt_tokens=39,generated_tokens=512,expert_layers={0})


def test_decode_breakdown_excludes_prefill_and_uses_explicit_phase():
    result=analyze(fixture())
    assert result['decode_total_us']==100
    assert result['cpu_compute_us']==80 and result['input_boundary_us']==12
    assert result['cuda_completion_wait_us']==7
    assert result['pure_transfer_us'] is None


@pytest.mark.parametrize('mutation',[
    lambda d:d['phases'][1].update(tokens=512),
    lambda d:d['phases'][0].update(tokens=38),
    lambda d:d['records'][0].update(phase='mixed'),
    lambda d:d['records'][0].update(calls=510),
    lambda d:d['records'][0].update(total_us=91),
    lambda d:d['records'][0].update(compute_submit_us=-1),
    lambda d:d['records'].append(copy.deepcopy(d['records'][0])),
    lambda d:d.update(schema_version='1.0.0'),
    lambda d:d.update(diagnostic_synchronization=False),
    lambda d:d['records'].pop(1),
    lambda d:d['records'].pop(2),
    lambda d:d['phases'].append({'phase':'mixed','tokens':1,'graph_calls':0,'split_count':0}),
])
def test_invalid_phase_or_accounting_fails_closed(mutation):
    value=fixture();mutation(value)
    with pytest.raises(ValueError):analyze(value)
