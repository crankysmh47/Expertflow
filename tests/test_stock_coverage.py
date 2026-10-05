"""Registration guards; these are not wider native coverage results."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from expertflow.compiler.schema import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT/'configs/compiler/stock-coverage-20261005.json'


def registration():
    return json.loads(REGISTRATION.read_text())


def rehash(data):
    payload = dict(data)
    payload.pop('registration_sha256', None)
    data['registration_sha256'] = canonical_sha256(payload)
    return data


def test_four_cases_have_fixed_independent_budgets_and_no_native_result():
    from expertflow.stock.coverage import verify_registration
    data = verify_registration(registration(), ROOT)
    assert data['status'] == 'REGISTERED-NOT-RUN'
    assert data['maximum_native_processes'] == sum(c['maximum_native_processes'] for c in data['cases']) == 428
    assert [c['case_id'] for c in data['cases']] == [
        'gemma4-q4-prose', 'gemma4-q4-code', 'granite-q6-prose', 'granite-q6-code']
    assert all(c['status'] == 'NOT-RUN' for c in data['cases'])


@pytest.mark.parametrize('mutation', ['budget', 'gate', 'family', 'completed', 'path', 'controls', 'schedule', 'host'])
def test_rehashed_registration_cannot_loosen_scope_or_claim_results(mutation):
    from expertflow.stock.coverage import verify_registration
    data = deepcopy(registration())
    if mutation == 'budget': data['cases'][0]['maximum_native_processes'] = 108
    if mutation == 'gate': data['gates']['minimum_defaults_gain_pct'] = 0
    if mutation == 'family': data['cases'][0]['family'] = 'unknown'
    if mutation == 'completed': data['cases'][0]['status'] = 'PASS-STOCK-UTILITY'
    if mutation == 'path': data['input_files']['../outside.json'] = 'a'*64
    if mutation == 'controls': data['cases'][0]['candidate_controls'][0]['threads'] = 4
    if mutation == 'schedule': data['cases'][0]['screening_schedule'][0].reverse()
    if mutation == 'host': data['host_environment']['cpu'][0]['cores'] = 16
    with pytest.raises(ValueError):
        verify_registration(rehash(data), ROOT)


def test_changed_workload_pin_is_rejected_before_any_weights_read():
    from expertflow.stock.coverage import verify_registration
    data = registration()
    data['input_files'][data['cases'][0]['workload']] = 'a'*64
    with pytest.raises(ValueError, match='input identity'):
        verify_registration(rehash(data), ROOT)


def test_public_inspection_is_explicitly_registered_not_executed(capsys):
    from expertflow.cli.main import main
    assert main(['stock','coverage','inspect','--registration',str(REGISTRATION)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'REGISTERED-NOT-RUN' and result['native_calls'] == 0
    assert result['execution_ready'] is False and result['utility_gain_established'] is False
    assert result['weight_hashes_verified'] is False


def test_neutral_default_optimal_and_incomplete_controls_do_not_claim_gain():
    from scripts.benchmark_compiler_stock_utility import evaluate_utility
    from expertflow.compiler.stock_search import rank_screening, screening_schedule
    schedule = screening_schedule(('defaults', 'alternate'))
    rows = [{'block':block, 'candidate_id':cid, 'decode_tps':20}
        for block, ids in enumerate(schedule) for cid in ids]
    assert rank_screening(rows,'defaults',schedule)[0]['candidate_id'] == 'defaults'
    assert evaluate_utility(*[[20]*10]*4,automatic_evaluations=18,manual_evaluations=18)['status'] == 'NO-UTILITY-GAIN'
    with pytest.raises(ValueError, match='incomplete'):
        rank_screening(rows[:-1],'defaults',schedule)
