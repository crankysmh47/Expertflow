import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import stock_product_status as product


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(point, interval):
    return {'gain': {'geometric_change_pct': point, 'ci95_pct': interval},
            'manual_equivalence': {'ci90_pct': [-0.5, 0.5]},
            'automatic_evaluations': 18, 'manual_evaluations': 18}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = tmp_path/'source.py'
    source.write_text('immutable\n', encoding='utf-8')
    frozen = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    q6root, widerroot = tmp_path/'q6', tmp_path/'wider'
    transfer = q6root/'transfer/utility/report.json'
    transfer_sha = save(transfer, {'status': 'PASS-STOCK-UTILITY-PRODUCT',
        'automatic_id': 'tuned', 'manifest': {'default_id': 'default'},
        'statistics': comparison(9.38, [7.92, 10.61])})
    q6 = q6root/'report.json'
    q6sha = save(q6, {'status': 'PASS-STOCK-REPEATABILITY-TRANSFER',
        'attempts': [{'status': 'measured'}]*148,
        'manifest': {'source_files': frozen, 'experiment_root': str(q6root)},
        'transfer': {'report_sha256': transfer_sha}})
    cases = [{'case_id': name, 'status': 'NO-UTILITY-GAIN', 'native_processes': 86,
              'selected_default': name.startswith('granite'),
              'statistics': comparison(point, interval)}
             for name, point, interval in (
                 ('gemma4-q4-prose', 1.03, [0.88, 1.19]),
                 ('gemma4-q4-code', 0.78, [0.62, 0.94]),
                 ('granite-q6-prose', -0.02, [-0.15, 0.09]),
                 ('granite-q6-code', -0.23, [-0.39, -0.08]))]
    wider = widerroot/'report.json'
    wider_sha = save(wider, {'status': 'COMPLETE-STOCK-COVERAGE', 'attempts': 344,
        'native_processes': 344, 'manifest': {'source_files': frozen,
            'experiment_root': str(widerroot), 'registration': str(tmp_path/'registration.json')}, 'cases': cases})
    catalogue = tmp_path/'catalogue.json'
    save(catalogue, {'schema_version': 1, 'studies': {
        'q6': {'report': str(q6), 'sha256': q6sha},
        'wider': {'report': str(wider), 'sha256': wider_sha}}})
    monkeypatch.setattr(product, 'CATALOGUE_SHA256', hashlib.sha256(catalogue.read_bytes()).hexdigest(), raising=False)
    return tmp_path, catalogue, q6, wider


def mutate(evidence, study, change):
    _, catalogue, q6, wider = evidence
    path = q6 if study == 'q6' else wider
    data = json.loads(path.read_bytes())
    change(data)
    digest = save(path, data)
    catalog = json.loads(catalogue.read_bytes())
    catalog['studies'][study]['sha256'] = digest
    save(catalogue, catalog)
    # Rebind only trusted fixture bytes to exercise checks beneath the pin.
    product.CATALOGUE_SHA256 = hashlib.sha256(catalogue.read_bytes()).hexdigest()


def native_starts(evidence):
    for root, count in ((evidence[2].parent, 148), (evidence[3].parent, 344)):
        for index in range(count):
            save(root/f'raw/{index}/run-start.json', {'index': index})


def successful_validator(command, **kwargs):
    workflow = command[command.index('stock')+3]
    status = ('PASS-STOCK-REPEATABILITY-TRANSFER' if workflow == 'repeatability'
              else 'COMPLETE-STOCK-COVERAGE')
    if workflow == 'repeatability':
        report = str(Path(command[command.index('--output-dir')+1])/'report.json')
    else:
        report = str(Path(command[command.index('--registration')+1]).parent/'wider/report.json')
    return subprocess.CompletedProcess(command, 0,
        json.dumps({'status': status, 'report': report, 'decision': {'evidence_verified': True}}), '')


def test_separate_negative_results_are_preserved(evidence):
    result = product.snapshot(*evidence[:2])
    assert result['fresh_validation'] is False
    assert len(result['cases']) == 5
    assert result['cases'][0]['gain_pct'] == 9.38
    assert all(case['status'] == 'NO-UTILITY-GAIN' for case in result['cases'][1:])
    assert [case['selected_default'] for case in result['cases'][1:]] == [False, False, True, True]
    assert result['global_optimum_established'] is False


def test_changed_report_fails_before_display(evidence):
    evidence[3].write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        product.snapshot(*evidence[:2])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), '9.38', True])
def test_nonfinite_or_nonnumeric_gain_is_rejected(evidence, bad):
    mutate(evidence, 'wider', lambda data: data['cases'][0]['statistics']['gain'].update(geometric_change_pct=bad))
    with pytest.raises(ValueError, match='finite'):
        product.snapshot(*evidence[:2])


def test_missing_case_cannot_be_hidden_by_complete_status(evidence):
    mutate(evidence, 'wider', lambda data: data['cases'].pop())
    with pytest.raises(ValueError, match='cases'):
        product.snapshot(*evidence[:2])


def test_wrong_outer_status_is_not_accepted(evidence):
    mutate(evidence, 'q6', lambda data: data.update(status='IDENTITY-STOP'))
    with pytest.raises(ValueError, match='status'):
        product.snapshot(*evidence[:2])


def test_reversed_interval_is_rejected(evidence):
    mutate(evidence, 'wider', lambda data: data['cases'][0]['statistics']['gain'].update(ci95_pct=[2, 1]))
    with pytest.raises(ValueError, match='interval'):
        product.snapshot(*evidence[:2])


def test_transfer_report_is_bound_to_parent_digest(evidence):
    path = evidence[2].parent/'transfer/utility/report.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        product.snapshot(*evidence[:2])


def test_verify_delegates_only_validate_and_keeps_logs(evidence, monkeypatch):
    native_starts(evidence)
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return successful_validator(command, **kwargs)
    monkeypatch.setattr(product.subprocess, 'run', launch)
    result = product.verify(*evidence[:2], evidence[0]/'verification')
    assert result['fresh_validation'] is True
    assert result['additional_native_calls'] == 0
    assert len(commands) == 2 and all('validate' in c for c in commands)
    assert all('run' not in c and 'execute' not in c for c in commands)
    assert (evidence[0]/'verification/verification.json').is_file()


@pytest.mark.parametrize('reply', [
    {'status': 'COMPLETE-STOCK-COVERAGE', 'decision': {'evidence_verified': True}},
    {'status': 'PASS-STOCK-REPEATABILITY-TRANSFER', 'decision': {'evidence_verified': False}},
])
def test_failed_or_wrong_public_decision_stops(evidence, monkeypatch, reply):
    native_starts(evidence)
    monkeypatch.setattr(product.subprocess, 'run', lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, json.dumps(reply), ''))
    result = product.verify(*evidence[:2], evidence[0]/'verification')
    assert result['fresh_validation'] is False
    assert result['status'] == 'VALIDATION-STOP'
    assert len(result['checks']) == 1


def test_source_mutation_invalidates_otherwise_successful_checks(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        (evidence[0]/'source.py').write_text('changed', encoding='utf-8')
        return successful_validator(command, **kwargs)
    monkeypatch.setattr(product.subprocess, 'run', launch)
    result = product.verify(*evidence[:2], evidence[0]/'verification')
    assert result['fresh_validation'] is False


def test_native_start_mutation_invalidates_success(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        save(evidence[2].parent/'raw/0/run-start.json', {'changed': True})
        return successful_validator(command, **kwargs)
    monkeypatch.setattr(product.subprocess, 'run', launch)
    assert product.verify(*evidence[:2], evidence[0]/'verification')['fresh_validation'] is False


@pytest.mark.parametrize('which', [2, 3])
def test_cannot_write_verification_inside_a_study(evidence, which):
    with pytest.raises(ValueError, match='study'):
        product.verify(*evidence[:2], evidence[which].parent/'new-output')


def test_existing_output_is_never_overwritten(evidence):
    output = evidence[0]/'verification'
    output.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        product.verify(*evidence[:2], output)


def test_wrong_frozen_root_is_rejected(evidence):
    mutate(evidence, 'wider', lambda data: data['manifest'].update(experiment_root='C:/some-other-study'))
    with pytest.raises(ValueError, match='root'):
        product.snapshot(*evidence[:2])


def test_success_for_a_different_report_is_not_accepted(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        completed = successful_validator(command, **kwargs)
        data = json.loads(completed.stdout)
        data['report'] = 'C:/different/report.json'
        completed.stdout = json.dumps(data)
        return completed
    monkeypatch.setattr(product.subprocess, 'run', launch)
    assert product.verify(*evidence[:2], evidence[0]/'verification')['fresh_validation'] is False


def test_timeout_preserves_partial_diagnostics(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 3600, output=b'partial output', stderr=b'partial error')
    monkeypatch.setattr(product.subprocess, 'run', launch)
    output = evidence[0]/'verification'
    result = product.verify(*evidence[:2], output)
    assert result['fresh_validation'] is False
    assert (output/'q6-stdout.json').read_bytes() == b'partial output'
    assert (output/'q6-stderr.log').read_bytes() == b'partial error'
    assert result['checks'][0]['timed_out'] is True


def test_malformed_public_json_is_a_retained_failure(evidence, monkeypatch):
    native_starts(evidence)
    monkeypatch.setattr(product.subprocess, 'run', lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, '[]', ''))
    result = product.verify(*evidence[:2], evidence[0]/'verification')
    assert result['status'] == 'VALIDATION-STOP'


def test_nonzero_validator_exit_cannot_be_overridden_by_json(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        completed = successful_validator(command, **kwargs)
        completed.returncode = 2
        return completed
    monkeypatch.setattr(product.subprocess, 'run', launch)
    assert product.verify(*evidence[:2], evidence[0]/'verification')['fresh_validation'] is False


def test_missing_evidence_is_an_explicit_cli_failure(evidence, capsys):
    evidence[3].unlink()
    assert product.main(['status', '--catalogue', str(evidence[1]), '--project', str(evidence[0]), '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'


def test_catalogue_and_report_cannot_be_changed_together(evidence):
    data = json.loads(evidence[3].read_bytes())
    data['cases'][0]['statistics']['gain']['geometric_change_pct'] = 999
    sha = save(evidence[3], data)
    catalog = json.loads(evidence[1].read_bytes())
    catalog['studies']['wider']['sha256'] = sha
    save(evidence[1], catalog)
    with pytest.raises(ValueError, match='digest'):
        product.snapshot(*evidence[:2])


def test_malformed_studies_is_a_cli_identity_stop(evidence, capsys):
    data = json.loads(evidence[1].read_bytes())
    data['studies'] = ['q6', 'wider']
    product.CATALOGUE_SHA256 = save(evidence[1], data)
    assert product.main(['status', '--catalogue', str(evidence[1]), '--project', str(evidence[0]), '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'


def test_malformed_decision_keeps_summary(evidence, monkeypatch):
    native_starts(evidence)
    def launch(command, **kwargs):
        completed = successful_validator(command, **kwargs)
        data = json.loads(completed.stdout)
        data['decision'] = []
        completed.stdout = json.dumps(data)
        return completed
    monkeypatch.setattr(product.subprocess, 'run', launch)
    output = evidence[0]/'verification'
    assert product.verify(*evidence[:2], output)['fresh_validation'] is False
    assert (output/'verification.json').is_file()


def test_readable_output_preserves_counts_and_comparators(evidence, capsys):
    assert product.main(['status', '--catalogue', str(evidence[1]), '--project', str(evidence[0])]) == 0
    text = capsys.readouterr().out
    assert '148' in text and '344' in text
    assert 'manual CI90' in text and 'selected default' in text
    assert product.snapshot(*evidence[:2])['cases'][0]['selected_default'] is False
