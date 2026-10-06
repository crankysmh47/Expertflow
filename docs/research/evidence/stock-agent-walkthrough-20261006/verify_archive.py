"""Verify retained walkthrough copies and unchanged study/source inventories."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / 'archive.json').read_bytes())
    for entry in manifest['files']:
        assert sha(HERE / entry['copy']) == entry['sha256'], entry['copy']
        assert sha(entry['original']) == entry['sha256'], entry['original']
    for attempt in manifest['attempts']:
        directory = HERE / attempt['copy_directory']
        result = json.loads((directory / 'verification.json').read_bytes())
        assert result['status'] == attempt['status']
        assert result['wall_seconds'] == attempt['wall_seconds']
        for check in result['checks']:
            study = check['study']
            assert sha(directory / (study + '-stdout.json')) == check['stdout_sha256']
            assert sha(directory / (study + '-stderr.log')) == check['stderr_sha256']
            stdout = json.loads((directory / (study + '-stdout.json')).read_bytes())
            assert stdout['status'] == check['status']
        if result['status'] == 'PASS-LOCAL-STOCK-QUALIFICATION':
            assert result['fresh_validation'] is True
            assert result['additional_native_calls'] == 0
            assert [c['study'] for c in result['checks']] == ['q6', 'wider']
            assert all(c['exit_code'] == 0 for c in result['checks'])
            for study in ('q6', 'wider'):
                assert json.loads((directory / (study + '-stdout.json')).read_bytes())['decision']['evidence_verified'] is True
    previous = PROJECT / 'docs/evidence/stock-followthrough-20261006'
    source = json.loads((previous / 'corrected-helper-source-snapshot.json').read_bytes())
    with zipfile.ZipFile(previous / 'corrected-helper-source-snapshot.zip') as saved:
        for entry in source['entries']:
            assert saved.read(entry['path']) == (PROJECT / entry['path']).read_bytes()
    helper = runpy.run_path(str(PROJECT / 'scripts/stock_product_status.py'))
    inventory = helper['inventory'](PROJECT, previous / 'catalogue.json')
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    prior = json.loads((previous / 'verification.json').read_bytes())
    assert digest == manifest['integrity_sha256'] == prior['integrity_sha256']
    expected, starts, report_digest = runpy.run_path(str(
        PROJECT / 'docs/evidence/stock-cli-20261005/validate_live.py'))['integrity']()
    assert len(starts) == 148 and len(expected['frozen_files']) == 41
    assert len(expected['historical_files']) == 6
    result = {'status': 'PASS-WALKTHROUGH-ARCHIVE-AND-INTEGRITY',
              'attempts_retained': len(manifest['attempts']),
              'latest_qualification': manifest['attempts'][-1]['status'],
              'additional_native_starts': 0, 'original_starts': 148, 'wider_starts': 344,
              'source_commit': manifest['source_commit'], 'helper_source_commit': source['source_commit'],
              'original_report_sha256': report_digest,
              'independent_user_acceptance_established': False}
    (HERE / 'archive-verification.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
