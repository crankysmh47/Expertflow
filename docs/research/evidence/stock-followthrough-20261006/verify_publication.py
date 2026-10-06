"""Verify the published helper qualification and original study bindings."""
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
    published = json.loads((HERE/'publication.json').read_bytes())
    for name, digest in published['files'].items():
        assert sha(HERE/name) == digest, name
        assert sha(Path(published['original_root'])/name) == digest, name
    final = json.loads((HERE/'verification.json').read_bytes())
    assert final['status'] == 'PASS-LOCAL-STOCK-QUALIFICATION'
    assert final['fresh_validation'] is True and final['additional_native_calls'] == 0
    assert [c['study'] for c in final['checks']] == ['q6', 'wider']
    for check in final['checks']:
        assert check['exit_code'] == 0
        for suffix, field in (('-stdout.json', 'stdout_sha256'), ('-stderr.log', 'stderr_sha256')):
            assert sha(HERE/(check['study']+suffix)) == check[field]
        response = json.loads((HERE/(check['study']+'-stdout.json')).read_bytes())
        assert response['decision']['evidence_verified'] is True
        assert response['status'] == check['status']
    source = json.loads((HERE/'corrected-helper-source-snapshot.json').read_bytes())
    assert source['source_commit'] == published['helper_source_commit']
    archive = HERE/'corrected-helper-source-snapshot.zip'
    assert sha(archive) == source['archive_sha256']
    with zipfile.ZipFile(archive) as saved:
        for entry in source['entries']:
            raw = saved.read(entry['path'])
            assert hashlib.sha256(raw).hexdigest() == entry['sha256']
            assert (PROJECT/entry['path']).read_bytes() == raw
    old = json.loads((HERE/'precorrection/preservation.json').read_bytes())
    for name, digest in old['files'].items():
        assert sha(HERE/'precorrection'/name) == digest
    assert old['overall_fresh_validation'] is False and old['additional_native_calls'] == 0
    helper = runpy.run_path(str(PROJECT/'scripts/stock_product_status.py'))
    catalogue = HERE/'catalogue.json'
    inventory = helper['inventory'](PROJECT, catalogue)
    assert hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest() == final['integrity_sha256']
    expected, starts, original_report = runpy.run_path(
        str(PROJECT/'docs/evidence/stock-cli-20261005/validate_live.py'))['integrity']()
    assert len(starts) == 148
    result = {'status': 'PASS-PUBLISHED-SOURCE-AND-QUALIFICATION-BINDINGS',
        'helper_source_commit': source['source_commit'], 'qualification_seconds': final['wall_seconds'],
        'additional_native_calls': 0, 'original_frozen_files': len(expected['frozen_files']),
        'original_history_pins': len(expected['historical_files']), 'original_native_starts': len(starts),
        'original_report_sha256': original_report, 'wider_native_starts': 344,
        'user_acceptance': 'NOT-TRIED',
        'limits': 'Verifies retained read-only qualification and current source/inventory bindings; no new model outputs or user observations.'}
    (HERE/'publication-verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
