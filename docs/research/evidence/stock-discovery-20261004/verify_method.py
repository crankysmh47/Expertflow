"""Final read-only audit of all registered live stages and their source revisions."""
import hashlib
import json
from pathlib import Path
import subprocess

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import atomic_json
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.schema import canonical_payload


def main():
    evidence = Path(__file__).parent
    runs = Path('C:/models/expertflow/runs')
    stages = (
        ('q6-product', 'compiler-stock-product-20261004', 'product', 'product-verification.json', 20, 'PASS-STOCK-FALLBACK'),
        ('q6-search', 'compiler-stock-search-20261004', 'search', 'search-verification.json', 32, 'RECOMMENDED-INCUMBENT'),
        ('q4-reference', 'compiler-q4-stock-reference-20261004', 'reference', 'q4-reference-verification.json', 10, 'REFERENCE-STABLE'),
        ('q4-product', 'compiler-q4-stock-product-20261004', 'product', 'q4-product-verification.json', 20, 'PASS-STOCK-FALLBACK'),
        ('q4-search', 'compiler-q4-stock-search-20261004', 'search', 'q4-search-verification.json', 18, 'RECOMMENDED-INCUMBENT'),
    )
    host = canonical_payload(capture_host_environment())
    owners, measurements = set(), set()
    results = {}
    for name, folder, subdir, audit_file, count, status in stages:
        root = runs / folder
        report_path = root / subdir / 'report.json'
        report = json.loads(report_path.read_text())
        audit_path = evidence / audit_file
        audit = json.loads(audit_path.read_text())
        assert report['status'] == status
        assert file_sha256(report_path) == audit['report_sha256']
        assert len(audit['artifacts']) == count == len(report['outcomes'])
        freeze = report.get('frozen', report.get('manifest'))
        assert freeze['host_environment'] == host
        for path, digest in freeze['source_files'].items():
            blob = subprocess.check_output(['git', 'show', f'{freeze["source_commit"]}:{path.replace(chr(92), "/")}'])
            assert hashlib.sha256(blob).hexdigest() == digest, (name, path)
        for path, digest in freeze.get('prerequisite_files', {}).items():
            assert file_sha256(Path(path)) == digest
        if 'source_evidence_db' in freeze:
            assert file_sha256(Path(freeze['source_evidence_db'])) == freeze['source_evidence_db_sha256']
        store = EvidenceStore(root / 'compiler.sqlite3')
        for mid, expected in audit['artifacts'].items():
            native = store.verify_measurement(mid)
            record = store.measurement(mid)
            assert {a.role: a.identity.sha256 for a in record.artifacts} == expected
            assert native['measured'] is True and native['exit_code'] == 0
            assert all(native['validations'][key] is True for key in ('exact_tokens', 'memory', 'cleanup'))
            assert mid not in measurements and native['owned_run_sha256'] not in owners
            measurements.add(mid)
            owners.add(native['owned_run_sha256'])
        results[name] = {'native_processes': count, 'status': status,
            'report_sha256': file_sha256(report_path), 'audit_sha256': file_sha256(audit_path),
            'database_sha256': file_sha256(root / 'compiler.sqlite3'),
            'plan_sha256': audit['plan_sha256'], 'source_commit': freeze['source_commit'],
            'all_native_artifacts_reverified': True, 'frozen_source_matches_git_revision': True}
        print(f'{name}: {count} native records reverified', flush=True)
    history = json.loads((evidence / 'history-audit.json').read_text())
    for path, digest in history['sources'].items():
        assert file_sha256(Path(path)) == digest
    log = Path('.superpowers/sdd/stock-configuration-discovery/reuse-full-suite-post-review.log')
    assert '670 passed, 7 skipped' in log.read_text()
    source_log = Path('.superpowers/sdd/stock-configuration-discovery/reuse-native-source-contracts-post-review.log')
    assert '6 passed' in source_log.read_text()
    assert len(owners) == len(measurements) == 100
    result = {'status': 'VERIFIED-REGISTERED-STOCK-METHOD', 'stages': results,
        'unique_native_processes': 100, 'q4_consumed': 48, 'q4_maximum': 68,
        'q4_unused_conditional_confirmation': 20, 'historical_sources_unchanged': True,
        'full_suite_log_sha256': file_sha256(log), 'source_check_log_sha256': file_sha256(source_log),
        'live_family_count': 1, 'live_quantizations': ['Q6_K', 'Q4_0'],
        'universal_live_coverage': False, 'global_optimum_proven': False,
        'historical_q6_server_speed_recovered': False,
        'native_accepted_search_consumer_extra_run': False}
    atomic_json(evidence / 'method-verification.json', result)
    print(json.dumps({'status': result['status'], 'unique_native_processes': len(owners), 'q4_consumed': 48}))


if __name__ == '__main__':
    main()
