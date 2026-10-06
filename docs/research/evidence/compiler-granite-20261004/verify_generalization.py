"""Read-only cross-stage Granite audit; never launches a model process."""
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
    for path, digest in json.loads((evidence/'history-pins.json').read_text()).items():
        assert file_sha256(Path(path)) == digest, path
    runs = Path('C:/models/expertflow/runs')
    stages = (('reference', 10, 'REFERENCE-STABLE'),
              ('product', 20, 'PASS-STOCK-FALLBACK'),
              ('search', None, None))
    host = canonical_payload(capture_host_environment())
    owners, measurements, tokens = set(), set(), set()
    results = {}
    for stage, count, status in stages:
        root = runs/f'compiler-granite-{stage}-20261004'
        report_path = root/stage/'report.json'
        report = json.loads(report_path.read_text())
        audit_path = evidence/f'{stage}-verification.json'
        audit = json.loads(audit_path.read_text())
        if stage == 'search':
            assert report['status'] in ('RECOMMENDED-INCUMBENT', 'RECOMMENDED-CHALLENGER')
            count = 18+len(report['confirmation'])
            assert count in (18, 38)
        else:
            assert report['status'] == status
        assert file_sha256(report_path) == audit['report_sha256']
        assert len(audit['artifacts']) == count == len(report['outcomes'])
        freeze = report.get('frozen', report.get('manifest'))
        assert freeze['host_environment'] == host
        assert any(p.endswith('2026-10-04-granite-generalization.md') for p in freeze['source_files'])
        for path, digest in freeze['source_files'].items():
            blob = subprocess.check_output(['git', 'show', f'{freeze["source_commit"]}:{Path(path).as_posix()}'])
            assert hashlib.sha256(blob).hexdigest() == digest, (stage, path)
        for path, digest in freeze.get('prerequisite_files', {}).items():
            assert file_sha256(Path(path)) == digest
        if 'source_evidence_db' in freeze:
            assert file_sha256(Path(freeze['source_evidence_db'])) == freeze['source_evidence_db_sha256']
        store = EvidenceStore(root/'compiler.sqlite3')
        for mid, expected in audit['artifacts'].items():
            native = store.verify_measurement(mid)
            record = store.measurement(mid)
            assert {a.role:a.identity.sha256 for a in record.artifacts} == expected
            assert native['measured'] is True and native['exit_code'] == 0
            assert all(native['validations'][k] is True for k in ('exact_tokens', 'memory', 'cleanup'))
            assert mid not in measurements and native['owned_run_sha256'] not in owners
            measurements.add(mid)
            owners.add(native['owned_run_sha256'])
            assert native['identities']['model_sha256'] == '54f52d76dba171c90ee73150688351fd8a88aa96373af6b4c66b8e00aa1fbbce'
            settings = json.loads(record.settings_json)
            assert settings['cpu_moe'] is False and settings['gpu_layers'] == 99
            assert settings['static'] is None and 'cuda_pdl' not in settings
            assert record.key.settings_sha256 == native['settings_sha256']
            tokens.add((native['prompt_tokens_sha256'], native['generated_tokens_sha256']))
        results[stage] = {'native_processes':count, 'status':report['status'],
            'source_commit':freeze['source_commit'], 'plan_sha256':audit['plan_sha256'],
            'report_sha256':file_sha256(report_path), 'audit_sha256':file_sha256(audit_path),
            'database_sha256':file_sha256(root/'compiler.sqlite3')}
    assert len(tokens) == 1 and len(measurements) == len(owners) <= 68
    result = {'status':'VERIFIED-GRANITE-GENERALIZATION', 'stages':results,
        'unique_native_processes':len(owners), 'maximum_native_processes':68,
        'remaining_conditional_budget':68-len(owners), 'own_model_tokens_stable':True,
        'source_matches_git_revision':True, 'all_native_artifacts_reverified':True,
        'historical_gemma_plans_and_terminal_evidence_unchanged':True,
        'live_families':['gemma4', 'granitemoe'], 'universal_support':False,
        'new_static_profile_cache_or_fork_coverage':False, 'global_optimum_proven':False}
    atomic_json(evidence/'generalization-verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
