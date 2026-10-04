"""Independent read-only ten-process reference audit; no model launch."""

import argparse
import json
from pathlib import Path
import statistics

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.schema import canonical_sha256
from expertflow.compiler.stock_discovery import _candidate
from expertflow.compiler.stock_reference import load_reference_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root/'report.json').read_text())
    assert report['status'] == 'REFERENCE-STABLE' and report['product_accepted'] is False
    manifest = report['manifest']
    assert json.loads((args.root/'frozen-manifest.json').read_text()) == manifest
    body = dict(manifest)
    assert body.pop('manifest_sha256') == canonical_sha256(body)
    assert manifest['maximum_native_processes'] == 10 and manifest['maximum_cv_pct'] == 10
    for path,digest in manifest['source_files'].items():
        assert file_sha256(Path(path)) == digest, path
    store = EvidenceStore(args.database)
    rows = report['rows']
    assert len(rows) == len(report['outcomes']) == 10
    owners = set()
    artifacts = {}
    for index,row in enumerate(rows):
        native = store.verify_measurement(row['measurement_id'])
        record = store.measurement(row['measurement_id'])
        assert all(row[key] == value for key,value in native.items())
        assert row['reference_index'] == index and record.stage == 'confirmation'
        assert record.numerical_path == 'stock_same_runtime'
        assert native['measured'] is True and native['exit_code'] == 0
        assert all(native['validations'][key] is True for key in ('exact_tokens','memory','cleanup'))
        assert native['owned_run_sha256'] not in owners
        owners.add(native['owned_run_sha256'])
        assert report['outcomes'][index]['status'] == 'measured'
        assert report['outcomes'][index]['measurement_id'] == row['measurement_id']
        paths = {a.role:Path(a.identity.path) for a in record.artifacts}
        assert all(path.resolve().parent == (args.root/'raw'/f'reference-{index:02}').resolve() for path in paths.values())
        launch = json.loads(paths['launch'].read_text())
        assert launch['host_environment'] == manifest['host_environment']
        assert launch['experiment_context'] == {'manifest_sha256':manifest['manifest_sha256']}
        assert json.loads(paths['run-start'].read_text())['started_monotonic_ns'] >= manifest['frozen_monotonic_ns']
        artifacts[row['measurement_id']] = {a.role:a.identity.sha256 for a in record.artifacts}
    for key in ('prompt_tokens_sha256','generated_tokens_sha256','candidate_id','settings_sha256'):
        assert len({row[key] for row in rows}) == 1
    mean = statistics.mean(row['decode_tps'] for row in rows)
    cv = 100*statistics.stdev(row['decode_tps'] for row in rows)/mean
    assert mean == report['mean_tps'] and cv == report['cv_pct'] and cv <= 10
    plan = load_reference_plan(args.root/'diagnostic',store,
        identities=_candidate(manifest['candidate']).identities,
        host_environment=capture_host_environment())
    result = {'status':'VERIFIED-STOCK-REFERENCE','native_runs':10,'mean_tps':mean,'cv_pct':cv,
        'plan_sha256':plan.plan_sha256,'product_accepted':False,'source_files_unchanged':True,
        'report_sha256':file_sha256(args.root/'report.json'),'artifacts':artifacts}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','native_runs','mean_tps','cv_pct','plan_sha256','product_accepted')}))


if __name__ == '__main__':
    main()
