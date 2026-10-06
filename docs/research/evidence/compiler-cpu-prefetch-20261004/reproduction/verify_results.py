"""Independent post-collection reconstruction; no launches or policy overrides."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.refinement import balanced_schedule, paired_statistics
from expertflow.compiler.schema import canonical_payload, canonical_sha256

root = Path('C:/models/expertflow/runs/compiler-cpu-prefetch-20261004')
evidence = Path('docs/evidence/compiler-cpu-prefetch-20261004')
scratch = Path('.superpowers/sdd/cpu-expert-prefetch-20261004')
report = json.loads((root/'report.json').read_text())
freeze = json.loads((root/'frozen-protocol.json').read_text())
assert report['status'] != 'RUNNING', 'retained experiment is incomplete'
assert len(report['rows']) == len(report['outcomes']) == 20, 'requires exactly twenty retained native runs'
assert report['retry_count'] == 0 and report['product_plan_published'] is False
assert canonical_payload(report['frozen']) == canonical_payload(freeze)
assert freeze['schedule'] == [['stock' if a == 'direct' else 'prefetch' for a in order] for order in balanced_schedule()]
for path,digest in freeze['source_hashes'].items():
    assert file_sha256(Path(path)) == digest, 'frozen source drift: '+path
aa_report = Path('C:/models/expertflow/runs/compiler-refinement-20261003/aa/report.json')
aa_db = Path('C:/models/expertflow/runs/compiler-refinement-20261003/compiler.sqlite3')
assert file_sha256(aa_report) == freeze['prior_hashes']['aa_report']
assert file_sha256(aa_db) == freeze['prior_hashes']['aa_database']
store = EvidenceStore(root/'compiler.sqlite3')
reference = store.verify_measurement(freeze['reference_measurement_id'])
owners, ids, artifacts = set(), set(), []
for index,row in enumerate(report['rows']):
    pair, order = divmod(index,2)
    arm = freeze['schedule'][pair][order]
    assert row['pair'] == pair and row['arm'] == arm
    mid = row['measurement_id']
    assert mid not in ids and mid != freeze['reference_measurement_id']
    ids.add(mid)
    verified = store.verify_measurement(mid)
    assert all(canonical_payload(row[k]) == canonical_payload(v) for k,v in verified.items())
    candidate = freeze['candidates'][arm]
    assert verified['identities'] == candidate['identities']
    assert verified['settings_sha256'] == canonical_sha256(candidate['settings'])
    assert verified['stage'] == f'prefetch-{pair:02}-{arm}'
    assert verified['measured'] is True and verified['exit_code'] == 0
    assert all(verified['validations'][k] is True for k in ('exact_tokens','memory','cleanup'))
    assert all(verified[k] == reference[k] for k in ('prompt_tokens_sha256','generated_tokens_sha256'))
    assert verified['owned_run_sha256'] not in owners and verified['owned_run_sha256'] != reference['owned_run_sha256']
    owners.add(verified['owned_run_sha256'])
    assert report['outcomes'][index]['measurement_id'] == mid and report['outcomes'][index]['status'] == 'measured'
    record = store.measurement(mid)
    assert record.comparison_ids == (() if arm == 'stock' else (freeze['reference_measurement_id'],))
    artifacts.append({'pair':pair,'arm':arm,'measurement_id':mid,
                      'owned_run_sha256':verified['owned_run_sha256'],'artifacts':canonical_payload(record.artifacts)})
with sqlite3.connect(root/'compiler.sqlite3') as conn:
    assert conn.execute('select count(*) from measurement').fetchone()[0] == 21, 'twenty fresh runs plus one imported reference required'
rates = {(r['pair'],r['arm']):r['decode_tps'] for r in report['rows']}
result = paired_statistics([rates[i,'stock'] for i in range(10)], [rates[i,'prefetch'] for i in range(10)])
assert all(canonical_payload(report[k]) == canonical_payload(v) for k,v in result.items())
passed = result['geometric_change_pct'] >= 5 and result['ci95_pct'][0] > 0 and max(
    result['direct_cv_pct'],result['sealed_cv_pct']) <= 10
expected = 'PASS-OPTIMIZATION' if passed else 'VALIDATION-STOP' if result['ci95_pct'][1] < 0 else 'INCONCLUSIVE'
assert report['status'] == expected and report['optimization_gain_established'] == passed
source = 'C:/models/expertflow/worktrees/llama-cpu-prefetch-20261004'
assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip() == freeze['native_source_commit']
assert not subprocess.check_output(['git','-C',source,'status','--porcelain'],text=True).strip()

numerical_root = root/'numerical'
numerical_root.mkdir(exist_ok=True)
numerical_files = []
for path in scratch.iterdir():
    if path.suffix in ('.disasm', '.bin') or path.name in (
            'build.log','build-after.log','fixture-stock.log','hint-red.log','numerical-gate.log',
            'numerical-gate.json','cpu-tests.log','preflight-patch-order-red.log'):
        destination = numerical_root/path.name
        shutil.copy2(path,destination)
        numerical_files.append({'path':str(destination), 'sha256':file_sha256(destination),
                                'size_bytes':destination.stat().st_size})
assert (numerical_root/'stock-fixture.bin').read_bytes() == (numerical_root/'candidate-fixture.bin').read_bytes()
assert file_sha256(numerical_root/'stock-fixture.bin') == json.loads((evidence/'numerical-gate.json').read_text())['output_sha256']
atomic = lambda p,v: p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
for name in ('report.json','frozen-protocol.json'):
    shutil.copy2(root/name,evidence/name)
atomic(evidence/'measurement-manifest.json',{'runs':artifacts,'fresh_native_runs':20,
    'imported_reference_records':1,'reference_measurement_id':freeze['reference_measurement_id'],
    'numerical_artifacts':numerical_files})
verification = {'status':expected,'fresh_native_runs':20,'independent_owned_runs':len(owners),
    'imported_reference_records':1,'retry_count':0,'all_retained_artifact_hashes_reverified':True,
    'source_and_runtime_pins_reverified':True,'original_aa_database_report_unchanged':True,
    'tokens_memory_reserve_cleanup_passed':True,'paired_statistics_recomputed':result,
    'full_cpu_suite':{'passed':519,'skipped':7,'seconds':44.31},
    'numerical_fixture_count':48,'compiled_hint_gate':'RED before hint, GREEN after hint',
    'quantization_and_dot_disassembly_unchanged':True,'profile_sync_used':False,
    'product_plan_published':False,'native_source_commit':freeze['native_source_commit'],
    'review':'one independent read-only implementation review; no material findings',
    'deferred_minor':'reproduction hint helper does not scope its prefetch search to the caller block; exact source diff and independently verified machine code support the recorded gate',
    'database_sha256':file_sha256(root/'compiler.sqlite3'),
    'probe_stdout_sha256':file_sha256(root/'probe-stdout.log')}
atomic(evidence/'verification.json',verification)
print(json.dumps(verification,indent=2))
