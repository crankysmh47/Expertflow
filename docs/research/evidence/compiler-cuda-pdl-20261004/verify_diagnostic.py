"""Independent terminal native/source/statistical reconstruction; no new model run."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

from expertflow.compiler.evidence import EvidenceStore

root = Path('C:/models/expertflow/runs/compiler-cuda-pdl-20261004')
report = json.loads((root/'experiment/report.json').read_text())
frozen = report['frozen']
for name, digest in frozen['source_files'].items():
    data = subprocess.check_output(['git', 'show', frozen['source_commit']+':'+Path(name).as_posix()])
    assert hashlib.sha256(data).hexdigest() == digest, name
store = EvidenceStore(root/'compiler.sqlite3')
assert len(report['rows']) == len(report['outcomes']) == 20
owners, ids, arms = set(), set(), {'direct': {}, 'sealed': {}}
for index, row in enumerate(report['rows']):
    native = store.verify_measurement(row['measurement_id'])
    assert native['measured'] and native['exit_code'] == 0
    assert all(native['validations'][k] for k in ('exact_tokens', 'memory', 'cleanup'))
    assert native['owned_run_sha256'] not in owners
    assert row['measurement_id'] not in ids
    owners.add(native['owned_run_sha256']); ids.add(row['measurement_id'])
    pair, ordinal = divmod(index, 2)
    arm = frozen['schedule'][pair][ordinal]
    assert row['pair'] == pair and row['arm'] == arm
    for key in ('prompt_tokens_sha256', 'generated_tokens_sha256'):
        assert native[key] == frozen['reference_tokens'][key]
    record = store.measurement(row['measurement_id'])
    paths = {a.role: Path(a.identity.path) for a in record.artifacts}
    launch = json.loads(paths['launch'].read_text())
    assert launch['environment'].get('GGML_CUDA_PDL') == ('0' if arm == 'sealed' else None)
    completion = json.loads(paths['completion'].read_text())
    t = completion['timings']
    rate = t['predicted_n'] * 1000 / t['predicted_ms']
    assert math.isclose(rate, native['decode_tps'], rel_tol=1e-12)
    arms[arm][pair] = rate
logs = [math.log(arms['sealed'][i]/arms['direct'][i]) for i in range(10)]
rng = random.Random(20261003)
draws = sorted(100*math.expm1(statistics.mean(rng.choices(logs, k=10))) for _ in range(10000))
gain = 100*math.expm1(statistics.mean(logs))
ci = [draws[249], draws[9749]]
claimed = report['statistics']
assert abs(gain-claimed['geometric_change_pct']) < 1e-10
assert all(abs(a-b) < 1e-10 for a,b in zip(ci, claimed['ci95_pct']))
for arm, prefix in [('direct','direct'), ('sealed','sealed')]:
    values = list(arms[arm].values())
    cv = 100*statistics.stdev(values)/statistics.mean(values)
    assert abs(cv-claimed[prefix+'_cv_pct']) < 1e-10
assert report['status'] == 'NO-GO' and not claimed['gain_pass'] and not report['product_accepted']
result = {'status': 'VERIFIED-NATIVE-PDL-NO-GO', 'native_runs':20, 'unique_owners':len(owners),
          'source_commit': frozen['source_commit'], 'source_files_checked': len(frozen['source_files']),
          'geometric_change_pct':gain, 'ci95_pct':ci, 'accepted_plan_changed':False}
Path(__file__).with_name('verification.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
print(json.dumps(result))
