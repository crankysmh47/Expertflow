"""Fresh public read-only reconstruction and before/after integrity checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time

root = Path.cwd().resolve()
workspace = root / '.superpowers/sdd/2026-10-05-wider-stock-collector'
study = Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')
report_path = study / 'report.json'
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
integrity = runpy.run_path('docs/evidence/stock-cli-20261005/validate_live.py')['integrity']

def inventory():
    expected, starts, digest = integrity()
    for filename, wanted in report['manifest']['source_files'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == wanted, filename
    assert report_path.read_bytes() == report_bytes
    return {'wider_native_starts': len(list(study.rglob('run-start.json'))),
            'wider_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
            'original_frozen_files': len(expected['frozen_files']),
            'original_history_pins': len(expected['historical_files']),
            'original_native_starts': len(starts), 'original_report_sha256': digest,
            'frozen_source_files': len(report['manifest']['source_files'])}

before = inventory()
assert before['wider_native_starts'] == 344
assert before['original_native_starts'] == 148
started = datetime.now(timezone.utc).isoformat()
timer = time.perf_counter()
command = ['uv', 'run', '--no-sync', 'expertflow', 'stock', 'coverage', 'validate']
result = subprocess.run(command, cwd=root, capture_output=True)
wall = time.perf_counter() - timer
(workspace / 'public-validation.log').write_bytes(result.stdout)
(workspace / 'public-validation-stderr.log').write_bytes(result.stderr)
after = inventory()
assert before == after
validation = json.loads(result.stdout)
assert result.returncode == 0, validation
assert validation['status'] == 'COMPLETE-STOCK-COVERAGE'
assert validation['validation']['native_calls'] == 0
assert validation['decision']['evidence_verified'] is True
assert len(validation['cases']) == 4
assert all(case['status'] == 'NO-UTILITY-GAIN' for case in validation['cases'])
record = {'status': 'PASS', 'command': command, 'exit_code': result.returncode,
          'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
          'public_cli_wall_seconds': wall, 'additional_native_calls': 0,
          'before': before, 'after': after, 'validation': validation['validation'],
          'source_commit': report['manifest']['source_commit'],
          'scope': 'Fresh public reconstruction, full live input verification, unchanged raw/source/history evidence.'}
(workspace / 'native-verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
