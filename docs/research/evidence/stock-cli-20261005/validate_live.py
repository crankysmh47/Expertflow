"""Archive the original measured sources and time one read-only public validation."""
from datetime import datetime, timezone
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def integrity():
    expected = json.loads((ROOT/'docs/evidence/stock-repeatability-20261004/source-integrity.json').read_text())
    for path, value in {**expected['frozen_files'], **expected['historical_files']}.items():
        if digest(path) != value:
            raise ValueError('original source/history changed: ' + path)
    report = json.loads((STUDY/'report.json').read_text())
    starts = {str(p):digest(p) for p in STUDY.rglob('run-start.json')}
    if len(starts) != 148 or len(report['attempts']) != 148 or any(a['status'] != 'measured' for a in report['attempts']):
        raise ValueError('original 148-call inventory changed')
    return expected, starts, digest(STUDY/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true', help='verify final adapter without overwriting the first timing/archive')
    args = parser.parse_args()
    prefix = 'final-' if args.final else ''
    expected, starts, report_digest = integrity()
    archive = HERE/'measured-source-snapshot.zip'
    if (not args.final and archive.exists()) or (HERE/(prefix+'validation.json')).exists():
        raise ValueError('fresh evidence outputs required; do not overwrite verification')
    adapter_paths = [ROOT/'src/expertflow/cli/main.py',*sorted((ROOT/'src/expertflow/stock').glob('*.py'))]
    adapter_sources = {p.relative_to(ROOT).as_posix():digest(p) for p in adapter_paths}
    entries = []
    if args.final:
        snapshot = json.loads((HERE/'source-snapshot.json').read_text())
        if digest(archive) != snapshot['archive_sha256']:
            raise ValueError('measured source archive changed')
        entries = snapshot['entries']
    else:
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            for index, (path, value) in enumerate(sorted(expected['frozen_files'].items())):
                data = Path(path).read_bytes()
                if sha256(data).hexdigest() != value:
                    raise ValueError('snapshot changed while archiving')
                member = f'{index:02d}/{Path(path).name}'
                info = zipfile.ZipInfo(member, date_time=(2026,10,5,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(info, data)
                entries.append({'original_path':path, 'member':member, 'sha256':value, 'size_bytes':len(data)})
    with zipfile.ZipFile(archive) as saved:
        assert all(sha256(saved.read(e['member'])).hexdigest() == e['sha256'] for e in entries)
    if not args.final:
        snapshot = {'source_commit':expected['source_commit'], 'source_file_count':41,
            'archive_sha256':digest(archive), 'entries':entries,
            'scope':'Exact measured source/prerequisite bytes; native artifacts remain bound to original paths.'}
        (HERE/'source-snapshot.json').write_text(json.dumps(snapshot,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    command = [str(ROOT/'.venv/Scripts/expertflow.exe'), 'stock', '--project', str(ROOT),
        'repeatability', 'validate', '--output-dir', str(STUDY)]
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter()-before
    finished = datetime.now(timezone.utc).isoformat()
    (HERE/(prefix+'public-validation.log')).write_bytes(run.stdout)
    (HERE/(prefix+'public-validation-stderr.log')).write_bytes(run.stderr)
    after, after_starts, after_digest = integrity()
    assert expected == after and starts == after_starts and report_digest == after_digest
    assert adapter_sources == {p.relative_to(ROOT).as_posix():digest(p) for p in adapter_paths}
    result = json.loads(run.stdout) if run.stdout else {}
    verification = {'command':command, 'started_at_utc':started, 'finished_at_utc':finished,
        'wall_seconds':elapsed, 'exit_code':run.returncode, 'result':result, 'adapter_source_files':adapter_sources,
        'original_cli_wall_seconds':2191.483952, 'additional_native_calls':0,
        'frozen_files_unchanged':41, 'historical_pins_unchanged':6,
        'study_report_sha256':after_digest, 'native_start_count':len(after_starts),
        'performance_scope':'One same-host read-only invocation; timing comparison is descriptive, not a controlled performance study.'}
    (HERE/(prefix+'validation.json')).write_text(json.dumps(verification,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:verification[k] for k in ('wall_seconds','exit_code','additional_native_calls','frozen_files_unchanged')}))
    if run.returncode or result.get('status') != 'PASS-STOCK-REPEATABILITY-TRANSFER':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
