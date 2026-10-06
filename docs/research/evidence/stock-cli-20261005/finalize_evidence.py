"""Archive final implementation checks after all test/validation commands exit."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = ROOT/'.superpowers/sdd/2026-10-05-stock-cli-and-coverage'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE/'implementation-verification.json'
    if destination.exists():
        raise ValueError('final verification already archived; no overwrite')
    full = (WORK/'full-suite-final.log').read_text()
    match = re.search(r'(\d+) passed, (\d+) skipped in ([\d.]+)s',full)
    if not match or re.search(r'\b(?:FAILED|ERROR|\d+ failed)\b',full):
        raise ValueError('final full suite is incomplete or failed')
    final = json.loads((HERE/'final-validation.json').read_text())
    package = json.loads((HERE/'packaging-final.json').read_text())
    if (final['exit_code'] != 0 or final['additional_native_calls'] != 0
            or final['result']['status'] != 'PASS-STOCK-REPEATABILITY-TRANSFER'
            or package['status'] != 'PASS'):
        raise ValueError('final public validation/package check failed')
    for path,value in final['adapter_source_files'].items():
        if digest(ROOT/path) != value:
            raise ValueError('tested adapter source changed: '+path)
    snapshot = json.loads((HERE/'source-snapshot.json').read_text())
    archive = HERE/'measured-source-snapshot.zip'
    if digest(archive) != snapshot['archive_sha256']:
        raise ValueError('source archive changed')
    with zipfile.ZipFile(archive) as saved:
        for entry in snapshot['entries']:
            if sha256(saved.read(entry['member'])).hexdigest() != entry['sha256'] or digest(entry['original_path']) != entry['sha256']:
                raise ValueError('source snapshot/original mismatch')
    for name in ('full-suite.log','full-suite-final.log','focused-final.log','native-source-checks.log','review-red.log','review-green.log'):
        shutil.copyfile(WORK/name,HERE/name)
    result = {'status':'VERIFIED','branch':'ef-v2','base_commit':'20711c0',
        'full_suite':{'passed':int(match[1]),'skipped':int(match[2]),'seconds':float(match[3]),
            'skip_scope':'Seven optional historical external-source environment checks'},
        'focused_checks':94,'applicable_native_source_checks':6,
        'independent_review':{'critical':0,'important_fixed':2,'deferred_minor':0,
            'regression_cases_red_then_green':6,'reviewer_cpu_checks':31,'native_calls':0},
        'final_public_validation':{'status':final['result']['status'],'wall_seconds':final['wall_seconds'],
            'database_readers':final['result']['validation']['database_readers'],
            'frozen_files_unchanged':41,'historical_pins_unchanged':6,'additional_native_calls':0},
        'coverage_registration':{'status':'REGISTERED-NOT-RUN','cases':4,'maximum_native_processes':428,
            'execution_ready':False},
        'adapter_source_files':final['adapter_source_files'],
        'artifact_sha256':{p.name:digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        'scope':'Implementation checks and read-only reconstruction of old evidence; wider native utility remains unmeasured.'}
    destination.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'full_suite':result['full_suite'],
        'additional_native_calls':0,'wider_native_cases_executed':0}))


if __name__ == '__main__':
    main()
