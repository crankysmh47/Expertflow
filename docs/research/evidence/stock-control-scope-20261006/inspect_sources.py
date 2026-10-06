"""Read-only upstream source/record inventory; never invokes a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path.cwd().resolve()
DESTINATION = ROOT / 'docs/evidence/stock-control-scope-20261006'
STUDY = Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')
REVISION = 'a7312ae94f801fc9c6786dc56e38df57b964f697'
FILES = (
    'src/llama-model.cpp', 'src/llama-context.cpp', 'src/llama-graph.cpp',
    'src/llama-batch.cpp', 'common/arg.cpp',
    'ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ops.cpp',
    'ggml/src/ggml-cpu/arch/x86/repack.cpp',
    'ggml/src/ggml-cuda/ggml-cuda.cu', 'ggml/src/ggml-cuda/mmvq.cu',
    'ggml/src/ggml-cuda/mmq.cu', 'ggml/src/ggml-cuda/fattn.cu',
    'ggml/src/ggml-cuda/fattn-common.cuh',
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def inventory():
    raw = (STUDY / 'report.json').read_bytes()
    report = json.loads(raw)
    for name, expected in report['manifest']['source_files'].items():
        assert sha(Path(name).read_bytes()) == expected, name
    cases = []
    for case in report['cases']:
        output = STUDY / case['case_id'] / 'raw/reference-00'
        token_bytes = (output / 'tokenize.json').read_bytes()
        completion_bytes = (output / 'completion.json').read_bytes()
        completion = json.loads(completion_bytes)
        workload = report['manifest']['case_inputs'][case['case_id']]['workload']
        cases.append({'case_id': case['case_id'],
            'tokenize_prompt_tokens': len(json.loads(token_bytes)['tokens']),
            'native_prompt_tokens': completion['timings']['prompt_n'],
            'native_decode_tokens': completion['timings']['predicted_n'],
            'tokenize_sha256': sha(token_bytes), 'completion_sha256': sha(completion_bytes),
            'workload': {k: workload[k] for k in ('context_size','batch_size','microbatch_size',
                'concurrency','policy','objective','predict_tokens')},
            'native_reference_dir': str(output)})
    return {'report_sha256': sha(raw), 'native_starts': len(list(STUDY.rglob('run-start.json'))),
            'frozen_source_files': len(report['manifest']['source_files']), 'cases': cases}

def main():
    registration = json.loads((ROOT/'configs/compiler/stock-coverage-20261005.json').read_bytes())
    repository = registration['source_repository']
    git = lambda *args: subprocess.check_output(['git','-C',repository,*args])
    assert git('rev-parse',REVISION+'^{commit}').decode().strip() == REVISION
    before = inventory()
    archive = DESTINATION / 'upstream-source-objects.zip'
    if archive.exists():
        raise ValueError('inspection archive exists; do not overwrite')
    entries = {}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for name in FILES:
            data = git('show',REVISION+':'+name)
            blob = git('rev-parse',REVISION+':'+name).decode().strip()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == blob
            entries[name] = {'git_blob':blob,'sha256':sha(data),'size_bytes':len(data)}
            output.writestr(name,data)
    with zipfile.ZipFile(archive) as output:
        assert all(sha(output.read(name)) == entry['sha256'] for name,entry in entries.items())
    after = inventory()
    assert before == after
    local_files = ('src/expertflow/compiler/stock_eligibility.py',
        'src/expertflow/compiler/stock_search.py','src/expertflow/compiler/runner.py',
        'src/expertflow/compiler/plan.py','src/expertflow/compiler/schema.py')
    result = {'status':'READ-ONLY-INSPECTION-PASS','inspected_at_utc':datetime.now(timezone.utc).isoformat(),
        'upstream_revision':REVISION,'repository':repository,
        'repository_worktree_head':git('rev-parse','HEAD').decode().strip(),
        'source_objects':entries,'source_archive_sha256':sha(archive.read_bytes()),
        'local_contract_files':{name:sha((ROOT/name).read_bytes()) for name in local_files},
        'before':before,'after':after,'additional_native_calls':0,
        'scope':'Source inventory and actual short-prompt shape evidence only; no new control admission or performance/quality measurement.'}
    (DESTINATION/'source-inspection.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'upstream_objects':len(entries),'additional_native_calls':0,
        'native_starts_unchanged':after['native_starts'],'prompt_tokens':[c['native_prompt_tokens'] for c in after['cases']]}))

if __name__ == '__main__':
    main()
