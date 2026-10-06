"""Reconstruct the explicit-enable mechanism rejection without a native launch."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

PROJECT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REVISION = 'a7312ae94f801fc9c6786dc56e38df57b964f697'
REPOSITORY = Path('C:/models/expertflow/worktrees/llama-q6-placement-final')
ROOTS = [Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004'),
         Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def state():
    result = {}
    for root, expected_count in zip(ROOTS, (148, 344), strict=True):
        report_bytes = (root/'report.json').read_bytes()
        report = json.loads(report_bytes)
        for name, expected in report['manifest']['source_files'].items():
            assert sha(Path(name).read_bytes()) == expected, name
        starts = {str(p): sha(p.read_bytes()) for p in root.rglob('run-start.json')}
        assert len(starts) == expected_count
        launches = {}
        for path in root.rglob('launch.json'):
            data = path.read_bytes()
            argv = json.loads(data)['argv']
            assert not any(arg == '-fa' or arg.startswith('--flash-attn') for arg in argv)
            launches[str(path)] = sha(data)
        assert len(launches) == expected_count
        result[str(root)] = {'report_sha256': sha(report_bytes),
                             'run_starts': starts, 'launches': launches}
    return result


def main():
    before = state()
    archive = HERE/'research-source-objects.zip'
    inventory_path = HERE/'research-source-inventory.json'
    objects = {}
    files = ('common/common.h', 'common/common.cpp', 'src/llama-context.cpp')
    for name in files:
        data = subprocess.check_output(['git', '-C', str(REPOSITORY), 'show', REVISION+':'+name])
        blob = subprocess.check_output(['git', '-C', str(REPOSITORY), 'rev-parse', REVISION+':'+name]).decode().strip()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == blob
        objects[name] = data
    objects['local/runner.py'] = (PROJECT/'src/expertflow/compiler/runner.py').read_bytes()
    header, context, common, runner = (objects[n].decode('utf-8') for n in (
        'common/common.h', 'src/llama-context.cpp', 'common/common.cpp', 'local/runner.py'))
    assert 'flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO' in header
    assert 'cparams.flash_attn_type   = params.flash_attn_type;' in common
    assert 'cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;' in context
    assert 'cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;' in context
    assert 'resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);' in context
    assert 'if (device_mismatch) {\n            enabled = false;' in context
    assert "startswith(('EXPERTFLOW', 'LLAMA_EXPERTFLOW', 'LLAMA_ARG_', 'GGML_'))" in runner
    if not archive.exists():
        assert not inventory_path.exists()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            for name, data in objects.items():
                output.writestr(name, data)
        record = {'revision': REVISION, 'source_objects': {
            name: {'sha256': sha(data), 'size_bytes': len(data)} for name, data in objects.items()},
            'archive_sha256': sha(archive.read_bytes()), 'before': before}
        inventory_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    record = json.loads(inventory_path.read_bytes())
    assert sha(archive.read_bytes()) == record['archive_sha256']
    with zipfile.ZipFile(archive) as saved:
        assert set(saved.namelist()) == set(objects)
        assert all(saved.read(name) == data for name, data in objects.items())
    assert before == record['before'] == state()
    result = {'status': 'NO-NEW-SUPPORTED-FUSED-DECODE-MECHANISM',
        'candidate': 'explicit flash-attention enabled versus tuned pristine stock AUTO',
        'upstream_revision': REVISION, 'source_objects': 3, 'local_bound_runner': 1,
        'launches_checked': 492, 'additional_native_calls': 0,
        'source_reason': 'AUTO already enables fused attention when its support probe accepts backend placement; '
                         'explicit enable supplies no new supported GPU kernel when the probe rejects it.',
        'limits': 'Conditional source-path rejection of this explicit-enable rationale, not measured speed/quality, '
                  'not proof that every attention control is useless; resolved per-layer baseline modes remain unmeasured.',
        'next_path': 'bounded stock prototype qualification; no alternate attention candidate or native budget reuse'}
    (HERE/'research-gate.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
