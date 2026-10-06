"""Verify archived upstream objects, native shape evidence and unchanged studies."""
import hashlib
import json
from pathlib import Path
import runpy
import re
import subprocess
import zipfile

ROOT = Path.cwd().resolve()
DIRECTORY = ROOT/'docs/evidence/stock-control-scope-20261006'
REVISION = 'a7312ae94f801fc9c6786dc56e38df57b964f697'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    first = json.loads((DIRECTORY/'source-inspection.json').read_bytes())
    second = json.loads((DIRECTORY/'supplemental-source-objects.json').read_bytes())
    assert first['before'] == first['after']
    assert first['upstream_revision'] == second['upstream_revision'] == REVISION
    files = {}
    for name, record, digest_name in (
            ('upstream-source-objects.zip',first,'source_archive_sha256'),
            ('supplemental-source-objects.zip',second,'archive_sha256')):
        archive = DIRECTORY/name
        assert sha(archive.read_bytes()) == record[digest_name]
        with zipfile.ZipFile(archive) as saved:
            assert set(saved.namelist()) == set(record['source_objects'])
            for member, entry in record['source_objects'].items():
                data = saved.read(member)
                assert len(data) == entry['size_bytes'] and sha(data) == entry['sha256']
                assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry['git_blob']
                current = subprocess.check_output(['git','-C',first['repository'],'show',REVISION+':'+member])
                assert data == current
                files[member] = data.decode('utf-8').splitlines()
    excerpts = (DIRECTORY/'source-excerpts.md').read_text(encoding='utf-8')
    sections = re.findall(r'`([^`]+):(\d+)-(\d+)`\s+```cpp\n(.*?)\n```', excerpts, re.S)
    assert len(sections) == 17
    for name, start, end, body in sections:
        start, end = int(start), int(end)
        expected_excerpt = '\n'.join(f'{number}: {files[name][number-1]}'.rstrip()
                                     for number in range(start, end+1))
        assert body == expected_excerpt, (name, start, end)
    for name, digest in first['local_contract_files'].items():
        assert sha((ROOT/name).read_bytes()) == digest
    wider_root = Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')
    report = json.loads((wider_root/'report.json').read_bytes())
    assert sha((wider_root/'report.json').read_bytes()) == first['after']['report_sha256']
    assert len(list(wider_root.rglob('run-start.json'))) == first['after']['native_starts'] == 344
    for name, digest in report['manifest']['source_files'].items():
        assert sha(Path(name).read_bytes()) == digest
    for case in first['after']['cases']:
        output = Path(case['native_reference_dir'])
        token_bytes = (output/'tokenize.json').read_bytes()
        completion_bytes = (output/'completion.json').read_bytes()
        assert sha(token_bytes) == case['tokenize_sha256']
        assert sha(completion_bytes) == case['completion_sha256']
        assert len(json.loads(token_bytes)['tokens']) == case['tokenize_prompt_tokens'] == case['native_prompt_tokens']
        timings = json.loads(completion_bytes)['timings']
        assert timings['prompt_n'] == case['native_prompt_tokens']
        assert timings['predicted_n'] == case['native_decode_tokens']
        bound_workload = report['manifest']['case_inputs'][case['case_id']]['workload']
        assert all(bound_workload[key] == value for key, value in case['workload'].items())
        assert case['native_decode_tokens'] == 512 and case['workload']['concurrency'] == 1
        assert case['workload']['objective'] == 'decode_tps' and case['workload']['policy'] == 'exact'
    expected, starts, digest = runpy.run_path('docs/evidence/stock-cli-20261005/validate_live.py')['integrity']()
    assert len(starts) == 148
    result = {'status':'AUDIT-EVIDENCE-VERIFIED','upstream_revision':REVISION,
        'upstream_objects':len(files),'verified_operation_excerpts':len(sections),
        'wider_source_files_unchanged':len(report['manifest']['source_files']),
        'wider_native_starts':344,'wider_report_sha256':first['after']['report_sha256'],
        'original_frozen_files':len(expected['frozen_files']),
        'original_history_pins':len(expected['historical_files']),
        'original_native_starts':len(starts),'original_report_sha256':digest,
        'additional_native_calls':0,'limits':'Verifies source/shape provenance, not new arithmetic equivalence or performance.'}
    (DIRECTORY/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
