"""Capture the qualified native fixture, its owned exit and loaded DLL identities."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from expertflow.compiler.schema import canonical_sha256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('C:/models/expertflow/runs/cuda-pdl-feasibility-20261004-qualified')
root.mkdir(parents=True, exist_ok=False)
exe = root / 'probe.exe'
shutil.copyfile('C:/models/expertflow/builds/cuda-pdl-audit-20261004/probe.exe', exe)
stock = Path('C:/models/expertflow/builds/llama-a7312ae-cuda128-clean/bin')
cupti = Path('C:/models/expertflow/dependencies/cuda-audit-12.9.1/cuda_cupti-windows-x86_64-12.9.79-archive/lib')
cuda = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin')
proof = {'runtime_sha256': 'd13355831e463b73536253b5fc9204e1fba3537f918c2ac9cf6e172bb33d59cc',
         'probe_source_sha256': digest(Path(__file__).with_name('probe.cpp')),
         'probe_executable': {'path': str(exe), 'sha256': digest(exe)},
         'inference_runtime_replaced': False}
for arm, value in [('on', '1'), ('off', '0')]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('GGML_', 'LLAMA_ARG_', 'EXPERTFLOW'))}
    env['PATH'] = os.pathsep.join(map(str, [stock, cuda, cupti])) + os.pathsep + env.get('PATH', '')
    env['GGML_CUDA_PDL'] = value
    output, stdout, stderr = [root / (arm + suffix) for suffix in ('.bin', '-stdout.log', '-stderr.log')]
    command = [str(exe), str(output)]
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        exit_code = process.wait(timeout=60)
    loaded = {}
    for line in stderr.read_text().splitlines():
        if line.startswith('MODULE\t'):
            _, name, path = line.split('\t')
            if name.lower().startswith(('ggml', 'cudart', 'cublas', 'cupti')):
                loaded[name] = {'path': path, 'sha256': digest(path)}
    launch = root / (arm + '-launch.json')
    launch.write_text(json.dumps({'command': command, 'environment': {'PATH': env['PATH'], 'GGML_CUDA_PDL': value},
                                 'pid': process.pid, 'exit_code': exit_code, 'loaded_libraries': loaded}, indent=2)+'\n')
    assert exit_code == 0
    assert 'CUPTI unsubscribe: 0 (CUPTI_SUCCESS)' in stderr.read_text()
    proof[arm] = {'counts': json.loads(stdout.read_text()), 'output_sha256': digest(output),
                  'artifacts': {str(p): digest(p) for p in (output, stdout, stderr, launch)},
                  'launch_path': str(launch)}
assert (root/'on.bin').read_bytes() == (root/'off.bin').read_bytes()
proof['proof_sha256'] = canonical_sha256(proof)
Path(__file__).with_name('feasibility.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps({'status': 'QUALIFIED-NATIVE-PDL', 'proof_sha256': proof['proof_sha256'],
                  'on': proof['on']['counts'], 'off': proof['off']['counts']}))
