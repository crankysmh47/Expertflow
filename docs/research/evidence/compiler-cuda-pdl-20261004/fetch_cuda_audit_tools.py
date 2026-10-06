"""Fetch small official audit tools locally; never install or replace CUDA runtime."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cupti-compatibility', action='store_true')
args = parser.parse_args()
version = '12.9.1' if args.cupti_compatibility else '12.8.1'
names = ('cuda_cupti',) if args.cupti_compatibility else ('cuda_cuobjdump', 'cuda_cupti', 'cuda_nvdisasm')
base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
root = Path('C:/models/expertflow/dependencies') / ('cuda-audit-' + version)
metadata = urllib.request.urlopen(base + 'redistrib_' + version + '.json', timeout=30).read()
manifest = json.loads(metadata)
packages = {name: manifest[name]['windows-x86_64'] for name in
    names}
assert sum(int(p['size']) for p in packages.values()) < (14 if args.cupti_compatibility else 26) * 1024 * 1024
root.mkdir(parents=True, exist_ok=True)
results = []
for name, package in packages.items():
    archive = root / Path(package['relative_path']).name
    if not archive.exists():
        with urllib.request.urlopen(base + package['relative_path'], timeout=60) as response:
            data = response.read(int(package['size']) + 1)
        assert len(data) == int(package['size'])
        assert hashlib.sha256(data).hexdigest() == package['sha256']
        archive.write_bytes(data)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['sha256']
    with zipfile.ZipFile(archive) as z:
        assert all((root / member.filename).resolve().is_relative_to(root.resolve()) for member in z.infolist())
        z.extractall(root)
    results.append({'package': name, 'url': base + package['relative_path'],
        'sha256': package['sha256'], 'size_bytes': int(package['size'])})
output = Path(__file__).with_name('cupti-compatibility-tools.json' if args.cupti_compatibility else 'cuda-audit-tools.json')
output.write_text(json.dumps({'metadata_url': base + 'redistrib_' + version + '.json',
    'metadata_sha256': hashlib.sha256(metadata).hexdigest(), 'root': str(root),
    'packages': results, 'installed': False, 'runtime_replaced': False}, indent=2) + '\n')
print(json.dumps({'status': 'VERIFIED-LOCAL-AUDIT-TOOLS', 'bytes': sum(x['size_bytes'] for x in results)}))
