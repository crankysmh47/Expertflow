"""Fetch one immutable public GGUF under a fixed2GiB transfer budget."""
import hashlib
import json
from pathlib import Path
import urllib.request

repo = 'bartowski/granite-3.1-1b-a400m-instruct-GGUF'
revision = '940d2e1f9f65330615c7c8e980e6c5ac73d3360c'
filename = 'granite-3.1-1b-a400m-instruct-Q6_K.gguf'
expected_size = 1099212096
expected_sha = '4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f'
root = Path('C:/models/expertflow/models/granite-3.1-1b-a400m-q6')
root.mkdir(parents=True, exist_ok=True)
model = root / filename
part = model.with_suffix('.gguf.part')
evidence = Path(__file__).parent
metadata_url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
metadata_bytes = urllib.request.urlopen(metadata_url, timeout=30).read()
metadata = json.loads(metadata_bytes)
entry = next(x for x in metadata['siblings'] if x['rfilename'] == filename)
assert metadata['sha'] == revision
assert entry['lfs']['sha256'] == expected_sha and entry['lfs']['size'] == expected_size
(evidence/'publisher-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', newline='\n')
base_repo = 'ibm-granite/granite-3.1-1b-a400m-instruct'
base_meta = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{base_repo}', timeout=30))
base_revision = base_meta['sha']
config_url = f'https://huggingface.co/{base_repo}/resolve/{base_revision}/config.json'
config_bytes = urllib.request.urlopen(config_url, timeout=30).read()
(evidence/'official-config.json').write_bytes(config_bytes)
freeze = {'repository': repo, 'revision': revision, 'filename': filename,
          'size_bytes': expected_size, 'sha256': expected_sha, 'path': str(model),
          'transfer_budget_bytes': 2*1024**3, 'official_base_repository': base_repo,
          'official_base_revision': base_revision, 'official_config_url': config_url,
          'official_config_sha256': hashlib.sha256(config_bytes).hexdigest()}
(evidence/'artifact-pin.json').write_text(json.dumps(freeze, indent=2)+'\n', newline='\n')
assert not model.exists() and not part.exists(), 'preserve existing files; no silent overwrite'
url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true'
count = 0
hasher = hashlib.sha256()
with urllib.request.urlopen(url, timeout=30) as response, part.open('xb') as output:
    while True:
        data = response.read(1024*1024)
        if not data: break
        count += len(data)
        assert count <= expected_size and count <= freeze['transfer_budget_bytes']
        hasher.update(data); output.write(data)
        if count % (64*1024*1024) == 0:
            print(json.dumps({'downloaded_bytes':count, 'expected_bytes':expected_size}), flush=True)
assert count == expected_size and hasher.hexdigest() == expected_sha
part.rename(model)
(evidence/'download-verification.json').write_text(json.dumps({**freeze,
    'status':'VERIFIED-GGUF-BYTES', 'transferred_bytes':count}, indent=2)+'\n', newline='\n')
print(json.dumps({'status':'VERIFIED-GGUF-BYTES','bytes':count,'sha256':expected_sha}), flush=True)
