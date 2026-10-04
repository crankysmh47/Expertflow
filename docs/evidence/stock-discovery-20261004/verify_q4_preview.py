"""Verify actual Q4 preparation and trusted source scope; no native child."""

import json
from pathlib import Path

from expertflow.compiler.pipeline import atomic_json
from expertflow.compiler.preflight import capture_host_environment
from expertflow.compiler.stock_reference import _validate

root = Path(__file__).parent
manifest = json.loads((root / 'q4-reference-initial-preview.json').read_text())
candidate = _validate(manifest, capture_host_environment(), None)
atomic_json(root / 'q4-scheduling-eligibility.json', manifest['eligibility'])
result = {'status': 'VERIFIED-Q4-INITIAL-PREVIEW-NO-NATIVE',
    'manifest_sha256': manifest['manifest_sha256'],
    'model_ir_sha256': candidate.identities.model_sha256,
    'maximum_native_processes': manifest['maximum_native_processes'],
    'native_samples': 0,
    'reviewed_native_freeze': False}
atomic_json(root / 'q4-preview-verification.json', result)
print(json.dumps(result))
