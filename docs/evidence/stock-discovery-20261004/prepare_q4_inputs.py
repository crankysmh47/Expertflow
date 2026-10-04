"""Normalize the complete, separately hash-verified Q4 inventory; no inference."""

import json
from pathlib import Path

from expertflow.compiler.pipeline import inspect_model
from expertflow.compiler.schema import canonical_payload, canonical_sha256

root = Path(__file__).parent
verified = json.loads((root / 'q4-weight-verification.json').read_text(encoding='utf-8-sig'))
inventory = json.loads((root / 'q4-tensor-inventory.json').read_text())
assert verified['Hash'].lower() == inventory['model']['sha256']
assert Path(verified['Path']).resolve() == Path(inventory['model']['path']).resolve()
assert Path(verified['Path']).stat().st_size == inventory['model']['bytes']
model = inspect_model(Path('configs/compiler/gemma4-q4-model.json'), root / 'q4-tensor-inventory.json')
runtime = json.loads(Path('docs/evidence/compiler-phase3/inputs/runtime-identity.json').read_text())
runtime['model'] = canonical_payload(model.identity)
(root / 'q4-runtime-identity.json').write_text(json.dumps(runtime, indent=2, sort_keys=True) + '\n', encoding='utf-8')
result = {'status': 'VERIFIED-Q4-NORMALIZATION-NOT-NATIVE-VALIDATION',
    'model': canonical_payload(model), 'model_ir_sha256': canonical_sha256(model),
    'inventory_sha256': model.inventory_sha256,
    'routed_layer_count': len(model.moe_layers),
    'routed_expert_bank_bytes': sum(layer.routed_expert_bank_bytes for layer in model.moe_layers),
    'tensor_count': inventory['tensor_count'],
    'non_expert_tensor_bytes': inventory['non_expert_tensor_bytes']}
(root / 'q4-normalization.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'model'}))
