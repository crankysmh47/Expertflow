"""Complete metadata-only inventory of the separately verified Granite artifact."""
import json
from pathlib import Path
import sys

from expertflow.analysis.q6_inventory import summarize_tensor_inventory
from expertflow.compiler.preflight import file_sha256

sys.path.insert(0, 'C:/models/expertflow/worktrees/llama-q6-placement-final/gguf-py')
from gguf import GGUFReader, GGML_QUANT_SIZES

root = Path(__file__).parent
pin = json.loads((root/'download-verification.json').read_text())
model = Path(pin['path'])
assert model.stat().st_size == pin['size_bytes'] and file_sha256(model) == pin['sha256']
reader = GGUFReader(str(model), 'r')
metadata = {name: field.contents() for name, field in reader.fields.items()
            if name.startswith(('general.', 'granitemoe.'))}
assert metadata['general.architecture'] == 'granitemoe'
assert (metadata['granitemoe.expert_count'], metadata['granitemoe.expert_used_count'],
        metadata['granitemoe.block_count']) == (32, 8, 24)
tensors = []
for tensor in reader.tensors:
    block, size = GGML_QUANT_SIZES[tensor.tensor_type]
    tensors.append({'name': tensor.name, 'type': tensor.tensor_type.name,
        'shape': [int(x) for x in tensor.shape], 'bytes': int(tensor.n_bytes),
        'data_offset': int(tensor.data_offset), 'quant_block_elements': int(block),
        'quant_block_bytes': int(size), 'slice_contiguous': True})
inventory = summarize_tensor_inventory(tensors, expert_count=32)
inventory.update(schema_version='1.0.0', measurement_kind='gguf_metadata_derived',
    model={'path': str(model), 'bytes': pin['size_bytes'], 'sha256': pin['sha256'],
           'repository': pin['repository'], 'revision': pin['revision']}, gguf_metadata=metadata)
(root/'tensor-inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True)+'\n', newline='\n')
(root/'gguf-metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n', newline='\n')
print(json.dumps({k: inventory[k] for k in ('tensor_count', 'expert_count', 'routed_layer_count',
    'expert_component_names', 'total_tensor_payload_bytes', 'total_routed_expert_bank_bytes')}))
print(json.dumps(inventory['layers'][0]))
print(json.dumps([t for t in inventory['tensors'] if t['routed_expert_tensor'] and t['layer_id']==0]))
