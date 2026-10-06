"""Header-only inspection; model bytes and numerical eligibility are not verified here."""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/models/expertflow/worktrees/llama-q6-placement-final/gguf-py')
from gguf import GGUFReader

model = Path('C:/models/expertflow/google--gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf')
reader = GGUFReader(str(model), 'r')
metadata = {key: field.contents() for key, field in reader.fields.items()
    if key.startswith('gemma4.') or key in ('general.architecture', 'general.name', 'general.file_type', 'general.quantization_version')}
experts = [t for t in reader.tensors if '_exps.' in t.name]
result = {'status': 'HEADER-INSPECTED-NOT-WEIGHT-VERIFIED',
    'model_path': model.as_posix(), 'model_bytes': model.stat().st_size,
    'metadata': metadata, 'tensor_count': len(reader.tensors),
    'tensor_types': dict(Counter(t.tensor_type.name for t in reader.tensors)),
    'routed_tensor_count': len(experts),
    'routed_tensor_types': dict(Counter(t.tensor_type.name for t in experts)),
    'routed_tensors': [{'name': t.name, 'shape': [int(v) for v in t.shape],
        'type': t.tensor_type.name, 'bytes': int(t.n_bytes)} for t in experts],
    'scope': 'Separate Q4 artifact, not a quality-preserving substitution for Q6. Requires its own verified normalization, numerical source proof and native reference before search.'}
output = Path(__file__).parent / 'q4-header-inspection.json'
output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'routed_tensors'}))
