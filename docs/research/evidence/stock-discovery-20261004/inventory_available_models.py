"""Record bounded local metadata inventory without reading weights or launching models."""
import json
from pathlib import Path

from expertflow.compiler.pipeline import atomic_json
from expertflow.compiler.preflight import file_sha256


def main():
    cache = Path('C:/Users/Hank47/.cache/huggingface/hub')
    cached = []
    for path in sorted(cache.glob('models--*/snapshots/*/config.json')):
        config = json.loads(path.read_text(encoding='utf-8'))
        text = config.get('text_config') or {}
        cached.append({'config_path': str(path), 'config_sha256': file_sha256(path),
            'model_type': config.get('model_type'), 'architectures': config.get('architectures'),
            'num_experts': config.get('num_experts'), 'num_local_experts': config.get('num_local_experts'),
            'text_model_type': text.get('model_type'), 'text_num_experts': text.get('num_experts'),
            'eligible_live_moe': False,
            'reason': 'not identified as a supported routed MoE; no verified GGUF/runtime/provider scope'})
    artifacts = []
    for quant, path in (
        ('Q6_K', Path('C:/models/gemma-4-26b-a4b-q6/google_gemma-4-26B-A4B-it-Q6_K.gguf')),
        ('Q4_0', Path('C:/models/expertflow/google--gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf')),
    ):
        artifacts.append({'family': 'gemma4', 'quantization': quant, 'path': str(path),
            'exists': path.is_file(), 'size_bytes': path.stat().st_size if path.is_file() else None,
            'weight_digest_recomputed_by_this_inventory': False,
            'verification_evidence': 'q4-weight-verification.json' if quant == 'Q4_0' else 'scheduling-eligibility.json'})
    result = {'status': 'LOCAL-METADATA-INVENTORY-NOT-UNIVERSAL-COVERAGE',
        'scope': ['C:/models', str(cache)], 'recursive_pc_search': False,
        'verified_supported_family_count': 1, 'supported_artifacts': artifacts,
        'cached_metadata': cached, 'second_verified_moe_family_available_in_this_inventory': False,
        'limitation': 'does not prove absence elsewhere on the PC; cache metadata is not verified model weights'}
    output = Path(__file__).with_name('local-model-inventory.json')
    atomic_json(output, result)
    print(json.dumps({'status': result['status'], 'artifacts': len(artifacts), 'cached_configs': len(cached)}))


if __name__ == '__main__':
    main()
