"""Strict reference inputs for the early compiler feasibility gate."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA_VERSION = '1.0.0'
_SERVER_FIELDS = {
    'schema_version', 'objective', 'concurrency', 'policy', 'runtime_interface',
    'prompt_file', 'context_size', 'predict_tokens', 'threads', 'seed', 'temperature',
    'ignore_eos', 'cache_prompt', 'kv_type_k', 'kv_type_v', 'cuda_graphs',
    'batch_size', 'microbatch_size', 'warmup_runs', 'measured_runs',
    'minimum_vram_reserve_mib', 'maximum_cv_pct', 'confirmation_pairs',
    'bootstrap_samples', 'bootstrap_seed', 'replay_tolerance_pct',
    'health_timeout_seconds', 'completion_timeout_seconds',
}
_HISTORICAL_FIELDS = {
    'schema_version', 'objective', 'concurrency', 'policy', 'runtime_interface',
    'prompt', 'context_size', 'predict_tokens', 'threads', 'seed', 'temperature',
    'ignore_eos', 'cpu_moe', 'gpu_layers', 'cuda_graphs', 'static_layers',
    'static_precompute', 'runtime_arguments', 'environment', 'exact_plan_eligible',
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise ValueError(f'invalid JSON input: {path}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}')


def _finite_number(value: object, field: str, *, minimum: float = 0) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{field} must be finite and >= {minimum}')


@dataclass(frozen=True, slots=True)
class ReferenceWorkload:
    """Frozen canonical input; callers receive copies of its JSON payload."""

    payload_json: str

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload_json.encode('utf-8')).hexdigest()


def load_reference_workload(root: Path, path: Path) -> ReferenceWorkload:
    value = read_json(path)
    interface = value.get('runtime_interface')
    if interface not in {'server_completion', 'historical_cli'}:
        raise ValueError('unsupported runtime_interface')
    expected = _SERVER_FIELDS if interface == 'server_completion' else _HISTORICAL_FIELDS
    if value.keys() != expected:
        raise ValueError(f'workload fields mismatch: missing={sorted(expected - value.keys())}, '
                         f'unknown={sorted(value.keys() - expected)}')
    for field, wanted in [('schema_version', SCHEMA_VERSION), ('objective', 'decode_tps'),
                          ('policy', 'exact'), ('concurrency', 1), ('temperature', 0.0),
                          ('ignore_eos', True)]:
        if value[field] != wanted:
            raise ValueError(f'unsupported {field}: {value[field]!r}')
    if type(value['concurrency']) is not int or type(value['ignore_eos']) is not bool:
        raise ValueError('invalid concurrency/ignore_eos type')
    _finite_number(value['temperature'], 'temperature')
    for field in ('context_size', 'predict_tokens', 'threads'):
        _positive_int(value[field], field)
    _positive_int(value['seed'], 'seed', minimum=0)
    if interface == 'server_completion':
        for field in ('warmup_runs', 'measured_runs', 'batch_size', 'microbatch_size',
                      'minimum_vram_reserve_mib', 'confirmation_pairs', 'bootstrap_samples',
                      'health_timeout_seconds', 'completion_timeout_seconds'):
            _positive_int(value[field], field)
        _positive_int(value['bootstrap_seed'], 'bootstrap_seed', minimum=0)
        for field in ('maximum_cv_pct', 'replay_tolerance_pct'):
            _finite_number(value[field], field, minimum=0.001)
        if value['microbatch_size'] > value['batch_size']:
            raise ValueError('microbatch_size exceeds batch_size')
        if value['kv_type_k'] != 'f16' or value['kv_type_v'] != 'f16':
            raise ValueError('exact workload requires f16 KV')
        if value['cache_prompt'] is not False:
            raise ValueError('cache_prompt must be false')
        if not isinstance(value['prompt_file'], str) or not value['prompt_file']:
            raise ValueError('prompt_file must be nonempty')
        try:
            value['prompt'] = (root / value['prompt_file']).read_bytes().decode('utf-8')
        except (OSError, UnicodeError) as error:
            raise ValueError(f'cannot read UTF-8 prompt: {error}') from error
    else:
        if value['exact_plan_eligible'] is not False:
            raise ValueError('historical CLI evidence cannot be exact-plan eligible')
        if value['cpu_moe'] is not True or value['static_precompute'] is not True:
            raise ValueError('invalid historical static configuration')
        layers = value['static_layers']
        if not isinstance(layers, list) or not layers or len(layers) > 12:
            raise ValueError('historical static layer capacity exceeded')
        for layer in layers:
            _positive_int(layer, 'static layer', minimum=0)
        if len(layers) != len(set(layers)):
            raise ValueError('duplicate static layers')
        if not isinstance(value['runtime_arguments'], list) or not all(
            isinstance(arg, str) for arg in value['runtime_arguments']
        ) or not isinstance(value['environment'], dict):
            raise ValueError('invalid historical launch contract')
    if value['cuda_graphs'] not in {'on', 'off'}:
        raise ValueError('invalid cuda_graphs setting')
    if not isinstance(value['prompt'], str) or not value['prompt'].strip():
        raise ValueError('prompt must be nonempty')
    if value['predict_tokens'] >= value['context_size']:
        raise ValueError('generation must fit context')
    return ReferenceWorkload(canonical_json(value))


def load_runtime_manifest(path: Path) -> dict:
    value = read_json(path)
    if value.get('schema_version') != SCHEMA_VERSION or value.get('launcher_abi') != SCHEMA_VERSION:
        raise ValueError('unsupported runtime schema/launcher ABI')
    for field in ('upstream_commit', 'expertflow_commit'):
        if not isinstance(value.get(field), str) or not re.fullmatch('[0-9a-f]{40}', value[field]):
            raise ValueError(f'invalid runtime {field}')
    binaries = value.get('binaries')
    if not isinstance(binaries, dict) or set(binaries) != {'llama-cli.exe', 'llama-server.exe'}:
        raise ValueError('runtime manifest requires CLI and server binary identities')
    patches = value.get('patches')
    if not isinstance(patches, list):
        raise ValueError('invalid ordered patches')
    digests = list(binaries.values())
    for index, patch in enumerate(patches, start=1):
        if not isinstance(patch, dict) or set(patch) != {'path', 'sha256'}:
            raise ValueError('invalid patch record')
        if not isinstance(patch['path'], str) or not Path(patch['path']).name.startswith(f'{index:04d}-'):
            raise ValueError('invalid patch order')
        digests.append(patch['sha256'])
    if any(not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) for digest in digests):
        raise ValueError('invalid runtime SHA-256')
    capabilities = value.get('capabilities')
    if not isinstance(capabilities, dict) or set(capabilities) != {'max_static_layers', 'max_static_shadows'}:
        raise ValueError('invalid runtime capabilities')
    for key, capacity in capabilities.items():
        _positive_int(capacity, key, minimum=0)
    if not patches and (value['expertflow_commit'] != value['upstream_commit'] or any(capabilities.values())):
        raise ValueError('stock must be pristine with no static capabilities')
    build = value.get('build')
    required_build = {'generator', 'configuration', 'compiler', 'cuda', 'flags'}
    if not isinstance(build, dict) or set(build) != required_build:
        raise ValueError('incomplete build identity')
    if any(not isinstance(build[field], str) or not build[field] for field in required_build - {'flags'}):
        raise ValueError('invalid build identity')
    if not isinstance(build['flags'], list) or not build['flags'] or not all(
        isinstance(flag, str) and flag for flag in build['flags']
    ):
        raise ValueError('invalid build flags')
    return value
