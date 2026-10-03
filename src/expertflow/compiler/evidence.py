"""Append-only SQLite evidence; sealing rechecks raw bytes and native records."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .plan import PlanIdentities, RuntimeSettings, StaticPlacement
from .preflight import file_sha256
from .reference import canonical_json, read_json
from .schema import (ArtifactIdentity, WorkloadIR, canonical_payload, canonical_sha256,
                     require_hash, require_int, require_number, require_text)


@dataclass(frozen=True, slots=True)
class MeasurementKey:
    model_sha256: str
    hardware_sha256: str
    workload_sha256: str
    runtime_sha256: str
    settings_sha256: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            require_hash(getattr(self, name), name)

    @classmethod
    def from_candidate(cls, candidate):
        i = candidate.identities
        return cls(i.model_sha256, i.hardware_sha256, i.workload_sha256,
                   i.runtime_sha256, canonical_sha256(candidate.settings))


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    role: str
    identity: ArtifactIdentity

    def __post_init__(self):
        require_text(self.role, 'artifact role')
        if not isinstance(self.identity, ArtifactIdentity):
            raise ValueError('invalid artifact identity')


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    key: MeasurementKey
    candidate_id: str
    identities_json: str
    settings_json: str
    artifacts: tuple[EvidenceArtifact, ...]
    metrics_json: str
    validation_json: str
    exit_code: int
    measured: bool
    stage: str
    numerical_path: str
    comparison_ids: tuple[str, ...] = ()
    measurement_id: str = ''

    def __post_init__(self):
        require_text(self.candidate_id, 'candidate ID')
        require_text(self.stage, 'measurement stage')
        if type(self.exit_code) is not int or type(self.measured) is not bool:
            raise ValueError('invalid measurement exit/status')
        if self.numerical_path not in {'stock_same_runtime', 'fork_off_vs_pristine', 'numerical_path_change'}:
            raise ValueError('unsupported numerical path')
        for name in ('identities_json', 'settings_json', 'metrics_json', 'validation_json'):
            payload = json.loads(getattr(self, name))
            if not isinstance(payload, dict):
                raise ValueError(f'{name} requires JSON object')
            object.__setattr__(self, name, canonical_json(canonical_payload(payload)))
        artifacts = tuple(self.artifacts)
        if len({a.role for a in artifacts}) != len(artifacts):
            raise ValueError('duplicate artifact role')
        object.__setattr__(self, 'artifacts', artifacts)
        object.__setattr__(self, 'comparison_ids', tuple(self.comparison_ids))


class EvidenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS measurement (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL, key_sha256 TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact (
                    measurement_id TEXT NOT NULL REFERENCES measurement(id),
                    role TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (measurement_id, role)
                );
                CREATE TABLE IF NOT EXISTS validation (
                    measurement_id TEXT NOT NULL REFERENCES measurement(id),
                    name TEXT NOT NULL, passed INTEGER NOT NULL,
                    PRIMARY KEY (measurement_id, name)
                );
                CREATE INDEX IF NOT EXISTS measurement_key ON measurement(key_sha256, sequence);
            ''')
            for table in ('measurement', 'artifact', 'validation'):
                for action in ('UPDATE', 'DELETE'):
                    conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_blocked
                        BEFORE {action} ON {table} BEGIN
                        SELECT RAISE(ABORT, 'append-only evidence'); END''')

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _verify_artifacts(record):
        for artifact in record.artifacts:
            p = Path(artifact.identity.path)
            try:
                valid = p.stat().st_size == artifact.identity.size_bytes and file_sha256(p) == artifact.identity.sha256
            except OSError:
                valid = False
            if not valid:
                raise ValueError(f'artifact identity mismatch: {artifact.role}: {p}')

    def append_measurement(self, record, *, measurement_id=None):
        from dataclasses import replace
        mid = measurement_id or str(uuid.uuid4())
        require_text(mid, 'measurement ID')
        record = replace(record, measurement_id=mid)
        self._verify_artifacts(record)
        payload = canonical_payload(record)
        try:
            with self._connection() as conn:
                conn.execute('INSERT INTO measurement(id,key_sha256,payload,payload_sha256,created_at) VALUES(?,?,?,?,?)',
                             (mid, canonical_sha256(record.key), canonical_json(payload), canonical_sha256(payload),
                              datetime.now(timezone.utc).isoformat()))
                for a in record.artifacts:
                    conn.execute('INSERT INTO artifact VALUES(?,?,?)', (mid, a.role, canonical_json(canonical_payload(a))))
                for name, passed in json.loads(record.validation_json).items():
                    if type(passed) is not bool:
                        raise ValueError('validation requires boolean values')
                    conn.execute('INSERT INTO validation VALUES(?,?,?)', (mid, name, int(passed)))
        except sqlite3.IntegrityError as error:
            raise ValueError(f'duplicate or invalid evidence ID: {mid}') from error
        return mid

    def replace_measurement(self, *_):
        raise ValueError('append-only evidence')

    @staticmethod
    def _decode(row):
        payload = json.loads(row[0])
        if canonical_sha256(payload) != row[1]:
            raise ValueError('measurement payload hash mismatch')
        payload['key'] = MeasurementKey(**payload['key'])
        payload['artifacts'] = tuple(EvidenceArtifact(a['role'], ArtifactIdentity(**a['identity'])) for a in payload['artifacts'])
        return MeasurementRecord(**payload)

    def measurement(self, mid):
        with self._connection() as conn:
            row = conn.execute('SELECT payload,payload_sha256 FROM measurement WHERE id=?', (mid,)).fetchone()
        if row is None:
            raise ValueError(f'missing evidence: {mid}')
        return self._decode(row)

    def measurements_for(self, key):
        with self._connection() as conn:
            rows = conn.execute('SELECT payload,payload_sha256 FROM measurement WHERE key_sha256=? ORDER BY sequence',
                                (canonical_sha256(key),)).fetchall()
        return tuple(self._decode(row) for row in rows)

    def verify_artifacts(self, mid):
        self._verify_artifacts(self.measurement(mid))

    def verify_measurement(self, mid, _seen=frozenset()):
        if mid in _seen:
            raise ValueError('comparison evidence cycle')
        record = self.measurement(mid)
        self._verify_artifacts(record)
        artifacts = {a.role: read_json(Path(a.identity.path)) for a in record.artifacts
                     if a.role in {'completion', 'tokenize', 'request', 'memory', 'process', 'launch'}}
        if artifacts.keys() != {'completion', 'tokenize', 'request', 'memory', 'process', 'launch'}:
            raise ValueError('missing native measurement artifacts')
        raw_i = json.loads(record.identities_json)
        i = PlanIdentities(**{**raw_i, 'workload': WorkloadIR(**raw_i['workload'])})
        raw_s = json.loads(record.settings_json)
        settings = RuntimeSettings(**{**raw_s, 'static': StaticPlacement(**raw_s['static']) if raw_s['static'] else None})
        from .plan import CandidatePlan
        candidate = CandidatePlan(i, settings)
        if MeasurementKey.from_candidate(candidate) != record.key or candidate.candidate_id != record.candidate_id:
            raise ValueError('record identity/settings mismatch')
        w = i.workload
        launch = artifacts['launch']
        try:
            from .runner import RuntimeBinding
            raw_binding = launch['runtime_binding']
            binding = RuntimeBinding(ArtifactIdentity(**raw_binding['server']), raw_binding['manifest_json'],
                tuple(ArtifactIdentity(**a) for a in raw_binding['dependencies']),
                ArtifactIdentity(**raw_binding['cuda_runtime']) if raw_binding['cuda_runtime'] else None)
            binding.verify()
            if binding.sha256 != i.runtime_sha256:
                raise ValueError('runtime binding hash mismatch')
            is_fork = bool(json.loads(binding.manifest_json).get('patches'))
            if is_fork != (record.numerical_path == 'fork_off_vs_pristine') and record.numerical_path != 'numerical_path_change':
                raise ValueError('runtime comparison role mismatch')
        except (KeyError, TypeError, OSError) as error:
            raise ValueError('missing or malformed verified runtime binding') from error
        for name, expected in {'candidate_id': candidate.candidate_id,
                               'settings_sha256': canonical_sha256(settings),
                               'runtime_sha256': i.runtime_sha256, 'model_sha256': i.model_sha256,
                               'workload_sha256': i.workload_sha256}.items():
            if launch.get(name) != expected:
                raise ValueError(f'launch mismatch: {name}')
        tokens = artifacts['completion'].get('tokens')
        prompt_tokens = artifacts['tokenize'].get('tokens')
        for values in (tokens, prompt_tokens):
            if not isinstance(values, list) or not values or any(type(t) is not int or t < 0 for t in values):
                raise ValueError('missing or invalid native token IDs')
        request = artifacts['request']
        controls = {'prompt': prompt_tokens, 'n_predict': w.predict_tokens, 'seed': w.seed,
                    'temperature': w.temperature, 'ignore_eos': w.ignore_eos, 'cache_prompt': False,
                    'return_tokens': True, 'stream': False}
        if any(request.get(name) != value for name, value in controls.items()):
            raise ValueError('completion request differs from frozen workload/tokenization')
        timing = artifacts['completion'].get('timings', {})
        predicted_n, predicted_ms = timing.get('predicted_n'), timing.get('predicted_ms')
        require_int(predicted_n, 'native predicted_n')
        require_number(predicted_ms, 'native predicted_ms', 0.000001)
        if len(tokens) != w.predict_tokens or predicted_n != len(tokens):
            raise ValueError('incomplete native generated tokens')
        tps = predicted_n * 1000 / predicted_ms
        reported = json.loads(record.metrics_json).get('decode_tps')
        require_number(reported, 'recorded TPS', 0.000001)
        if not abs(tps - reported) <= 1e-9 * max(1, tps):
            raise ValueError('recorded TPS differs from native timings')
        process = artifacts['process']
        require_int(process.get('pid'), 'owned PID')
        if record.exit_code != 0 or process.get('exit_code') != 0 or process.get('exited') is not True or process.get('cleanup') is not True:
            raise ValueError('child process did not exit cleanly')
        samples = artifacts['memory'].get('samples')
        if artifacts['memory'].get('errors'):
            raise ValueError('memory counter failed')
        if not isinstance(samples, list) or not samples:
            raise ValueError('unknown process-owned memory')
        for sample in samples:
            if sample.get('pid') != process['pid']:
                raise ValueError('foreign memory PID')
            require_int(sample.get('dedicated_bytes'), 'process-owned GPU memory')
            require_int(sample.get('device_free_bytes'), 'device free memory')
            if sample['device_free_bytes'] < w.minimum_vram_reserve_mib << 20:
                raise ValueError('VRAM reserve violated')
        if record.numerical_path == 'numerical_path_change' or settings.static is not None:
            raise ValueError('numerical_path_change')
        if record.numerical_path == 'fork_off_vs_pristine':
            if not record.comparison_ids:
                raise ValueError('missing pristine comparison')
            for comparison in record.comparison_ids:
                baseline = self.verify_measurement(comparison, _seen | {mid})
                b = baseline['identities']
                if baseline['numerical_path'] != 'stock_same_runtime' or b['runtime_sha256'] == i.runtime_sha256 or any(
                    b[name] != raw_i[name] for name in ('model_sha256', 'hardware_sha256', 'workload_sha256')
                ) or baseline['generated_tokens_sha256'] != canonical_sha256(tokens) or baseline['prompt_tokens_sha256'] != canonical_sha256(prompt_tokens):
                    raise ValueError('pristine comparison mismatch')
        validations = json.loads(record.validation_json)
        if any(validations.get(name) is not True for name in ('exact_tokens', 'memory', 'cleanup')):
            raise ValueError('record validation failed')
        return {'candidate_id': record.candidate_id, 'identities': canonical_payload(i),
                'settings_sha256': canonical_sha256(settings), 'validations': validations,
                'exit_code': record.exit_code, 'measured': record.measured, 'stage': record.stage,
                'decode_tps': tps, 'generated_tokens_sha256': canonical_sha256(tokens),
                'prompt_tokens_sha256': canonical_sha256(prompt_tokens), 'numerical_path': record.numerical_path}
