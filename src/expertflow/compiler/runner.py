"""Owned sequential native server measurements, with fail-closed GPU accounting."""

from dataclasses import dataclass
import json
import http.client
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request

from .evidence import EvidenceArtifact, MeasurementKey, MeasurementRecord
from .preflight import DEFAULT_CUDA_RUNTIME, file_sha256
from .reference import canonical_json, load_runtime_manifest
from .schema import ArtifactIdentity, ModelIR, canonical_payload, canonical_sha256, require_int


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    server: ArtifactIdentity
    manifest_json: str
    dependencies: tuple[ArtifactIdentity, ...]
    cuda_runtime: ArtifactIdentity | None

    @property
    def sha256(self):
        return canonical_sha256(self)

    def verify(self):
        for identity in (self.server, *self.dependencies, *((self.cuda_runtime,) if self.cuda_runtime else ())):
            p = Path(identity.path)
            if not p.is_file() or p.stat().st_size != identity.size_bytes or file_sha256(p) != identity.sha256:
                raise ValueError(f'runtime dependency identity mismatch: {p}')

    @classmethod
    def from_manifest(cls, root, manifest_path, binary_dir, cuda_runtime=DEFAULT_CUDA_RUNTIME):
        manifest = load_runtime_manifest(Path(manifest_path))

        def identity(path, expected):
            path = Path(path).resolve()
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f'pinned runtime identity mismatch: {path}')
            return ArtifactIdentity(str(path), path.stat().st_size, expected)

        for patch in manifest['patches']:
            identity(Path(root) / patch['path'], patch['sha256'])
        binary_dir = Path(binary_dir)
        if {p.name.lower() for p in binary_dir.glob('*.dll')} != {name.lower() for name in manifest['dependencies']}:
            raise ValueError('runtime contains missing or unpinned DLLs')
        # Both launchers are pinned even though the product uses only the server.
        for name, expected in manifest['binaries'].items():
            identity(binary_dir / name, expected)
        return cls(identity(binary_dir / 'llama-server.exe', manifest['binaries']['llama-server.exe']),
                   canonical_json(manifest), tuple(identity(binary_dir / name, expected)
                       for name, expected in sorted(manifest['dependencies'].items())),
                   identity(cuda_runtime, manifest['cuda_runtime_sha256']))


@dataclass(frozen=True, slots=True)
class LoweredLaunch:
    argv: tuple[str, ...]
    environment: dict[str, str]
    request: dict


def lower_launch(candidate, model, binding, port, output_dir, *, inherited=None):
    w, settings = candidate.identities.workload, candidate.settings
    require_int(port, 'owned port')
    if port > 65535:
        raise ValueError('invalid owned port')
    env = {name: value for name, value in (os.environ if inherited is None else inherited).items()
           if not name.upper().startswith(('EXPERTFLOW', 'LLAMA_EXPERTFLOW', 'LLAMA_ARG_', 'GGML_'))}
    prefixes = [str(Path(binding.server.path).parent)]
    if binding.cuda_runtime:
        prefixes.append(str(Path(binding.cuda_runtime.path).parent))
    env['PATH'] = os.pathsep.join(prefixes + [env.get('PATH', '')])
    if settings.cuda_graphs == 'off':
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    if settings.static:
        env['LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER'] = ','.join(map(str, settings.static.layer_ids))
        env['LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE'] = '1'
    argv = [binding.server.path, '-m', model.path, '-ngl', str(settings.gpu_layers),
            '-c', str(w.context_size), '-b', str(settings.batch_size), '-ub', str(settings.microbatch_size),
            '-t', str(w.threads), '-tb', str(w.threads), '-ctk', settings.kv_type_k, '-ctv', settings.kv_type_v,
            '-np', '1', '--host', '127.0.0.1', '--port', str(port)]
    if settings.cpu_moe:
        argv.append('--cpu-moe')
    request = {'n_predict': w.predict_tokens, 'seed': w.seed, 'temperature': w.temperature,
               'ignore_eos': w.ignore_eos, 'cache_prompt': False, 'return_tokens': True, 'stream': False}
    return LoweredLaunch(tuple(argv), env, request)


def _write(path, payload):
    # Keep artifact bytes unchanged by this checkout's JSON eol=lf rule.
    Path(path).write_text(json.dumps(canonical_payload(payload), indent=2, sort_keys=True) + '\n',
                          encoding='utf-8', newline='\n')


def _http(port, route, payload=None, timeout=1):
    body = None if payload is None else canonical_json(payload).encode('utf-8')
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    expired = threading.Event()
    owned_socket = []
    def expire():
        expired.set()
        if owned_socket:
            try:
                owned_socket[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        connection.connect()
        owned_socket.append(connection.sock)
        if expired.is_set():
            raise TimeoutError('absolute connection deadline exceeded')
        connection.request('GET' if payload is None else 'POST', route, body,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise OSError(f'native HTTP status {response.status}')
        chunks = []
        size = 0
        while True:
            if expired.is_set():
                raise TimeoutError(f'{route} absolute response deadline exceeded')
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > 16 * 1024 * 1024:
                raise ValueError('native response exceeds bounded JSON size')
        if expired.is_set():
            raise TimeoutError(f'{route} absolute response deadline exceeded')
        value = json.loads(b''.join(chunks))
    except Exception as error:
        if expired.is_set():
            raise TimeoutError(f'{route} absolute response deadline exceeded') from error
        raise
    finally:
        watchdog.cancel()
        connection.close()
    if not isinstance(value, dict):
        raise ValueError('server response is not a JSON object')
    return value


def _process_creation(process):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(wintypes.HANDLE(int(process._handle)), *(ctypes.byref(v) for v in values)):
            raise RuntimeError('owned process creation time unavailable')
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime, 'GetProcessTimes'
    fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19]), 'proc-start-ticks'


def _owns_port(pid, port):
    if os.name != 'nt':
        # The supported live launcher is Windows. Integration tests elsewhere
        # still require the child's socket inode to appear in its own fd table.
        target = f'{port:04X}'
        inodes = set()
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            parts = line.split()
            if parts[1].split(':')[1] == target and parts[3] == '0A':
                inodes.add(parts[9])
        return any(p.is_symlink() and os.readlink(p) in {f'socket:[{inode}]' for inode in inodes}
                   for p in Path(f'/proc/{pid}/fd').iterdir())
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('iphlpapi')
    api.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                       wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    size = wintypes.DWORD()
    api.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)
    if not size.value:
        return False
    buffer = ctypes.create_string_buffer(size.value)
    if api.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 3, 0):
        raise RuntimeError('owned port table unavailable')
    class Row(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in
                    ('state', 'local_addr', 'local_port', 'remote_addr', 'remote_port', 'pid')]
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    rows = ctypes.cast(ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD), ctypes.POINTER(Row))
    return any(rows[n].pid == pid and socket.ntohs(rows[n].local_port & 0xffff) == port for n in range(count))


@dataclass(frozen=True, slots=True)
class MeasurementOutcome:
    status: str
    measurement_id: str | None
    reason: str | None
    resolved_placement: str | None
    output_dir: str


def compute_activity_by_pid(items):
    valid = [(name, value) for name, status, value in items if status in (0, 1)]
    if not valid:
        raise RuntimeError('GPU engine telemetry unavailable')
    result = {}
    for name, value in valid:
        match = re.match(r'pid_(\d+)_.*engtype_(?:Compute|CUDA)', name)
        if match and value > 0:
            pid = int(match.group(1))
            result[pid] = result.get(pid, 0.0) + value
    return result


class WindowsGpuMemorySampler:
    """English PDH counter names, owned PID, plus device-wide free reserve."""
    def __init__(self, gpu_uuid):
        if os.name != 'nt':
            raise RuntimeError('process-owned WDDM memory requires Windows PDH')
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.pdh = ctypes.WinDLL('pdh')
        self.query = wintypes.HANDLE()
        self.counter = wintypes.HANDLE()
        self.compute_counter = wintypes.HANDLE()
        self.gpu_uuid = gpu_uuid
        self.nvml = ctypes.WinDLL('nvml.dll')
        if self.nvml.nvmlInit_v2():
            raise RuntimeError('NVML memory telemetry unavailable')
        self.device = ctypes.c_void_p()
        self.nvml.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        if self.nvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode(), ctypes.byref(self.device)):
            raise RuntimeError('NVML GPU UUID unavailable')
        class Memory(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('total', 'free', 'used')]
        self.Memory = Memory
        self.nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]
        class ValueUnion(ctypes.Union):
            _fields_ = [('largeValue', ctypes.c_longlong), ('doubleValue', ctypes.c_double)]
        class Value(ctypes.Structure):
            _anonymous_ = ('value',)
            _fields_ = [('CStatus', wintypes.DWORD), ('value', ValueUnion)]
        class Item(ctypes.Structure):
            _fields_ = [('szName', wintypes.LPWSTR), ('FmtValue', Value)]
        self.Item = Item
        self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
            raise RuntimeError('PDH query unavailable')
        if self.pdh.PdhAddEnglishCounterW(self.query, r'\GPU Process Memory(*)\Dedicated Usage', 0, ctypes.byref(self.counter)):
            self.close()
            raise RuntimeError('GPU Process Memory counter unavailable')
        if self.pdh.PdhAddEnglishCounterW(self.query, r'\GPU Engine(*)\Utilization Percentage', 0, ctypes.byref(self.compute_counter)):
            self.close()
            raise RuntimeError('GPU Engine counter unavailable')

    def check_idle(self):
        self.pdh.PdhCollectQueryData(self.query)
        readings = []
        for _ in range(3):
            time.sleep(0.2)
            if self.pdh.PdhCollectQueryData(self.query):
                raise RuntimeError('GPU engine collection failed')
            c = self.ctypes
            from ctypes import wintypes
            size, count = wintypes.DWORD(), wintypes.DWORD()
            self.pdh.PdhGetFormattedCounterArrayW(self.compute_counter, 0x200, c.byref(size), c.byref(count), None)
            if not size.value:
                raise RuntimeError('GPU engine telemetry unavailable')
            buffer = c.create_string_buffer(size.value)
            if self.pdh.PdhGetFormattedCounterArrayW(self.compute_counter, 0x200, c.byref(size), c.byref(count), buffer):
                raise RuntimeError('GPU engine telemetry unavailable')
            items = c.cast(buffer, c.POINTER(self.Item))
            activity = compute_activity_by_pid([(items[n].szName, items[n].FmtValue.CStatus,
                                                 items[n].FmtValue.doubleValue) for n in range(count.value)])
            readings.append(activity)
            if any(rate >= 1 for rate in activity.values()):
                raise RuntimeError(f'another compute workload is active: {activity}')
        return {'compute_activity_pct_by_pid': readings, 'sampling_period_seconds': 0.2}

    def __call__(self, pid):
        c = self.ctypes
        from ctypes import wintypes
        if self.pdh.PdhCollectQueryData(self.query):
            raise RuntimeError('PDH collection failed')
        size, count = wintypes.DWORD(), wintypes.DWORD()
        self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x400, c.byref(size), c.byref(count), None)
        if not size.value:
            return {'pid': pid, 'state': 'unavailable', 'counter_available': False}
        buffer = c.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x400, c.byref(size), c.byref(count), buffer):
            raise RuntimeError('PDH formatted memory unavailable')
        items = c.cast(buffer, c.POINTER(self.Item))
        if not count.value or not any(items[n].FmtValue.CStatus in (0, 1) for n in range(count.value)):
            return {'pid': pid, 'state': 'unavailable', 'counter_available': False}
        matching = [items[n] for n in range(count.value) if items[n].szName.startswith(f'pid_{pid}_')]
        if any(item.FmtValue.CStatus not in (0, 1) for item in matching):
            return {'pid': pid, 'state': 'unavailable', 'counter_available': False}
        owned = sum(item.FmtValue.largeValue for item in matching)
        memory = self.Memory()
        if self.nvml.nvmlDeviceGetMemoryInfo(self.device, c.byref(memory)):
            raise RuntimeError('device free memory unavailable')
        return {'pid': pid, 'state': 'allocated' if owned > 0 else 'absent', 'counter_available': True,
                'dedicated_bytes': owned, 'device_free_bytes': memory.free,
                'counter_instances': [item.szName for item in matching]}

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


class ServerMeasurementRunner:
    def __init__(self, store, *, process_factory=subprocess.Popen, http_request=_http,
                 memory_sampler=None, sample_interval=0.2, teardown_timeout_seconds=30):
        self.store, self.process_factory, self.http_request = store, process_factory, http_request
        self.memory_sampler, self.sample_interval = memory_sampler, sample_interval
        self.teardown_timeout_seconds = teardown_timeout_seconds

    def run_once(self, candidate, model, binding, *, output_dir, measured, stage='initial',
                 numerical_path='stock_same_runtime', comparison_ids=(), host_environment=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        w = candidate.identities.workload
        process = None
        monitor = None
        stop = threading.Event()
        samples, observations, sample_errors = [], [], []
        active = threading.Event()
        sampling_lock = threading.Lock()
        run_start = {}
        teardown_reading = None
        reason = None
        status = 'measured'
        request_completed = False
        forced_kill = False
        child_exit = None
        phases = {'load_health_ms': None, 'tokenize_ms': None, 'teardown_ms': None}
        # Reserve an available loopback port; any racing foreign bind makes our
        # child fail, rather than giving permission to stop that foreign PID.
        with socket.socket() as port_socket:
            port_socket.bind(('127.0.0.1', 0))
            port = port_socket.getsockname()[1]
        if not isinstance(model, ModelIR) or canonical_sha256(model) != candidate.identities.model_sha256:
            raise ValueError('model snapshot identity mismatch')
        launch = lower_launch(candidate, model.identity, binding, port, output_dir)
        host_binding = {}
        if host_environment is not None:
            from .preflight import capture_host_environment
            actual_host = capture_host_environment()
            if canonical_payload(actual_host) != canonical_payload(host_environment):
                raise ValueError('native launch host environment mismatch')
            host_binding = {'host_environment': actual_host}
        _write(output_dir / 'launch.json', {
            **host_binding,
            'argv': launch.argv, 'environment': {k: v for k, v in launch.environment.items()
                if k.startswith(('EXPERTFLOW', 'LLAMA_EXPERTFLOW', 'GGML_')) or k == 'PATH'},
            'runtime_binding': binding, 'candidate_id': candidate.candidate_id,
            'model_ir': model,
            'settings_sha256': canonical_sha256(candidate.settings),
            'runtime_sha256': candidate.identities.runtime_sha256,
            'model_sha256': candidate.identities.model_sha256,
            'workload_sha256': candidate.identities.workload_sha256,
        })

        def sample_once():
            with sampling_lock:
                phase = 'measurement' if active.is_set() else 'startup'
                try:
                    sample = self.memory_sampler(process.pid) if self.memory_sampler else None
                    if not isinstance(sample, dict):
                        sample = {'pid': process.pid, 'state': 'unavailable', 'counter_available': False}
                    observation = {**sample, 'phase': phase, 'time_monotonic': time.monotonic()}
                    observations.append(observation)
                    if sample.get('counter_available') is not True or sample.get('state') == 'unavailable':
                        raise RuntimeError('unknown process-owned memory')
                    if sample.get('state') == 'absent' and phase == 'startup':
                        return
                    if sample.get('state') != 'allocated' or type(sample.get('dedicated_bytes')) is not int or sample['dedicated_bytes'] <= 0:
                        raise RuntimeError('unknown process-owned memory during measurement')
                    samples.append(observation)
                except Exception as error:
                    sample_errors.append(str(error))

        def sample_loop():
            while not stop.is_set():
                tick = time.monotonic()
                sample_once()
                stop.wait(max(0, self.sample_interval - (time.monotonic() - tick)))

        try:
            binding.verify()
            if binding.sha256 != candidate.identities.runtime_sha256:
                raise ValueError('runtime binding identity mismatch')
            if hasattr(self.memory_sampler, 'check_idle'):
                _write(output_dir / 'idle-compute.json', self.memory_sampler.check_idle())
            with (output_dir / 'stdout.log').open('wb') as stdout, (output_dir / 'stderr.log').open('wb') as stderr:
                load_started = time.perf_counter()
                process = self.process_factory(list(launch.argv), env=launch.environment,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                creation, source = _process_creation(process)
                run_start = {'pid': process.pid, 'run_id': str(uuid.uuid4()), 'creation_time_100ns': creation,
                             'creation_source': source, 'started_monotonic_ns': time.monotonic_ns()}
                _write(output_dir / 'run-start.json', run_start)
                monitor = threading.Thread(target=sample_loop, daemon=True)
                monitor.start()
                deadline = time.monotonic() + w.health_timeout_seconds
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'server exited before health: {process.returncode}')
                    try:
                        if not _owns_port(process.pid, port):
                            raise OSError('owned server is not listening yet')
                        self.http_request(port, '/health', timeout=min(1, max(0.01, deadline - time.monotonic())))
                        phases['load_health_ms'] = (time.perf_counter() - load_started) * 1000
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('server health timeout')
                        stop.wait(0.1)
                active.set()
                sample_once()
                tokenize_request = {'content': w.prompt, 'add_special': True}
                _write(output_dir / 'tokenize-request.json', tokenize_request)
                tokenize_started = time.perf_counter()
                tokenize = self.http_request(port, '/tokenize', tokenize_request,
                                             timeout=w.completion_timeout_seconds)
                phases['tokenize_ms'] = (time.perf_counter() - tokenize_started) * 1000
                _write(output_dir / 'tokenize.json', tokenize)
                request = {**launch.request, 'prompt': tokenize.get('tokens')}
                _write(output_dir / 'request.json', request)
                started = time.perf_counter_ns()
                try:
                    completion = self.http_request(port, '/completion', request, timeout=w.completion_timeout_seconds)
                finally:
                    finished = time.perf_counter_ns()
                    _write(output_dir / 'completion-wall.json', {'started_monotonic_ns': started,
                        'finished_monotonic_ns': finished, 'elapsed_ms': (finished - started) / 1e6})
                _write(output_dir / 'completion.json', completion)
                request_completed = True
                # Ensure at least one post-response sample even for tiny tests.
                sample_once()
        except ValueError as error:
            status, reason = 'validation_stop', str(error)
        except Exception as error:
            status, reason = 'environment_blocked', str(error)
        finally:
            teardown_started = time.perf_counter()
            stop.set()
            if monitor:
                monitor.join(timeout=10)
                if monitor.is_alive():
                    status, reason = 'environment_blocked', 'memory monitor teardown failed'
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        forced_kill = True
                        process.kill()
                        process.wait(timeout=30)
                child_exit = process.returncode
            settled = False
            if process and self.memory_sampler:
                deadline = time.monotonic() + self.teardown_timeout_seconds
                while time.monotonic() < deadline:
                    try:
                        remaining = self.memory_sampler(process.pid)
                        teardown_reading = remaining
                        if isinstance(remaining, dict) and remaining.get('counter_available') is True and remaining.get('state') == 'absent' and remaining.get('dedicated_bytes') == 0:
                            settled = True
                            break
                    except Exception as error:
                        teardown_reading = {'state': 'unavailable', 'reason': str(error)}
                    time.sleep(self.sample_interval)
            _write(output_dir / 'memory.json', {'samples': samples, 'observations': observations, 'errors': sample_errors,
                                                'teardown_reading': teardown_reading,
                                                'sample_interval_seconds': self.sample_interval})
            cleanup = process is not None and process.poll() is not None and not forced_kill and settled
            phases['teardown_ms'] = (time.perf_counter() - teardown_started) * 1000
            _write(output_dir / 'phase-timing.json', phases)
            _write(output_dir / 'process.json', {'pid': process.pid if process else None,
                'exited': cleanup, 'exit_code': 0 if request_completed and cleanup else child_exit,
                'child_exit_code': child_exit, 'cleanup': cleanup,
                'owned_termination': request_completed and cleanup, 'forced_kill': forced_kill,
                'memory_settled': settled, 'run_id': run_start.get('run_id'),
                'creation_time_100ns': run_start.get('creation_time_100ns')})
        mid = None
        if status == 'measured':
            if not samples or sample_errors or not settled:
                status, reason = 'environment_blocked', 'process-owned memory unavailable: ' + '; '.join(sample_errors)
            else:
                try:
                    completion = json.loads((output_dir / 'completion.json').read_text())
                    timing = completion.get('timings', {})
                    count, milliseconds = timing.get('predicted_n'), timing.get('predicted_ms')
                    require_int(count, 'native predicted_n')
                    from .schema import require_number
                    require_number(milliseconds, 'native predicted_ms', 0.000001)
                    artifacts = []
                    for role in ('launch', 'request', 'tokenize-request', 'tokenize', 'completion', 'memory', 'process', 'run-start', 'completion-wall', 'phase-timing'):
                        p = (output_dir / f'{role}.json').resolve()
                        artifacts.append(EvidenceArtifact(role, ArtifactIdentity(str(p), p.stat().st_size, file_sha256(p))))
                    record = MeasurementRecord(MeasurementKey.from_candidate(candidate), candidate.candidate_id,
                        canonical_json(canonical_payload(candidate.identities)), canonical_json(canonical_payload(candidate.settings)),
                        tuple(artifacts), canonical_json({'decode_tps': count * 1000 / milliseconds}),
                        canonical_json({'exact_tokens': True, 'memory': True, 'cleanup': True}), 0, measured,
                        stage, numerical_path, tuple(comparison_ids))
                    mid = self.store.append_measurement(record)
                    self.store.verify_measurement(mid)
                except ValueError as error:
                    status, reason = 'validation_stop', str(error)
        if status != 'measured':
            _write(output_dir / 'failure.json', {'status': status, 'reason': reason, 'measurement_id': mid})
        text = (output_dir / 'stderr.log').read_text(encoding='utf-8', errors='replace') if (output_dir / 'stderr.log').exists() else ''
        offloaded = re.findall(r'offloaded (\d+)/(\d+) layers', text)
        placement = canonical_sha256({'offloaded': offloaded[-1], 'cpu_moe': candidate.settings.cpu_moe}) if offloaded else None
        outcome = MeasurementOutcome(status, mid, reason, placement, str(output_dir.resolve()))
        _write(output_dir / 'outcome.json', outcome)
        return outcome
