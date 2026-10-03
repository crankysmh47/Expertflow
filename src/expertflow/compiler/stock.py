"""Bounded pristine-runtime search and conservative measured selection."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import statistics

from .plan import CandidatePlan, CandidateStatus, RuntimeSettings
from .schema import canonical_sha256, require_number


StockCandidate = CandidatePlan


def stock_candidate_matrix(identities):
    w = identities.workload
    return tuple(CandidatePlan(identities, RuntimeSettings(gpu_layers, cpu_moe, w.cuda_graphs,
                        w.kv_type_k, w.kv_type_v, w.batch_size, w.microbatch_size))
                 for gpu_layers in ('auto', 'all', 99) for cpu_moe in (False, True))


@dataclass(frozen=True, slots=True)
class StockMeasurement:
    candidate: CandidatePlan
    measurement_ids: tuple[str, ...]
    decode_tps: tuple[float, ...]
    exact: bool
    peak_bytes: int
    cleanup: bool
    exit_code: int
    status: str
    key_sha256: str

    @property
    def mean_tps(self):
        return statistics.mean(self.decode_tps)

    @property
    def cv_pct(self):
        return statistics.stdev(self.decode_tps) * 100 / self.mean_tps if len(self.decode_tps) > 1 else 0


def import_stock_measurement(candidate, measurement_ids, store):
    ids = tuple(measurement_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('missing or duplicate measurement evidence')
    rows = tuple(store.verify_measurement(mid) for mid in ids)
    if any(r['candidate_id'] != candidate.candidate_id or not r['measured'] for r in rows):
        raise ValueError('foreign candidate evidence or warmup')
    if len({r['generated_tokens_sha256'] for r in rows}) != 1 or len({r['prompt_tokens_sha256'] for r in rows}) != 1:
        raise ValueError('unstable exact tokens')
    peak = 0
    for mid in ids:
        memory_artifact = next(a for a in store.measurement(mid).artifacts if a.role == 'memory')
        memory = json.loads(Path(memory_artifact.identity.path).read_text())
        peak = max(peak, max(s['dedicated_bytes'] for s in memory['samples']))
    measured_candidate = replace(candidate, status=CandidateStatus.MEASURED, measurement_ids=ids)
    return StockMeasurement(measured_candidate, ids, tuple(r['decode_tps'] for r in rows),
                            True, peak, True, 0, 'measured', canonical_sha256(candidate.identities))


def select_strongest_stock(measurements, *, total_vram_mib, reserve_mib):
    measurements = tuple(measurements)
    identities = {canonical_sha256(m.candidate.identities) for m in measurements}
    if len(identities) > 1:
        raise ValueError('stock comparison identity mismatch')
    valid = []
    for m in measurements:
        w = m.candidate.identities.workload
        if m.key_sha256 != canonical_sha256(m.candidate.identities):
            raise ValueError('stock measurement key mismatch')
        for tps in m.decode_tps:
            require_number(tps, 'measured TPS', 0.000001)
        if m.status != 'measured' or m.exit_code != 0 or not m.exact or not m.cleanup or m.candidate.settings.static is not None:
            continue
        if len(m.measurement_ids) != len(m.decode_tps) or len(m.decode_tps) < w.measured_runs:
            continue
        if len(set(m.measurement_ids)) != len(m.measurement_ids):
            continue
        if m.peak_bytes <= 0 or m.peak_bytes + (reserve_mib << 20) > total_vram_mib << 20:
            continue
        if m.cv_pct > w.maximum_cv_pct:
            continue
        valid.append(m)
    if not valid:
        raise ValueError('inconclusive: no measured exact memory-safe stock candidate')
    return min(valid, key=lambda m: (-m.mean_tps, m.candidate.candidate_id))
