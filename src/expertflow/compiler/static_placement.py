"""Generic static-MoE diagnostics with aligned bytes and hard runtime limits."""

from dataclasses import dataclass, replace
import statistics

from .passes.base import PassResult
from .plan import CandidateStatus, StaticPlacement
from .schema import canonical_sha256, require_int, require_number


@dataclass(frozen=True, slots=True)
class LayerBenefit:
    layer_id: int
    median_total_us: float
    arena_bytes: int
    score_us_per_mib: float
    shadow_count: int = 4


@dataclass(frozen=True, slots=True)
class StaticPlacementCandidate:
    layer_ids: tuple[int, ...]
    arena_bytes: int
    shadow_count: int
    kind: str
    rejection_reasons: tuple[str, ...]

    @property
    def candidate_id(self):
        return canonical_sha256({'layers': self.layer_ids, 'arena_bytes': self.arena_bytes})


def rank_layer_benefits(model, profile_rows, *, minimum_repetitions=1, resident_layers=()):
    require_int(minimum_repetitions, 'profile repetitions')
    by_id = {layer.layer_id: layer for layer in model.moe_layers}
    samples = {layer: {} for layer in by_id}
    for row in profile_rows:
        layer_id = row.get('layer_id')
        require_int(layer_id, 'profile layer', 0)
        if layer_id not in by_id or row.get('backend') != 'CPU':
            raise ValueError('profile layer/backend mismatch')
        profile_id = row.get('profile_id', 'single')
        if profile_id in samples[layer_id]:
            raise ValueError('duplicate profile layer')
        require_number(row.get('total_us'), 'profile timing', 0.000001)
        samples[layer_id][profile_id] = row['total_us']
    if any(len(values) < minimum_repetitions for values in samples.values()):
        raise ValueError('missing profile layers/repetitions')
    profile_sets = {tuple(sorted(values)) for values in samples.values()}
    if len(profile_sets) != 1:
        raise ValueError('profile repetition set mismatch')
    benefits = []
    alignment = model.alignment_bytes
    for layer_id, layer in by_id.items():
        if layer_id in resident_layers:
            continue
        size = sum(((b + alignment - 1) // alignment) * alignment for b in layer.component_bank_bytes)
        median = statistics.median(samples[layer_id].values())
        benefits.append(LayerBenefit(layer_id, median, size, median * (1 << 20) / size))
    return tuple(sorted(benefits, key=lambda b: (-b.score_us_per_mib, b.layer_id)))


def generate_static_candidates(benefits, *, available_bytes, reserve_bytes, max_static_layers=12,
                               max_static_shadows=48, diagnostic_layers=(0,1,2,3,4,5,6,7,8,9,15,20)):
    require_int(available_bytes, 'available bytes', 0)
    require_int(reserve_bytes, 'reserve bytes', 0)
    require_int(max_static_layers, 'runtime layer cap', 0)
    require_int(max_static_shadows, 'runtime shadow cap', 0)
    benefits = tuple(sorted(benefits, key=lambda b: (-b.score_us_per_mib, b.layer_id)))
    by_id = {b.layer_id: b for b in benefits}
    if len(by_id) != len(benefits):
        raise ValueError('duplicate layer benefits')
    budget = max(0, available_bytes - reserve_bytes)
    result = [StaticPlacementCandidate((), 0, 0, 'static_off', ())]

    def candidate(ids, kind):
        ids = tuple(sorted(ids))
        arena = sum(by_id[i].arena_bytes for i in ids)
        shadows = sum(by_id[i].shadow_count for i in ids)
        reasons = []
        if arena > budget:
            reasons.append('vram_budget')
        if len(ids) > max_static_layers:
            reasons.append('runtime_layer_cap')
        if shadows > max_static_shadows:
            reasons.append('runtime_shadow_cap')
        reasons.append('numerical_path_change')
        return StaticPlacementCandidate(ids, arena, shadows, kind, tuple(reasons))

    prefix = []
    for benefit in benefits[:min(12, max_static_layers)]:
        proposed = candidate(prefix + [benefit.layer_id], 'greedy')
        if any(r != 'numerical_path_change' for r in proposed.rejection_reasons):
            if not prefix:
                result.append(proposed)
            break
        prefix.append(benefit.layer_id)
        result.append(proposed)
    if diagnostic_layers and set(diagnostic_layers) <= by_id.keys():
        diagnostic = candidate(diagnostic_layers, 'historical_diagnostic')
        if diagnostic.layer_ids not in {c.layer_ids for c in result}:
            result.append(diagnostic)
    return tuple(result)


class StaticPlacementPass:
    name = 'static-placement'
    requires = frozenset({'stock-baseline'})
    provides = frozenset({'static-placement'})
    conflicts = frozenset()

    def __init__(self, profile_rows, *, baseline_peak_bytes, max_static_layers=12, max_static_shadows=48):
        self.profile_rows = tuple(profile_rows)
        self.baseline_peak_bytes = baseline_peak_bytes
        self.max_static_layers, self.max_static_shadows = max_static_layers, max_static_shadows

    def run(self, state):
        stock = next((c for c in state.candidates if c.status is CandidateStatus.MEASURED and c.settings.static is None), None)
        if stock is None:
            raise ValueError('placement requires measured stock baseline')
        if not stock.settings.cpu_moe:
            diagnostics = state.diagnostics + ('stock experts already resident: no static duplication',)
            return PassResult(replace(state, diagnostics=diagnostics), self.provides)
        benefits = rank_layer_benefits(state.model, self.profile_rows, minimum_repetitions=3)
        candidates = generate_static_candidates(benefits,
            available_bytes=max(0, state.hardware.usable_vram_bytes - self.baseline_peak_bytes),
            reserve_bytes=state.hardware.minimum_reserve_bytes,
            max_static_layers=self.max_static_layers, max_static_shadows=self.max_static_shadows)
        generated = []
        for c in candidates:
            if not c.layer_ids or any(r != 'numerical_path_change' for r in c.rejection_reasons):
                continue
            generated.append(replace(stock, settings=replace(stock.settings, static=StaticPlacement(c.layer_ids, arena_bytes=c.arena_bytes)),
                                     status=CandidateStatus.REJECTED, measurement_ids=(),
                                     rejection_reasons=c.rejection_reasons))
        output = state.with_analysis('static_candidates', candidates)
        output = replace(output, candidates=output.candidates + tuple(generated),
                         diagnostics=output.diagnostics + ('static CPU-to-CUDA candidates rejected: numerical_path_change',))
        return PassResult(output, self.provides)
