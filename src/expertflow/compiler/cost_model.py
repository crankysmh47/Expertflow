"""Conservative estimates and a deterministic, bounded exploration quota."""

from dataclasses import dataclass, replace

from .schema import require_hash, require_int, require_number, require_text


@dataclass(frozen=True, slots=True)
class CostEstimate:
    candidate_id: str
    metrics: tuple[tuple[str, float], ...]
    uncertainty_pct: float
    constraint_margin: float
    key_sha256: str
    feasible: bool = True

    def __post_init__(self):
        require_text(self.candidate_id, 'candidate ID')
        require_hash(self.key_sha256)
        require_number(self.uncertainty_pct, 'uncertainty')
        require_number(self.constraint_margin, 'constraint margin')
        metrics = tuple(sorted(tuple(x) for x in self.metrics))
        if len(dict(metrics)) != len(metrics) or 'decode_tps' not in dict(metrics):
            raise ValueError('estimates require unique decode_tps metric')
        for name, value in metrics:
            require_text(name, 'metric name')
            require_number(value, name, 0.000001)
        if type(self.feasible) is not bool:
            raise ValueError('feasibility must be boolean')
        object.__setattr__(self, 'metrics', metrics)


@dataclass(frozen=True, slots=True)
class CalibrationState:
    key_sha256: str
    error_threshold_pct: float
    empirical_error_pct: float = 0.0
    observations: int = 0
    pruning_suspended: bool = False

    def __post_init__(self):
        require_hash(self.key_sha256)
        require_number(self.error_threshold_pct, 'calibration threshold', 0.000001)
        require_number(self.empirical_error_pct, 'empirical error')
        require_int(self.observations, 'observation count', 0)

    @classmethod
    def empty(cls, error_threshold_pct=10, key_sha256='0' * 64):
        return cls(key_sha256, error_threshold_pct)


@dataclass(frozen=True, slots=True)
class MeasurementSelection:
    candidate_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    selected: tuple[MeasurementSelection, ...]
    pruned: tuple[str, ...]
    rejected: tuple[str, ...]
    status: str


def update_calibration(state, *, predicted, measured, candidate_role, key_sha256=None):
    if key_sha256 is not None and key_sha256 != state.key_sha256:
        raise ValueError('cross-key calibration is forbidden')
    if predicted.keys() != measured.keys() or not predicted:
        raise ValueError('calibration metric mismatch')
    error = 0.0
    for name, prediction in predicted.items():
        actual = measured[name]
        require_number(prediction, 'prediction', 0.000001)
        require_number(actual, 'measurement', 0.000001)
        error = max(error, abs(prediction - actual) * 100 / actual)
    # Worst observed residual, never an optimistic average over stale runs.
    worst = max(state.empirical_error_pct, error)
    return replace(state, empirical_error_pct=worst, observations=state.observations + 1,
                   pruning_suspended=state.pruning_suspended or (
                       candidate_role == 'sentinel' and error > state.error_threshold_pct))


def select_measurement_batch(estimates, budget, *, sentinel_period=1, iteration=0, calibration=None):
    require_int(budget, 'measurement budget')
    require_int(sentinel_period, 'sentinel period')
    require_int(iteration, 'iteration', 0)
    estimates = tuple(estimates)
    if len({e.candidate_id for e in estimates}) != len(estimates):
        raise ValueError('duplicate candidate estimates')
    keys = {e.key_sha256 for e in estimates}
    if len(keys) > 1 or calibration is not None and keys and keys != {calibration.key_sha256}:
        raise ValueError('cross-key candidate calibration')
    available = sorted((e for e in estimates if e.feasible), key=lambda e: e.candidate_id)
    rejected = tuple(sorted(e.candidate_id for e in estimates if not e.feasible))
    if len(available) <= budget:
        return CalibrationDecision(tuple(MeasurementSelection(e.candidate_id, 'exhaustive') for e in available),
                                   (), rejected, 'ready')
    selected = []
    error = calibration.empirical_error_pct if calibration else 0

    def take(reason, key):
        if available and len(selected) < budget:
            chosen = min(available, key=key)
            available.remove(chosen)
            selected.append(MeasurementSelection(chosen.candidate_id, reason))

    take('predicted_winner', lambda e: (-dict(e.metrics)['decode_tps'], e.candidate_id))
    take('constraint_boundary', lambda e: (e.constraint_margin, e.candidate_id))
    take('max_uncertainty', lambda e: (-max(e.uncertainty_pct, error), e.candidate_id))
    if iteration % sentinel_period == 0:
        # Deliberately samples a low-ranked estimate that optimistic pruning hides.
        take('sentinel', lambda e: (dict(e.metrics)['decode_tps'], e.candidate_id))
    while available and len(selected) < budget:
        take('exploration', lambda e: (-max(e.uncertainty_pct, error), e.candidate_id))
    suspended = calibration is not None and calibration.pruning_suspended
    threshold = calibration.error_threshold_pct if calibration else 10
    uncertain = any(max(e.uncertainty_pct, error) > threshold for e in estimates if e.feasible)
    # Unselected candidates remain unmeasured; this bounded phase never proves
    # they are safely dominated merely by an estimate.
    return CalibrationDecision(tuple(selected), (), rejected,
                               'inconclusive' if suspended or uncertain or budget < 4 else 'ready')
