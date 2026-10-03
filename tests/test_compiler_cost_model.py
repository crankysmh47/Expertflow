from dataclasses import replace

import pytest

from expertflow.compiler.cost_model import CalibrationState, CostEstimate, select_measurement_batch, update_calibration


def estimates():
    return tuple(CostEstimate(str(i), (('decode_tps', tps),), uncertainty_pct=unc,
                              constraint_margin=margin, key_sha256='a' * 64)
                 for i, tps, unc, margin in [(0, 30, 1, 100), (1, 25, 2, 0.1),
                                             (2, 24, 50, 50), (3, 10, 1, 100), (4, 9, 1, 100)])


def test_exploration_quota_and_deterministic_exhaustive_selection():
    decision = select_measurement_batch(estimates(), budget=4, sentinel_period=1)
    assert {s.reason for s in decision.selected} == {
        'predicted_winner', 'constraint_boundary', 'max_uncertainty', 'sentinel'}
    assert len({s.candidate_id for s in decision.selected}) == 4
    assert decision == select_measurement_batch(tuple(reversed(estimates())), budget=4, sentinel_period=1)
    exhaustive = select_measurement_batch(estimates(), budget=5)
    assert len(exhaustive.selected) == 5
    assert exhaustive.pruned == ()


def test_sentinel_residual_suspends_pruning_and_is_key_scoped():
    state = CalibrationState.empty(error_threshold_pct=10, key_sha256='a' * 64)
    state = update_calibration(state, predicted={'decode_tps': 30}, measured={'decode_tps': 20}, candidate_role='sentinel')
    assert state.pruning_suspended and state.empirical_error_pct == 50
    decision = select_measurement_batch(estimates(), budget=4, calibration=state, sentinel_period=1)
    assert decision.pruned == () and decision.status == 'inconclusive'
    with pytest.raises(ValueError, match='key'):
        update_calibration(state, predicted={'decode_tps': 20}, measured={'decode_tps': 20},
                           candidate_role='sentinel', key_sha256='b' * 64)
    with pytest.raises(ValueError, match='key'):
        select_measurement_batch([replace(estimates()[0], key_sha256='b' * 64)], 1, calibration=state)


def test_invalid_metrics_budget_and_hard_constraints():
    with pytest.raises(ValueError):
        replace(estimates()[0], uncertainty_pct=float('nan'))
    with pytest.raises(ValueError):
        replace(estimates()[0], metrics=(('decode_tps', float('inf')),))
    with pytest.raises(ValueError):
        select_measurement_batch(estimates(), budget=0)
    bad = replace(estimates()[0], feasible=False)
    decision = select_measurement_batch([bad, estimates()[1]], budget=2)
    assert decision.rejected == ('0',)
    assert [s.candidate_id for s in decision.selected] == ['1']
    assert select_measurement_batch(estimates(), budget=1).status == 'inconclusive'
