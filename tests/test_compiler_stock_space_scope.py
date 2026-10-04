"""A prior Q6 exclusion must not leak into another model's default coverage."""

import pytest

from test_compiler_stock_q4_eligibility import fixtures
from test_compiler_stock_search import candidate, host


def test_q4_requires_explicit_space_instead_of_inheriting_q6_thread_exclusion():
    from expertflow.compiler.stock_discovery import _space
    inp, _ = fixtures()
    with pytest.raises(ValueError, match='explicit'):
        _space(candidate(), host(), model=inp.model)


def test_explicit_q4_space_keeps_all_anchors_without_q6_exclusion():
    from expertflow.compiler.stock_discovery import _space
    inp, _ = fixtures()
    config = {'policy': 'explicit', 'excluded_threads': [], 'maximum_native_processes': 38}
    space, actual = _space(candidate(), host(), config=config, model=inp.model)
    assert actual == config
    assert space.excluded_threads == ()
    assert {c.identities.workload.threads for c in space.candidates} == {8, 12, 16}
    assert len(space.candidates) == 6
