from dataclasses import replace

import pytest

from expertflow.compiler.passes.base import CompilerState, PassManager, PassResult


class Pass:
    def __init__(self, name, requires=(), provides=(), conflicts=(), emitted=None):
        self.name = name
        self.requires, self.provides, self.conflicts = map(frozenset, (requires, provides, conflicts))
        self.emitted = self.provides if emitted is None else frozenset(emitted)

    def run(self, state):
        return PassResult(replace(state, diagnostics=state.diagnostics + (self.name,)), self.emitted)


def test_orders_capabilities_deterministically_and_preserves_diagnostics():
    passes = [Pass('placement', ['baseline'], ['placement']), Pass('baseline', ['inspect'], ['baseline']),
              Pass('inspect', provides=['inspect']), Pass('aaa')]
    manager = PassManager(passes)
    assert [p.name for p in manager.resolve()] == ['aaa', 'inspect', 'baseline', 'placement']
    initial = CompilerState(None, None, None, diagnostics=('original',))
    result = manager.run(initial)
    assert result.diagnostics == ('original', 'aaa', 'inspect', 'baseline', 'placement')
    assert initial.diagnostics == ('original',)
    assert result.capabilities == frozenset({'inspect', 'baseline', 'placement'})


def test_missing_cycle_conflict_duplicates():
    with pytest.raises(ValueError, match='missing capability'):
        PassManager([Pass('a', requires=['missing'])]).resolve()
    with pytest.raises(ValueError, match='cycle'):
        PassManager([Pass('a', ['b'], ['a']), Pass('b', ['a'], ['b'])]).resolve()
    with pytest.raises(ValueError, match='conflict'):
        PassManager([Pass('a', conflicts=['b']), Pass('b')]).resolve()
    with pytest.raises(ValueError, match='duplicate'):
        PassManager([Pass('a'), Pass('a')]).resolve()
    with pytest.raises(ValueError, match='ambiguous'):
        PassManager([Pass('a', provides=['same']), Pass('b', provides=['same'])]).resolve()


def test_undeclared_output_and_noop_missing_output():
    state = CompilerState(None, None, None)
    with pytest.raises(ValueError, match='undeclared'):
        PassManager([Pass('a', emitted=['secret'])]).run(state)
    with pytest.raises(ValueError, match='not produced'):
        PassManager([Pass('a', provides=['a'], emitted=[]), Pass('b', requires=['a'])]).run(state)
    no_op = PassManager([Pass('no-op')]).run(state)
    assert no_op.diagnostics == ('no-op',)


def test_analyses_are_copied_to_canonical_immutable_payloads():
    values = {'x': [1]}
    state = CompilerState(None, None, None).with_analysis('cost', values)
    values['x'].append(2)
    assert state.analysis('cost') == {'x': [1]}
    copy = state.analysis('cost')
    copy['x'].clear()
    assert state.analysis('cost') == {'x': [1]}
