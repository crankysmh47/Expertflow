"""Deterministic capability scheduling over immutable compiler state."""

from dataclasses import dataclass, replace
import json
from typing import Protocol

from ..reference import canonical_json
from ..schema import HardwareIR, ModelIR, WorkloadIR, canonical_payload
from ..plan import CandidatePlan


@dataclass(frozen=True, slots=True)
class CompilerState:
    model: ModelIR | None
    hardware: HardwareIR | None
    workload: WorkloadIR | None
    candidates: tuple[CandidatePlan, ...] = ()
    analyses: tuple[tuple[str, str], ...] = ()
    diagnostics: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, 'candidates', tuple(self.candidates))
        object.__setattr__(self, 'diagnostics', tuple(self.diagnostics))
        object.__setattr__(self, 'capabilities', frozenset(self.capabilities))
        entries = tuple(sorted(tuple(x) for x in self.analyses))
        if len({name for name, _ in entries}) != len(entries):
            raise ValueError('duplicate analysis')
        for name, payload in entries:
            if not name or canonical_json(json.loads(payload)) != payload:
                raise ValueError('analysis must be canonical JSON')
        object.__setattr__(self, 'analyses', entries)

    def with_analysis(self, name, value):
        entries = dict(self.analyses)
        entries[name] = canonical_json(canonical_payload(value))
        return replace(self, analyses=tuple(entries.items()))

    def analysis(self, name):
        return json.loads(dict(self.analyses)[name])


@dataclass(frozen=True, slots=True)
class PassResult:
    state: CompilerState
    provided: frozenset[str] = frozenset()


class CompilerPass(Protocol):
    name: str
    requires: frozenset[str]
    provides: frozenset[str]
    conflicts: frozenset[str]

    def run(self, state: CompilerState) -> PassResult: ...


class PassManager:
    def __init__(self, passes):
        self.passes = tuple(passes)

    def resolve(self, initial_capabilities=frozenset()):
        names = [p.name for p in self.passes]
        if len(set(names)) != len(names):
            raise ValueError('duplicate pass names')
        providers = {}
        for p in self.passes:
            if p.conflicts.intersection(names) or p.conflicts.intersection(
                set(initial_capabilities).union(*(x.provides for x in self.passes))
            ):
                raise ValueError(f'pass conflict: {p.name}')
            for capability in p.provides:
                if capability in providers or capability in initial_capabilities:
                    raise ValueError(f'ambiguous capability provider: {capability}')
                providers[capability] = p.name
        available = set(initial_capabilities)
        known = available | providers.keys()
        for p in self.passes:
            if p.requires - known:
                raise ValueError(f'missing capability: {sorted(p.requires - known)}')
        pending = list(self.passes)
        ordered = []
        while pending:
            ready = sorted((p for p in pending if p.requires <= available), key=lambda p: p.name)
            if not ready:
                raise ValueError('compiler pass dependency cycle')
            p = ready[0]
            ordered.append(p)
            pending.remove(p)
            available.update(p.provides)
        return tuple(ordered)

    def run(self, state):
        for p in self.resolve(state.capabilities):
            if not p.requires <= state.capabilities:
                raise ValueError(f'required capability not produced for {p.name}')
            result = p.run(state)
            if not isinstance(result, PassResult) or not isinstance(result.state, CompilerState):
                raise ValueError(f'invalid pass result: {p.name}')
            provided = frozenset(result.provided)
            if provided - p.provides or result.state.capabilities - state.capabilities:
                raise ValueError(f'undeclared capability: {p.name}')
            if result.state.diagnostics[:len(state.diagnostics)] != state.diagnostics:
                raise ValueError(f'pass discarded diagnostics: {p.name}')
            state = replace(result.state, capabilities=state.capabilities | provided)
        return state
