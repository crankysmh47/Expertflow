"""Family adapters are the only boundary allowed to interpret raw model names."""

from dataclasses import dataclass
from typing import Mapping, Protocol

from ..schema import ArtifactIdentity, ModelIR, require_int, require_text


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    family: str
    architecture: str
    quantization: str
    expert_count: int
    expert_top_k: int
    kv_kind: str
    mtp_kind: str

    def __post_init__(self):
        for name in ('family', 'architecture', 'quantization', 'kv_kind', 'mtp_kind'):
            require_text(getattr(self, name), name)
        require_int(self.expert_count, 'expert_count')
        require_int(self.expert_top_k, 'expert_top_k')
        if self.expert_top_k > self.expert_count:
            raise ValueError('expert top_k exceeds count')


class ModelAdapter(Protocol):
    family: str

    def normalize(self, descriptor: ModelDescriptor, inventory: Mapping[str, object],
                  identity: ArtifactIdentity) -> ModelIR: ...


class AdapterRegistry:
    def __init__(self):
        self._adapters = {}

    def register(self, adapter: ModelAdapter):
        if adapter.family in self._adapters:
            raise ValueError(f'duplicate adapter: {adapter.family}')
        self._adapters[adapter.family] = adapter

    def resolve(self, family: str):
        try:
            return self._adapters[family]
        except KeyError as error:
            raise ValueError(f'unsupported model family: {family}') from error

    @classmethod
    def with_builtins(cls):
        from .gemma4 import Gemma4Adapter
        registry = cls()
        registry.register(Gemma4Adapter())
        return registry
