"""Reuse verified readers within one validation, without caching its verdicts."""
from contextlib import contextmanager
from pathlib import Path

from expertflow.compiler.evidence import EvidenceStore


class EvidenceReaders:
    def __init__(self):
        self._stores = {}

    def __call__(self, path):
        key = Path(path).resolve()
        if key not in self._stores:
            self._stores[key] = EvidenceStore(key)
        return self._stores[key]

    @property
    def database_count(self):
        return len(self._stores)

    @property
    def model_verification_count(self):
        return sum(len(store._model_verifications) for store in self._stores.values())


@contextmanager
def reuse_readers(*modules):
    """Scope legacy driver constructor aliases to one synchronous CLI call.

    EvidenceStore itself and every verify_measurement call remain unchanged.
    Its size/mtime/ctime guard still runs before using a model digest cache.
    No reader survives this context and aliases restore even on failure.
    """
    readers = EvidenceReaders()
    originals = [(module, module.EvidenceStore) for module in modules]
    try:
        for module, _ in originals:
            module.EvidenceStore = readers
        yield readers
    finally:
        for module, constructor in originals:
            module.EvidenceStore = constructor
