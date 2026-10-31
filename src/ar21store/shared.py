"""Immutable bank aliasing and actual-storage accounting across video objects."""
import copy

from r15kv.persistent_registry import storages


class SharedVideoStore:
    def __init__(self, owner, max_objects=4):
        assert max_objects > 0
        assert getattr(owner, "plan", None) is None
        self.owner = owner
        self.max_objects = max_objects
        self.budget_bytes = max_objects * owner.budget_bytes
        self.caches = {}
        self.peak_bytes = 0
        self.events = []
        self._versions = {k: v._version for table in owner.tables for k, v in table.items()}
        self.bank_bytes = owner.static_bytes

    def fork_codec(self):
        if len(self.caches) >= self.max_objects:
            raise RuntimeError("STORE_OBJECT_CAPACITY_REACHED_BEFORE_GPU_ALLOCATION")
        codec = copy.copy(self.owner)
        codec.policy = dict(self.owner.policy)
        codec.metadata = {**self.owner.metadata, "policy": codec.policy,
                          "bank_load_scope": "aliases immutable owner tensors; no new bank CUDA allocation"}
        codec.context = {}
        codec.registry = None
        codec.plan = None
        codec.load_ms = 0.0
        assert codec.tables is self.owner.tables
        return codec

    def attach(self, name, cache):
        assert name not in self.caches and len(self.caches) < self.max_objects
        assert cache.codec.tables is self.owner.tables
        self.caches[name] = cache
        return self.observe("attach:" + name)

    def observe(self, event):
        roots = {"shared-bank": (self.owner.tables, "codec")}
        individual = []
        for name, cache in self.caches.items():
            cache.codec.registry.observe("store-object-check")
            individual.append(cache.registry.end)
            assert cache.registry.end <= self.owner.budget_bytes
            for key, value in cache.registry.roots.items():
                roots[name + "/" + key] = value
        arrays = [x for x in storages(roots) if x["device"].startswith("cuda")]
        total = sum(x["bytes"] for x in arrays)
        expected = sum(individual) - max(0, len(individual) - 1) * self.bank_bytes if individual else self.bank_bytes
        assert total == expected, (total, expected)
        assert total <= self.budget_bytes
        assert self._versions == {k: v._version for table in self.owner.tables for k, v in table.items()}
        self.peak_bytes = max(self.peak_bytes, total)
        receipt = {"event": event, "objects": len(self.caches), "resident_bytes": total,
                   "peak_bytes": self.peak_bytes, "budget_bytes": self.budget_bytes,
                   "shared_bank_unique_bytes": self.bank_bytes,
                   "sum_independent_object_receipts_bytes": sum(individual),
                   "avoided_duplicate_bank_bytes": max(0, len(individual) - 1) * self.bank_bytes,
                   "budget_pass": True}
        self.events.append(receipt)
        return receipt

    def release(self, name):
        cache = self.caches.pop(name)
        cache.codec.finish_request()
        del cache
        return self.observe("release:" + name)
