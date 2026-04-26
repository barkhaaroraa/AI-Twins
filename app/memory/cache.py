from collections import OrderedDict
from typing import Optional


class MemoryCache:
    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def invalidate_for_user(self, user_id: str) -> None:
        keys_to_remove = [k for k in self._cache if k.startswith(f"{user_id}:")]
        for k in keys_to_remove:
            del self._cache[k]

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }
