"""Per-(user, agent) working-memory blackboard. L1 of the hierarchy.

Volatile, in-process. Holds an activation map, a bounded working set, and the last cascade
trace used by Stage 6 reflection.
"""
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List


WORKING_SET_CAP = 50
ACTIVATION_FLOOR = 1e-3
DECAY_LAMBDA = 1.0 / 600.0  # half-life ≈ 7 minutes


@dataclass
class RetrievalTrace:
    """Snapshot of the most recent cascade output, used by Stage 2 working-set expansion."""
    query: str = ""
    used_memory_ids: List[str] = field(default_factory=list)


class Blackboard:
    """Per-agent working memory. Thread-safe via internal lock."""

    def __init__(self, user_id: str, agent_name: str):
        self.user_id = user_id
        self.agent_name = agent_name
        self.activation_map: Dict[str, float] = {}
        self.working_set: "OrderedDict[str, dict]" = OrderedDict()
        self.last_cascade: RetrievalTrace = RetrievalTrace()
        self._lock = threading.RLock()

    # ----- activation -----

    def activate(self, memory_id: str, delta: float):
        with self._lock:
            cur = self.activation_map.get(memory_id, 0.0)
            self.activation_map[memory_id] = min(1.0, cur + delta)

    def get_activation(self, memory_id: str) -> float:
        with self._lock:
            return self.activation_map.get(memory_id, 0.0)

    def decay(self, dt_seconds: float, lam: float = DECAY_LAMBDA):
        """Exponential decay sweep. Removes items below floor."""
        if dt_seconds <= 0:
            return
        factor = math.exp(-lam * dt_seconds)
        with self._lock:
            drops = []
            for mid, a in self.activation_map.items():
                new_a = a * factor
                if new_a < ACTIVATION_FLOOR:
                    drops.append(mid)
                else:
                    self.activation_map[mid] = new_a
            for mid in drops:
                del self.activation_map[mid]
                self.working_set.pop(mid, None)

    # ----- working set -----

    def admit(self, memory: dict, strength: float = 1.0):
        """Place a memory in the working set with a starting activation; evict lowest if over cap."""
        mid = memory.get("memory_id") or memory.get("_id")
        if not mid:
            return
        with self._lock:
            self.working_set[mid] = memory
            self.working_set.move_to_end(mid)
            self.activate(mid, strength)
            while len(self.working_set) > WORKING_SET_CAP:
                self._evict_lowest_activation()

    def _evict_lowest_activation(self):
        # Internal — caller holds _lock.
        if not self.working_set:
            return
        lowest_id = min(
            self.working_set,
            key=lambda mid: self.activation_map.get(mid, 0.0),
        )
        self.working_set.pop(lowest_id, None)

    def working_set_seed(self) -> Dict[str, float]:
        """Return the working set as {memory_id: activation}."""
        with self._lock:
            return {
                mid: self.activation_map.get(mid, 0.0)
                for mid in self.working_set
            }

    def recently_co_activated(self, memory_id: str) -> List[str]:
        """Other memory_ids that appeared in the same recent cascade as the given one."""
        if memory_id not in self.last_cascade.used_memory_ids:
            return []
        return [
            m for m in self.last_cascade.used_memory_ids
            if m != memory_id
        ]
