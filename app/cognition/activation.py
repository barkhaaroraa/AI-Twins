"""Pure activation kernel: merge/decay helpers used by the cascade."""
import math
from typing import Dict, Iterable


def decay(value: float, dt_seconds: float, lam: float = 1.0 / 600.0) -> float:
    return value * math.exp(-lam * dt_seconds)


def merge_seeds(*seed_maps: Dict[str, float]) -> Dict[str, float]:
    """Merge multiple {memory_id: strength} maps by max."""
    merged: Dict[str, float] = {}
    for m in seed_maps:
        for k, v in m.items():
            if v > merged.get(k, 0.0):
                merged[k] = v
    return merged


def expand_working_set(seeds: Dict[str, float], blackboard) -> Dict[str, float]:
    """Stage 2 priming: items already co-resident in the agent's working set get a head start."""
    expanded = dict(seeds)
    for mid, strength in seeds.items():
        for nb in blackboard.recently_co_activated(mid):
            expanded[nb] = max(expanded.get(nb, 0.0), 0.5 * strength)
    return expanded
