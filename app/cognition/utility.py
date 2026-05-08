"""Utility scoring per architecture §F.

utility = 0.30·recency + 0.25·frequency + 0.20·importance + 0.15·centrality
        + 0.10·lineage_value − 0.20·staleness − 0.30·conflict
Recomputed by the light consolidator hourly. Drives demotion and forgetting.
"""
import math
from datetime import datetime
from typing import Dict


def _recency_score(last_accessed_at) -> float:
    if not last_accessed_at:
        return 0.5
    try:
        delta = (datetime.utcnow() - last_accessed_at).total_seconds()
    except Exception:
        return 0.5
    # 1-week half-life.
    return float(math.exp(-delta / (7.0 * 24.0 * 3600.0)))


def _frequency_score(access_count: int) -> float:
    # log(1+x) capped to [0, 1] (~22 accesses → 1.0).
    return min(1.0, math.log1p(max(0, access_count)) / 4.0)


def utility(memory: Dict, centrality: float = 0.0, lineage_value: float = 0.0) -> float:
    importance = float(memory.get("importance", 0.5))
    confidence = float(memory.get("confidence", 0.8))
    recency = _recency_score(memory.get("last_accessed_at") or memory.get("created_at"))
    frequency = _frequency_score(int(memory.get("access_count", 0)))

    age_days = 0.0
    if memory.get("created_at"):
        try:
            age_days = (datetime.utcnow() - memory["created_at"]).days
        except Exception:
            pass
    staleness = 0.0 if age_days < 14 else min(1.0, (age_days - 14) / 90.0)

    contradicted_by = memory.get("lineage", {}).get("contradicted_by", []) or []
    conflict = min(1.0, 0.4 * len(contradicted_by) + 0.6 * (1 - confidence))

    return (
        0.30 * recency
      + 0.25 * frequency
      + 0.20 * importance
      + 0.15 * centrality
      + 0.10 * lineage_value
      - 0.20 * staleness
      - 0.30 * conflict
    )
