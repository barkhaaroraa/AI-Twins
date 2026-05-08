"""Light consolidator — runs on idle ≥ 60 s. No LLM calls.

Tasks:
1. Drain co_activation_log → reinforce CO_ACTIVATED edges in Neo4j.
2. Decay CO_ACTIVATED edges that haven't been reinforced in the last hour.
3. Cross-agent dedupe sweep: vector top-1 across agents in last hour → link with REFINES.
4. Utility recompute on warm memories.
"""
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional

from qdrant_client.http import models as qm

from app.cognition.utility import utility
from app.db.mongo import co_activation_log_collection, memory_collection
from app.graph import neo4j_client
from app.memory.vector import COLLECTION_NAME, qdrant

log = logging.getLogger(__name__)

IDLE_THRESHOLD_SEC = 60.0


# Last-request timestamp set by the orchestrator on every request.
_last_request_ts: float = 0.0
_last_request_lock = threading.Lock()


def mark_request():
    global _last_request_ts
    with _last_request_lock:
        _last_request_ts = time.time()


def idle_for_seconds() -> float:
    """Public accessor: how long since the last request, in seconds."""
    with _last_request_lock:
        return time.time() - _last_request_ts


def is_idle() -> bool:
    return idle_for_seconds() >= IDLE_THRESHOLD_SEC


REINFORCE_CO_ACT_CYPHER = """
UNWIND $pairs AS pair
MATCH (a:Memory {id: pair[0]}), (b:Memory {id: pair[1]})
MERGE (a)-[r:CO_ACTIVATED]-(b)
ON CREATE SET r.weight = $delta, r.last_reinforced_at = datetime()
ON MATCH  SET r.weight = CASE WHEN coalesce(r.weight, 0.0) + $delta > 1.0 THEN 1.0 ELSE coalesce(r.weight, 0.0) + $delta END,
              r.last_reinforced_at = datetime()
"""

EDGE_DECAY_CYPHER = """
MATCH ()-[r:CO_ACTIVATED]-()
WHERE r.last_reinforced_at < datetime() - duration('PT1H')
SET r.weight = r.weight * 0.95
WITH r WHERE r.weight < 0.05
DELETE r
"""

CO_ACTIVATED_DEGREE_CYPHER = """
MATCH (m:Memory {user_id: $uid})-[r:CO_ACTIVATED]-()
WITH m, count(r) AS deg
RETURN m.id AS id, deg
"""


# Pairs with CONTRADICTS edges: newer memory supersedes the older one.
# Recency-wins semantic, not confidence-wins. Reason: a long-held preference reinforced
# many times accumulates high confidence via _reinforce (×asymptote toward 1.0). When the
# user finally states the opposite, the new memory has only its initial confidence — a
# confidence-gated rule would never promote it. The contradiction LLM is already the
# gatekeeper for "is this actually opposing", so once a CONTRADICTS edge exists we trust
# created_at to pick the current statement.
SUPERSEDES_PROMOTE_CYPHER = """
MATCH (a:Memory)-[:CONTRADICTS]->(b:Memory)
WHERE NOT (a)-[:SUPERSEDES]->(b)
  AND a.created_at > b.created_at
MERGE (a)-[:SUPERSEDES]->(b)
RETURN a.id AS winner_id, b.id AS loser_id
"""


# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------

def reinforce_co_activations(delta: float = 0.1) -> int:
    """Drain co_activation_log; collapse pairs; merge into Neo4j."""
    cursor = co_activation_log_collection.find({})
    pair_counts: Dict[tuple, int] = defaultdict(int)
    ids_to_delete = []
    for entry in cursor:
        ids_to_delete.append(entry["_id"])
        for p in entry.get("pairs", []):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            a, b = sorted(p)
            pair_counts[(a, b)] += 1
    if not pair_counts:
        return 0
    # Group by hits to keep weight increment proportional.
    by_delta = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        by_delta[min(1.0, delta * cnt)].append([a, b])
    try:
        for d, pairs in by_delta.items():
            neo4j_client.run_write(REINFORCE_CO_ACT_CYPHER, pairs=pairs, delta=float(d))
        co_activation_log_collection.delete_many({"_id": {"$in": ids_to_delete}})
    except Exception:
        log.exception("reinforce_co_activations failed")
        return 0
    return len(pair_counts)


def decay_co_activated_edges() -> bool:
    try:
        neo4j_client.run_write(EDGE_DECAY_CYPHER)
        return True
    except Exception:
        log.exception("decay_co_activated_edges failed")
        return False


def promote_supersedes() -> int:
    """Walk CONTRADICTS pairs; if confidence drift has crystallised a winner, add SUPERSEDES.

    Mirrors the Neo4j edge into Mongo lineage so the cascade Stage 5 filter
    (which checks lineage.supersedes_by) can drop superseded memories.
    """
    try:
        rows = neo4j_client.run_write(SUPERSEDES_PROMOTE_CYPHER)
    except Exception:
        log.exception("promote_supersedes failed")
        return 0
    n = 0
    for row in rows or []:
        # run_write returns Records; key access works.
        winner = row["winner_id"]
        loser = row["loser_id"]
        try:
            memory_collection.update_one(
                {"_id": loser},
                {"$addToSet": {"lineage.supersedes_by": winner}},
            )
            memory_collection.update_one(
                {"_id": winner},
                {"$addToSet": {"lineage.supersedes": loser}},
            )
            n += 1
        except Exception:
            log.exception("promote_supersedes mongo mirror failed for %s -> %s", winner, loser)
    return n


REFINES_CYPHER = """
MATCH (a:Memory {id: $a}), (b:Memory {id: $b})
MERGE (a)-[:REFINES]->(b)
"""


def cross_agent_dedupe_sweep(window_minutes: int = 60, threshold: float = 0.92) -> int:
    """For memories created in the last `window_minutes`, look across agents for near-duplicates.
    Cross-agent matches get a REFINES edge in Neo4j; we never auto-merge across agents.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    recent = list(memory_collection.find(
        {"created_at": {"$gte": cutoff}, "tier": {"$ne": "forgotten"}},
        {"_id": 1, "user_id": 1, "agent_owner": 1, "embedding": 1},
    ))
    refines = 0
    for m in recent:
        emb = m.get("embedding")
        if not emb:
            continue
        flt = qm.Filter(must=[
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=m["user_id"])),
        ])
        try:
            res = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=emb,
                query_filter=flt,
                limit=3,
            )
        except Exception:
            continue
        for hit in res.points:
            if hit.id == m["_id"]:
                continue
            if hit.score < threshold:
                break
            other_owner = hit.payload.get("agent_owner") if hit.payload else None
            if other_owner == m.get("agent_owner"):
                continue  # same-agent dedupe was handled at write time
            try:
                neo4j_client.run_write(REFINES_CYPHER, a=m["_id"], b=hit.id)
                refines += 1
            except Exception:
                pass
    return refines


def recompute_utility(window_days: int = 7) -> int:
    """Recompute utility for memories with recent activity. Updates the doc in-place."""
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    recent_users = memory_collection.distinct(
        "user_id", {"updated_at": {"$gte": cutoff}, "tier": {"$ne": "forgotten"}},
    )
    n_updated = 0
    for uid in recent_users:
        # Centrality: degree in CO_ACTIVATED subgraph, normalised to [0, 1].
        try:
            rows = neo4j_client.run_read(CO_ACTIVATED_DEGREE_CYPHER, uid=uid)
        except Exception:
            rows = []
        deg_by_id = {r["id"]: int(r["deg"]) for r in rows}
        max_deg = max(deg_by_id.values(), default=0) or 1
        for m in memory_collection.find(
            {"user_id": uid, "tier": {"$ne": "forgotten"}},
            {"_id": 1, "importance": 1, "confidence": 1, "access_count": 1,
             "last_accessed_at": 1, "created_at": 1, "lineage": 1},
        ):
            cent = deg_by_id.get(m["_id"], 0) / max_deg
            lineage_val = 1.0 if (m.get("lineage", {}) or {}).get("derived_from") else 0.0
            u = utility(m, centrality=cent, lineage_value=lineage_val)
            memory_collection.update_one({"_id": m["_id"]}, {"$set": {"utility": u}})
            n_updated += 1
    return n_updated


# ------------------------------------------------------------------
# Loop
# ------------------------------------------------------------------

class LightConsolidator:
    """Background asyncio-free worker.

    Runs in a daemon thread. Wakes every ~10 s, runs the four tasks if the system
    has been idle ≥ IDLE_THRESHOLD_SEC.
    """

    def __init__(self, poll_seconds: float = 10.0):
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="light-consolidator")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                if is_idle():
                    self.run_once()
            except Exception:
                log.exception("light consolidator iteration failed")
            self._stop.wait(self.poll_seconds)

    def run_once(self):
        log.info("light consolidator: tick")
        reinforce_co_activations()
        decay_co_activated_edges()
        cross_agent_dedupe_sweep()
        recompute_utility()
        promote_supersedes()


_singleton: Optional[LightConsolidator] = None


def get_consolidator() -> LightConsolidator:
    global _singleton
    if _singleton is None:
        _singleton = LightConsolidator()
    return _singleton
