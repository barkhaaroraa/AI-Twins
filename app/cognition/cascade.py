"""6-stage retrieval cascade.

Stage 1 — Seed (vector + entity + working-set).
Stage 2 — Working-set expansion.
Stage 3 — Spreading activation (PPR). Stub here, real impl in Phase 4.
Stage 4 — Causal/temporal traversal. Stub here, real impl in Phase 4.
Stage 5 — Filter + synthesis (ACL, dedupe, rank).
Stage 6 — Reflective update (post-response).
"""
import logging
import math
import time
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from app.agents import AGENTS, AgentSpec
from app.agents.sharing_policy import acl_check
from app.cognition import activation as activation_kernel
from app.cognition.blackboard import RetrievalTrace
from app.cognition.blackboard_registry import get_blackboard
from app.cognition.ner import entity_names
from app.cognition.ppr import power_method_ppr
from app.db.mongo import co_activation_log_collection, memory_collection
from app.graph import neo4j_client
from app.graph.queries import causal_walk, fetch_subgraph
from app.memory.vector import search_memory

log = logging.getLogger(__name__)

DEFAULT_TOP_N = 4
ENTITY_SEED_LIMIT = 8


# ---------------------------------------------------------------------------
# Stage 1 — Seed
# ---------------------------------------------------------------------------

def _vector_seed(query: str, user_id: str, agent: AgentSpec, k: int = 5) -> Dict[str, float]:
    hits = search_memory(user_id, query, limit=k, agent_name=agent.name)
    return {h["memory_id"]: float(h.get("similarity_score", 0.0)) for h in hits}


ENTITY_SEED_CYPHER = """
MATCH (m:Memory {user_id: $user_id})-[:MENTIONS]->(e:Entity {user_id: $user_id})
WHERE e.name IN $names
  AND (
    m.visibility = 'public'
    OR (m.visibility = 'shared' AND $agent_name IN m.shared_with)
    OR (m.visibility = 'private' AND m.agent_owner = $agent_name)
  )
RETURN m.id AS id, count(DISTINCT e) AS hits
ORDER BY hits DESC
LIMIT $limit
"""


def _entity_seed(query: str, user_id: str, agent: AgentSpec) -> Dict[str, float]:
    names = entity_names(query)
    if not names:
        return {}
    try:
        rows = neo4j_client.run_read(
            ENTITY_SEED_CYPHER,
            user_id=user_id, names=names, agent_name=agent.name, limit=ENTITY_SEED_LIMIT,
        )
    except Exception:
        log.exception("entity seed failed")
        return {}
    out = {}
    max_hits = max((r["hits"] for r in rows), default=1)
    for r in rows:
        # Normalise to [0, 1] so it composes with the vector seed.
        out[r["id"]] = float(r["hits"]) / float(max_hits)
    return out


def _working_set_seed(blackboard) -> Dict[str, float]:
    return blackboard.working_set_seed()


def stage1_seed(query: str, user_id: str, agent: AgentSpec, blackboard) -> Dict[str, float]:
    v = _vector_seed(query, user_id, agent)
    e = _entity_seed(query, user_id, agent)
    w = _working_set_seed(blackboard)
    return activation_kernel.merge_seeds(v, e, w)


# ---------------------------------------------------------------------------
# Stage 3 — Spreading activation (PPR over Neo4j subgraph)
# ---------------------------------------------------------------------------

def stage3_spread(seeds: Dict[str, float], user_id: str, agent: AgentSpec) -> Dict[str, float]:
    if not seeds:
        return seeds
    sub = fetch_subgraph(list(seeds.keys()), user_id, agent.name)
    nodes = sub.get("node_ids") or list(seeds.keys())
    edges = sub.get("edges") or []
    if not edges:
        return seeds
    return power_method_ppr(nodes=nodes, edges=edges, seeds=seeds)


# ---------------------------------------------------------------------------
# Stage 4 — Causal/temporal walks
# ---------------------------------------------------------------------------

CAUSAL_BOOST = 0.4
CAUSAL_HOP_DECAY = 0.6  # 1-hop = 1.0, 2-hop = 0.6, 3-hop = 0.36


def stage4_causal(seeds: Dict[str, float], user_id: str, agent: AgentSpec) -> Dict[str, float]:
    if not seeds:
        return seeds
    # Pick top-N seeds for causal expansion (avoid expanding from low-activation noise).
    top_ids = [
        mid for mid, _ in sorted(seeds.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    walks = causal_walk(top_ids, user_id, agent.name)
    out = dict(seeds)
    for bucket in (walks.get("causes", []), walks.get("laters", []), walks.get("latests", [])):
        for entry in bucket:
            mid = entry.get("id") if isinstance(entry, dict) else entry
            if not mid:
                continue
            hops = max(1, int(entry.get("hops", 1)) if isinstance(entry, dict) else 1)
            boost = CAUSAL_BOOST * (CAUSAL_HOP_DECAY ** (hops - 1))
            out[mid] = max(out.get(mid, 0.0), boost)
    return out


# ---------------------------------------------------------------------------
# Stage 5 — Filter + synthesis
# ---------------------------------------------------------------------------

def _load_memories(memory_ids: List[str]) -> Dict[str, dict]:
    if not memory_ids:
        return {}
    docs = list(memory_collection.find({"_id": {"$in": memory_ids}}))
    return {d["_id"]: d for d in docs}


def _recency_decay(last_accessed_at) -> float:
    """Smooth recency factor in [0, 1]. Newer = closer to 1."""
    if not last_accessed_at:
        return 0.5
    try:
        delta = (datetime.utcnow() - last_accessed_at).total_seconds()
        return float(math.exp(-delta / (7.0 * 24.0 * 3600.0)))  # 1-week half-life
    except Exception:
        return 0.5


def stage5_synthesis(
    activated: Dict[str, float], agent: AgentSpec, top_n: int
) -> List[dict]:
    docs = _load_memories(list(activated.keys()))
    visible: List[Tuple[float, dict]] = []
    for mid, act in activated.items():
        d = docs.get(mid)
        if not d:
            continue
        if not acl_check(d, agent):
            continue
        # Drop superseded (Phase 7 will populate lineage.supersedes)
        if d.get("lineage", {}).get("supersedes_by"):
            continue
        score = act * float(d.get("confidence", 0.8)) * _recency_decay(d.get("last_accessed_at"))
        visible.append((score, d))
    visible.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, d in visible[:top_n]:
        out.append({
            "memory_id": d["_id"],
            "text": d.get("summary", ""),
            "summary": d.get("summary", ""),
            "similarity_score": round(score, 3),
            "memory_type": d.get("memory_type", "Semantic"),
            "intent": d.get("intent", "fact"),
            "entities": d.get("entities", []),
            "agent_owner": d.get("agent_owner"),
            "visibility": d.get("visibility", "private"),
            "tier": d.get("tier", "warm"),
            "confidence": d.get("confidence", 0.8),
            "lineage": d.get("lineage", {}),
            "created_at": (
                d["created_at"].isoformat()
                if hasattr(d.get("created_at"), "isoformat")
                else d.get("created_at")
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Stage 6 — Reflect
# ---------------------------------------------------------------------------

REFLECT_BUMP = 0.3


def stage6_reflect(used_memory_ids: List[str], user_id: str, agent: AgentSpec, blackboard, query: str):
    if not used_memory_ids:
        return
    for mid in used_memory_ids:
        blackboard.activate(mid, REFLECT_BUMP)
    docs = _load_memories(used_memory_ids)
    for mid in used_memory_ids:
        if mid in docs:
            blackboard.admit(docs[mid], strength=blackboard.get_activation(mid))

    # Bump retrieval freshness so recency_decay and utility see actual usage.
    try:
        memory_collection.update_many(
            {"_id": {"$in": list(used_memory_ids)}},
            {"$set": {"last_accessed_at": datetime.utcnow()}, "$inc": {"access_count": 1}},
        )
    except Exception:
        log.exception("last_accessed_at bump failed")

    # Co-activation log for the light consolidator to reinforce CO_ACTIVATED edges.
    pairs = list(combinations(sorted(used_memory_ids), 2))
    if pairs:
        try:
            co_activation_log_collection.insert_one({
                "user_id": user_id,
                "agent": agent.name,
                "ts": time.time(),
                "pairs": pairs,
            })
        except Exception:
            log.exception("co_activation_log insert failed")

    blackboard.last_cascade = RetrievalTrace(
        query=query,
        used_memory_ids=list(used_memory_ids),
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def cascade(query: str, user_id: str, agent_name: str, top_n: int = DEFAULT_TOP_N) -> List[dict]:
    agent = AGENTS.get(agent_name)
    if agent is None:
        raise ValueError(f"unknown agent {agent_name}")
    bb = get_blackboard(user_id, agent_name)

    seeds = stage1_seed(query, user_id, agent, bb)
    expanded = activation_kernel.expand_working_set(seeds, bb)
    spread = stage3_spread(expanded, user_id, agent)
    causal = stage4_causal(spread, user_id, agent)
    out = stage5_synthesis(causal, agent, top_n)
    return out


def reflect(used_memory_ids: List[str], user_id: str, agent_name: str, query: str = ""):
    agent = AGENTS.get(agent_name)
    if agent is None:
        return
    bb = get_blackboard(user_id, agent_name)
    stage6_reflect(used_memory_ids, user_id, agent, bb, query)
