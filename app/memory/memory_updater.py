import logging
import threading
import weakref
from datetime import datetime
from typing import Dict, List, Optional

from qdrant_client.http import models as qm
from ulid import ULID

from app.agents import AGENTS
from app.agents.sharing_policy import auto_visibility
from app.db.mongo import memory_collection
from app.graph.projections import project_memory
from app.memory.contradiction import detect_contradictions, record_contradiction
from app.memory.vector import COLLECTION_NAME, embed_text, qdrant, store_memory

log = logging.getLogger(__name__)

DEDUPE_THRESHOLD = 0.92

class _MemoryLock:
    """Thin wrapper so locks can live in a WeakValueDictionary (raw _thread.lock isn't weakrefable)."""
    __slots__ = ("_lock", "__weakref__")

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()
        return False


# Per-memory lock cache. WeakValueDictionary so locks are GC'd as soon as no caller holds
# a strong reference — prevents unbounded growth across the process's lifetime.
_lock_cache: "weakref.WeakValueDictionary[str, _MemoryLock]" = weakref.WeakValueDictionary()
_cache_guard = threading.Lock()


def memory_lock(memory_id: str) -> _MemoryLock:
    with _cache_guard:
        lock = _lock_cache.get(memory_id)
        if lock is None:
            lock = _MemoryLock()
            _lock_cache[memory_id] = lock
        return lock


def _build_lineage(source_event_id: Optional[str]) -> Dict:
    return {
        "source_event_id": source_event_id,
        "evidence": [source_event_id] if source_event_id else [],
        "derived_from": [],
        "supersedes": [],
        "abstraction_of": [],
        "contradicted_by": [],
        "version": 1,
    }


def _find_duplicate(
    user_id: str,
    agent_owner: str,
    summary: str,
    embedding: List[float],
    threshold: float = DEDUPE_THRESHOLD,
) -> Optional[Dict]:
    """Same user, same agent_owner, top-1 ≥ threshold => duplicate."""
    flt = qm.Filter(must=[
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(key="agent_owner", match=qm.MatchValue(value=agent_owner)),
    ])
    try:
        res = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=flt,
            limit=1,
        )
    except Exception:
        log.exception("dedupe query failed")
        return None
    if not res.points:
        return None
    top = res.points[0]
    if top.score < threshold:
        return None
    existing = memory_collection.find_one({"_id": top.id})
    return existing


def _reinforce(existing: Dict, source_event_id: Optional[str]) -> Dict:
    """Bump confidence asymptotically, append evidence, increment access_count."""
    mid = existing["_id"]
    with memory_lock(mid):
        new_conf = min(1.0, existing.get("confidence", 0.8) + 0.05 * (1.0 - existing.get("confidence", 0.8)))
        update = {
            "$set": {
                "confidence": new_conf,
                "updated_at": datetime.utcnow(),
            },
            "$inc": {"access_count": 1},
        }
        if source_event_id:
            update["$addToSet"] = {"lineage.evidence": source_event_id}
        memory_collection.update_one({"_id": mid}, update)
        log.info("reinforced %s (confidence %.3f -> %.3f)", mid, existing.get("confidence", 0.8), new_conf)
    return memory_collection.find_one({"_id": mid})


def update_memory(
    user_id: str,
    summarized_memory: Optional[Dict],
    source_agent: Optional[str] = None,
    source_event_id: Optional[str] = None,
    visibility: Optional[str] = None,
    shared_with: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Triple-write: Mongo memories_warm + Qdrant mem_warm + Neo4j projection.

    Dedupe step first: if a near-duplicate exists for the same (user_id, agent_owner),
    reinforce it instead of creating a new memory.
    """
    if summarized_memory is None:
        return None

    summary = summarized_memory["summary"]
    embedding = embed_text(summary)

    agent_owner = source_agent or "unknown"

    # Step 1: dedupe
    duplicate = _find_duplicate(user_id, agent_owner, summary, embedding)
    if duplicate is not None:
        return _reinforce(duplicate, source_event_id)

    # Step 2: build doc
    memory_id = str(ULID().to_uuid())  # sortable, Qdrant-compatible
    memory_type = summarized_memory.get("memory_type", "Semantic")
    agent_default = "private"
    if source_agent and source_agent in AGENTS:
        agent_default = AGENTS[source_agent].default_visibility
    final_visibility = visibility or auto_visibility(memory_type, agent_default)

    now = datetime.utcnow()
    memory_doc = {
        "_id": memory_id,
        "user_id": user_id,
        "version": 1,
        "tier": "warm",
        "type": summarized_memory.get("type", "fact"),
        "intent": summarized_memory.get("intent", summarized_memory.get("type", "fact")),
        "memory_type": memory_type,
        "summary": summary,
        "raw_text_event_id": source_event_id,
        "entities": summarized_memory.get("entities", []),
        "relationships": summarized_memory.get("relationships", []),
        "confidence": float(summarized_memory.get("confidence", 0.8)),
        "importance": float(summarized_memory.get("importance", 0.5)),
        "utility": None,
        "last_accessed_at": now,
        "access_count": 0,
        "embedding": embedding,
        "source_agent": source_agent,
        "agent_owner": agent_owner,
        "visibility": final_visibility,
        "shared_with": shared_with or [],
        "lineage": _build_lineage(source_event_id),
        "created_at": now,
        "updated_at": now,
    }

    # Step 3: triple-write under per-memory lock
    with memory_lock(memory_id):
        memory_collection.insert_one(memory_doc)
        store_memory(
            user_id=user_id,
            text=summary,
            memory_id=memory_id,
            memory_type=memory_type,
            entities=memory_doc["entities"],
            confidence=memory_doc["confidence"],
            agent_owner=agent_owner,
            visibility=final_visibility,
            shared_with=memory_doc["shared_with"],
            intent=memory_doc["intent"],
            created_at_epoch_ms=int(now.timestamp() * 1000),
            embedding=embedding,
        )
        project_memory(memory_doc)

    # Step 4: contradiction detection (after the new memory is visible to itself).
    # Best-effort — silently skipped if the LLM is unavailable.
    try:
        contradicts = detect_contradictions(user_id, agent_owner, memory_doc, embedding)
        for old_id in contradicts:
            if old_id == memory_id:
                continue
            record_contradiction(memory_id, old_id)
    except Exception:
        log.exception("contradiction detection failed for %s", memory_id)

    return memory_doc
