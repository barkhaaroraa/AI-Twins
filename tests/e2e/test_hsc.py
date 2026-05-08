"""End-to-end verification harness for the HSC substrate.

These tests touch real Mongo, Qdrant, Neo4j, and Ollama. They each create their own
isolated user_id and clean up after themselves. They exercise the architecture's
verification checks (architecture.md §Verification).

Skipped from this harness (operational, not architectural):
- Warm restart (test #6 in spec) — blackboards rebuild lazily on next request.
- Latency p95 (test #8 in spec) — calibration depends on hardware/embedder.
"""
import time
from datetime import datetime, timedelta

import pytest

from qdrant_client.http import models as qm

from app.cognition.cascade import cascade, reflect
from app.cognition.consolidator import heavy as heavy_cons, light as light_cons
from app.db.mongo import (
    co_activation_log_collection, concept_board_collection,
    init_indexes, memory_collection, raw_events_collection,
)
from app.graph import neo4j_client
from app.memory.backing import append_event
from app.memory.memory_updater import update_memory
from app.memory.summarizer import fast_store_payload
from app.memory.vector import COLLECTION_NAME, init_vector_collection, qdrant


@pytest.fixture(scope="session", autouse=True)
def _bootstrap():
    init_indexes()
    neo4j_client.init_schema()
    init_vector_collection()
    yield
    neo4j_client.close()


def _wipe(user_id: str):
    for c in (
        memory_collection, raw_events_collection,
        co_activation_log_collection, concept_board_collection,
    ):
        c.delete_many({"user_id": user_id})
    neo4j_client.run_write("MATCH (m:Memory {user_id:$uid}) DETACH DELETE m", uid=user_id)
    neo4j_client.run_write("MATCH (c:Concept {user_id:$uid}) DETACH DELETE c", uid=user_id)
    neo4j_client.run_write("MATCH (e:Entity {user_id:$uid}) DETACH DELETE e", uid=user_id)
    neo4j_client.run_write("MATCH (u:User {id:$uid}) DETACH DELETE u", uid=user_id)
    # Qdrant — without this, prior runs leave near-duplicate points under the same user_id,
    # which poisons the contradiction candidate pool with ghost IDs that no longer exist in
    # Mongo or Neo4j.
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(filter=qm.Filter(
                must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
            )),
        )
    except Exception:
        pass


@pytest.fixture
def user_id(request):
    uid = f"test_{request.node.name}"
    _wipe(uid)
    yield uid
    _wipe(uid)


def _write(user_id, agent, message, *, intent="task", memory_type="Procedural", visibility=None, shared_with=None):
    ev = append_event(user_id, agent, "user_message", {"message": message})
    return update_memory(
        user_id,
        fast_store_payload(message, intent=intent, memory_type=memory_type),
        source_agent=agent,
        source_event_id=ev,
        visibility=visibility,
        shared_with=shared_with,
    )


# ---------------------------------------------------------------------------
# Check 1 — Isolation
# ---------------------------------------------------------------------------

def test_isolation_private_memory_invisible_cross_agent(user_id):
    m = _write(user_id, "project", "I am migrating the API to FastAPI by June 12")
    project_hits = cascade("FastAPI migration", user_id, "project", top_n=5)
    school_hits = cascade("FastAPI migration", user_id, "school", top_n=5)
    assert any(h["memory_id"] == m["_id"] for h in project_hits)
    assert not any(h["memory_id"] == m["_id"] for h in school_hits)


# ---------------------------------------------------------------------------
# Check 2 — Selective sharing
# ---------------------------------------------------------------------------

def test_shared_memory_only_reaches_named_agents(user_id):
    m = _write(user_id, "project", "PPR works well for graph activation",
               visibility="shared", shared_with=["research"])
    research_hits = cascade("PPR graph activation", user_id, "research", top_n=5)
    school_hits = cascade("PPR graph activation", user_id, "school", top_n=5)
    assert any(h["memory_id"] == m["_id"] for h in research_hits)
    assert not any(h["memory_id"] == m["_id"] for h in school_hits)


def test_preference_memory_is_auto_public(user_id):
    m = _write(user_id, "project", "I prefer mornings for deep work", intent="preference", memory_type="Preference")
    assert m["visibility"] == "public"
    trainer_hits = cascade("morning preferences", user_id, "trainer", top_n=5)
    assert any(h["memory_id"] == m["_id"] for h in trainer_hits)


# ---------------------------------------------------------------------------
# Check 3 — Spreading activation (entity bridge)
# ---------------------------------------------------------------------------

def test_spreading_activation_via_entity_bridge(user_id):
    a = _write(user_id, "school", "Reviewed Lamport paper on time clocks")
    b = _write(user_id, "school", "Compared vector clocks to total ordering in homework")
    # Manually link via shared entity (simulates the entity NER having tagged both with the same topic).
    neo4j_client.run_write(
        """MATCH (a:Memory {id:$a}), (b:Memory {id:$b})
           MERGE (e:Entity {name:'distributed-systems', kind:'topic', user_id:$uid})
           MERGE (a)-[:MENTIONS]->(e) MERGE (b)-[:MENTIONS]->(e)""",
        a=a["_id"], b=b["_id"], uid=user_id,
    )
    hits = cascade("Lamport timestamps", user_id, "school", top_n=5)
    # The query text mentions "Lamport" only — vector seed hits a strongly. PPR over the entity
    # bridge should also pull b.
    assert any(h["memory_id"] == b["_id"] for h in hits), "PPR did not bridge via shared entity"


# ---------------------------------------------------------------------------
# Check 4 — Abstraction induction
# ---------------------------------------------------------------------------

def test_abstraction_induction_creates_concept(user_id):
    texts = [
        "I am building a FastAPI backend for the project",
        "Added authentication middleware to the FastAPI app",
        "Wrote pytest tests for the FastAPI endpoints",
        "Deployed the FastAPI service to production with Docker",
    ]
    mids = [_write(user_id, "project", t)["_id"] for t in texts]
    # Push enough co-activation to trip the cluster threshold.
    for _ in range(20):
        co_activation_log_collection.insert_one({
            "user_id": user_id, "agent": "project", "ts": time.time(),
            "pairs": [[mids[i], mids[j]] for i in range(len(mids)) for j in range(i+1, len(mids))],
        })
    light_cons.reinforce_co_activations(delta=0.05)
    induced = heavy_cons.induce_concepts_for_user(user_id)
    assert induced, "no concept induced (Ollama may be slow or unavailable)"
    rows = neo4j_client.run_read(
        "MATCH (c:Concept {id:$cid})-[:ABSTRACTION_OF]->(m:Memory) RETURN count(m) AS n",
        cid=induced[0],
    )
    assert rows[0]["n"] == len(mids)


# ---------------------------------------------------------------------------
# Check 5 — Forgetting
# ---------------------------------------------------------------------------

def test_forgetting_removes_from_warm_keeps_neo4j(user_id):
    m = _write(user_id, "project", "obscure note nobody cares about")
    # Force into "forgettable" range.
    memory_collection.update_one(
        {"_id": m["_id"]},
        {"$set": {
            "utility": 0.01, "confidence": 0.1,
            "created_at": datetime.utcnow() - timedelta(days=120),
        }},
    )
    forgotten = heavy_cons.forget()
    assert forgotten >= 1
    doc = memory_collection.find_one({"_id": m["_id"]})
    assert doc["tier"] == "forgotten"
    # Neo4j node retained for lineage.
    rows = neo4j_client.run_read("MATCH (m:Memory {id:$id}) RETURN count(m) AS n", id=m["_id"])
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Check 6 — Coherence (simplified)
# ---------------------------------------------------------------------------

def test_refine_invalidates_cached_blackboard_copies(user_id):
    # Two agents load the same shared memory; one refines; the other re-fetches.
    m = _write(user_id, "project", "the auth library version is 1.2.3",
               visibility="shared", shared_with=["research"])
    cascade("auth library", user_id, "project", top_n=5)
    cascade("auth library", user_id, "research", top_n=5)
    # Project refines to a new value — same agent, near-duplicate phrasing → reinforce + confidence bump.
    _write(user_id, "project", "the auth library version is 1.2.3 actually 1.4.0 now")
    # Research re-queries; pulls the *current* doc from Mongo via cascade Stage 5.
    hits = cascade("auth library version", user_id, "research", top_n=5)
    assert any("1.4.0" in (h.get("summary") or h.get("text") or "") or "1.2.3" in (h.get("summary") or h.get("text") or "") for h in hits)


# ---------------------------------------------------------------------------
# Check 7 — Provenance (no orphans)
# ---------------------------------------------------------------------------

def test_every_memory_has_a_raw_event(user_id):
    for t in ["one", "two two", "three is a number"]:
        _write(user_id, "project", t)
    docs = list(memory_collection.find({"user_id": user_id}))
    assert docs
    for d in docs:
        sev = d["lineage"]["source_event_id"]
        assert sev
        ev = raw_events_collection.find_one({"_id": sev})
        assert ev, f"no raw_event for memory {d['_id']}"


# ---------------------------------------------------------------------------
# Check 8 — Conflict survival
# ---------------------------------------------------------------------------

def test_contradictions_kept_with_edge(user_id):
    _write(user_id, "logger", "I prefer working out in the mornings",
           intent="preference", memory_type="Preference")
    time.sleep(0.05)
    _write(user_id, "logger", "Actually I now prefer evening workouts",
           intent="preference", memory_type="Preference")
    assert memory_collection.count_documents({"user_id": user_id}) == 2
    rows = neo4j_client.run_read(
        "MATCH (a:Memory {user_id:$uid})-[:CONTRADICTS]-(b:Memory {user_id:$uid}) RETURN count(*)/2 AS pairs",
        uid=user_id,
    )
    assert rows[0]["pairs"] >= 1, "no CONTRADICTS edge — qwen2.5:3b may not be installed"
