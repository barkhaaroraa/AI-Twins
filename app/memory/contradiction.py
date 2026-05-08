"""Contradiction detection at write time.

After a memory has cleared dedupe, check whether it conflicts with anything the user
previously said. If yes, both memories are kept and a CONTRADICTS edge is added in Neo4j;
confidences drift on both sides.
"""
import logging
from typing import Dict, List, Optional

from qdrant_client.http import models as qm

from app.db.mongo import memory_collection
from app.graph import neo4j_client
from app.llm.ollama_client import OllamaUnavailable, generate_json
from app.memory.vector import COLLECTION_NAME, qdrant

log = logging.getLogger(__name__)

CANDIDATE_LIMIT = 3
MIN_CANDIDATE_SIM = 0.55


CONTRADICTION_PROMPT = """You detect contradictions between user memories.

Two memories CONTRADICT if they make incompatible claims about the same fact, preference,
or state. Mere refinement, elaboration, or temporal updates without conflict do NOT
contradict.

Output ONLY a JSON object with this exact schema. No commentary, no prose.
{{"contradicting_ids": ["<id>", ...]}}

Example 1:
NEW: "I prefer evening workouts now"
OLDER:
- id=A: "I prefer working out in the mornings"
- id=B: "I had pasta for lunch yesterday"
Output: {{"contradicting_ids": ["A"]}}

Example 2:
NEW: "Migrated the API to FastAPI v2"
OLDER:
- id=C: "Started building the API in FastAPI"
Output: {{"contradicting_ids": []}}

Example 3:
NEW: "I'm vegetarian"
OLDER:
- id=D: "I had chicken for dinner"
- id=E: "I love steak"
Output: {{"contradicting_ids": ["D", "E"]}}

NEW memory:
"{new_summary}"

OLDER memories:
{older_block}

Output:"""


CONTRADICTS_CYPHER = """
MATCH (a:Memory {id: $a}), (b:Memory {id: $b})
MERGE (a)-[r1:CONTRADICTS]->(b)
MERGE (b)-[r2:CONTRADICTS]->(a)
SET r1.weight = -0.7, r2.weight = -0.7
"""

# Inline SUPERSEDES: a is the new memory, b is the older one being contradicted.
# By construction (a was written now, b was already in Qdrant) a.created_at > b.created_at,
# so we promote the supersedes immediately — without waiting for the light consolidator.
INLINE_SUPERSEDES_CYPHER = """
MATCH (a:Memory {id: $new_id}), (b:Memory {id: $old_id})
MERGE (a)-[:SUPERSEDES]->(b)
"""


def _candidate_pool(user_id: str, agent_owner: str, embedding: List[float], exclude_id: Optional[str] = None) -> List[Dict]:
    """Top-K visible to the writer agent (own private + shared + public), excluding self."""
    flt = qm.Filter(
        must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))],
        should=[
            qm.FieldCondition(key="visibility", match=qm.MatchValue(value="public")),
            qm.Filter(
                must=[
                    qm.FieldCondition(key="visibility", match=qm.MatchValue(value="shared")),
                    qm.FieldCondition(key="shared_with", match=qm.MatchValue(value=agent_owner)),
                ]
            ),
            qm.Filter(
                must=[
                    qm.FieldCondition(key="visibility", match=qm.MatchValue(value="private")),
                    qm.FieldCondition(key="agent_owner", match=qm.MatchValue(value=agent_owner)),
                ]
            ),
        ],
    )
    try:
        # +1 in case the new memory itself is the top hit; we'll filter it out.
        res = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=flt,
            limit=CANDIDATE_LIMIT + 1,
        )
    except Exception:
        log.exception("contradiction candidate query failed")
        return []
    raw = []
    for h in res.points:
        if not h.payload:
            continue
        if exclude_id and h.id == exclude_id:
            continue
        if h.score < MIN_CANDIDATE_SIM:
            continue
        raw.append({"id": h.id, "summary": h.payload.get("text", "")})

    # Defense in depth: drop candidates that have no live Mongo doc. A dangling Qdrant
    # point (e.g. forgotten-but-not-yet-purged, or test wipe that didn't touch Qdrant)
    # would otherwise feed the LLM a ghost ID that record_contradiction silently no-ops on.
    if not raw:
        return []
    live_ids = {
        d["_id"]
        for d in memory_collection.find(
            {"_id": {"$in": [r["id"] for r in raw]}, "tier": {"$ne": "forgotten"}},
            {"_id": 1},
        )
    }
    out = [r for r in raw if r["id"] in live_ids][:CANDIDATE_LIMIT]
    return out


def detect_contradictions(
    user_id: str, agent_owner: str, new_memory: Dict, embedding: List[float]
) -> List[str]:
    """Returns list of memory_ids that the new memory contradicts (possibly empty)."""
    candidates = _candidate_pool(user_id, agent_owner, embedding, exclude_id=new_memory.get("_id"))
    if not candidates:
        return []
    older_block = "\n".join(f"- id={c['id']}: \"{c['summary']}\"" for c in candidates)
    try:
        result = generate_json(
            CONTRADICTION_PROMPT.format(
                new_summary=new_memory.get("summary", ""),
                older_block=older_block,
            ),
            timeout=60,
        )
    except OllamaUnavailable:
        return []
    except Exception:
        log.exception("contradiction LLM call failed")
        return []
    raw = result.get("contradicting_ids") or []
    valid_ids = {c["id"] for c in candidates}
    return [x for x in raw if x in valid_ids]


def record_contradiction(new_id: str, old_id: str):
    """Record contradiction + supersession.

    new_id is the just-written memory; old_id is a previously-stored memory the LLM has
    flagged as opposing. We:
      1. Add bidirectional CONTRADICTS edges (suppression signal for PPR if reused later).
      2. Add a directed SUPERSEDES edge new -> old (newest-statement-wins semantic).
      3. Mirror the SUPERSEDES into Mongo lineage so the cascade Stage 5 filter drops the
         older memory from retrieval starting on the very next request.
      4. Drift confidence so prompt ranking favours the new memory in the brief window
         before retrieval has run again.
    """
    try:
        neo4j_client.run_write(CONTRADICTS_CYPHER, a=new_id, b=old_id)
    except Exception:
        log.exception("CONTRADICTS edge write failed")

    try:
        neo4j_client.run_write(INLINE_SUPERSEDES_CYPHER, new_id=new_id, old_id=old_id)
    except Exception:
        log.exception("inline SUPERSEDES edge write failed")

    memory_collection.update_one(
        {"_id": new_id},
        {
            "$addToSet": {
                "lineage.contradicted_by": old_id,
                "lineage.supersedes": old_id,
            },
            "$mul": {"confidence": 1.05},
        },
    )
    memory_collection.update_one(
        {"_id": old_id},
        {
            "$addToSet": {
                "lineage.contradicted_by": new_id,
                "lineage.supersedes_by": new_id,
            },
            "$mul": {"confidence": 0.7},
        },
    )
    for mid in (new_id, old_id):
        doc = memory_collection.find_one({"_id": mid}, {"confidence": 1})
        if doc:
            c = max(0.0, min(1.0, float(doc.get("confidence", 0.8))))
            memory_collection.update_one({"_id": mid}, {"$set": {"confidence": c}})
