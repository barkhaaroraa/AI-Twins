"""Project a Memory document into the Neo4j semantic cortex.

Writes the structural part of a memory: the Memory node + its OWNED_BY/AUTHORED_BY/MENTIONS/SHARED_WITH edges
and the Entity/Agent/User nodes they connect to.

Also projects:
- PRECEDES from temporal proximity (the previous memory by the same user within 1h).
- CAUSES / DEPENDS_ON when the LLM relationships block names a target memory's entity with a
  matching predicate.

Learned edges (CO_ACTIVATED, ABSTRACTION_OF, SUPERSEDES, CONTRADICTS, REFINES) are written by the
consolidators and contradiction-detection paths, not here.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional

from app.graph import neo4j_client

log = logging.getLogger(__name__)


PROJECT_MEMORY_CYPHER = """
MERGE (u:User {id: $user_id})
MERGE (a:Agent {name: $agent_owner})
MERGE (m:Memory {id: $id})
SET m.user_id = $user_id,
    m.type = $memory_type,
    m.intent = $intent,
    m.summary = $summary,
    m.importance = $importance,
    m.confidence = $confidence,
    m.visibility = $visibility,
    m.shared_with = $shared_with,
    m.agent_owner = $agent_owner,
    m.tier = $tier,
    m.created_at = $created_at,
    m.updated_at = $updated_at
MERGE (m)-[:OWNED_BY]->(u)
MERGE (m)-[:AUTHORED_BY]->(a)
WITH m, $entities AS ents, $shared_with AS sw, $user_id AS uid
UNWIND ents AS ent
  MERGE (e:Entity {name: ent.name, kind: ent.kind, user_id: uid})
  MERGE (m)-[:MENTIONS]->(e)
WITH m, sw
UNWIND sw AS shared_agent
  MERGE (sa:Agent {name: shared_agent})
  MERGE (m)-[:SHARED_WITH]->(sa)
RETURN m.id AS id
"""

# Link a new memory to the most recent prior memory by the same user, when the gap is small.
# datetime() - duration('PT1H') is "now minus 1 hour". The match is bounded so we don't link
# memories that are temporally distant — those aren't semantically a sequence.
PROJECT_PRECEDES_CYPHER = """
MATCH (m:Memory {id: $id})
MATCH (prev:Memory {user_id: $user_id})
WHERE prev.id <> $id
  AND prev.created_at < $created_at
  AND datetime(prev.created_at) > datetime($created_at) - duration('PT1H')
WITH m, prev
ORDER BY prev.created_at DESC
LIMIT 1
MERGE (prev)-[:PRECEDES]->(m)
"""

# Project explicit causal/dependency relationships. The LLM extracts triples like
# {subject: 'API', predicate: 'depends_on', object: 'Postgres'} — when the object names an
# entity attached to another memory of the same user, link the two memories.
PROJECT_CAUSES_CYPHER = """
MATCH (m:Memory {id: $id})
MATCH (other:Memory {user_id: $user_id})-[:MENTIONS]->(e:Entity {name: $object_name, user_id: $user_id})
WHERE other.id <> $id
WITH m, other LIMIT 5
MERGE (m)-[:CAUSES]->(other)
"""

PROJECT_DEPENDS_ON_CYPHER = """
MATCH (m:Memory {id: $id})
MATCH (other:Memory {user_id: $user_id})-[:MENTIONS]->(e:Entity {name: $object_name, user_id: $user_id})
WHERE other.id <> $id
WITH m, other LIMIT 5
MERGE (m)-[:DEPENDS_ON]->(other)
"""

CAUSAL_PREDICATES = {"causes", "cause", "leads_to", "results_in"}
DEPENDS_PREDICATES = {"depends_on", "requires", "needs"}

REINFORCE_MEMORY_CYPHER = """
MATCH (m:Memory {id: $id})
SET m.confidence = $confidence,
    m.updated_at = $updated_at
WITH m, $evidence_event_ids AS evs
UNWIND evs AS ev
  MERGE (m)-[:DERIVED_FROM]->(:RawEvent {id: ev})
RETURN m.id AS id
"""


def _normalize_entities(entities: List[Any]) -> List[Dict[str, str]]:
    """Accepts list[str] or list[{name, kind}], returns list[{name, kind}]."""
    out = []
    for e in entities or []:
        if isinstance(e, dict) and "name" in e:
            out.append({"name": str(e["name"]), "kind": str(e.get("kind", "topic"))})
        else:
            name = str(e).strip()
            if name:
                out.append({"name": name, "kind": "topic"})
    return out


def project_memory(memory_doc: Dict) -> bool:
    """Idempotent projection. Returns True on success, False on Neo4j unavailability."""
    try:
        created_iso = (
            memory_doc["created_at"].isoformat()
            if hasattr(memory_doc.get("created_at"), "isoformat")
            else memory_doc.get("created_at")
        )
        params = {
            "id": memory_doc["_id"],
            "user_id": memory_doc["user_id"],
            "agent_owner": memory_doc.get("agent_owner") or memory_doc.get("source_agent") or "unknown",
            "memory_type": memory_doc.get("memory_type", "Semantic"),
            "intent": memory_doc.get("intent", "fact"),
            "summary": memory_doc.get("summary", ""),
            "importance": float(memory_doc.get("importance", 0.5)),
            "confidence": float(memory_doc.get("confidence", 0.8)),
            "visibility": memory_doc.get("visibility", "private"),
            "shared_with": list(memory_doc.get("shared_with", []) or []),
            "tier": memory_doc.get("tier", "warm"),
            "created_at": created_iso,
            "updated_at": memory_doc["updated_at"].isoformat() if hasattr(memory_doc.get("updated_at"), "isoformat") else memory_doc.get("updated_at"),
            "entities": _normalize_entities(memory_doc.get("entities", [])),
        }
        neo4j_client.run_write(PROJECT_MEMORY_CYPHER, **params)
    except Exception:
        log.exception("Neo4j projection failed for %s", memory_doc.get("_id"))
        return False

    # Best-effort augmentations. Failures here don't invalidate the base projection.
    try:
        neo4j_client.run_write(
            PROJECT_PRECEDES_CYPHER,
            id=params["id"], user_id=params["user_id"], created_at=created_iso,
        )
    except Exception:
        log.exception("PRECEDES projection failed for %s", params["id"])

    try:
        _project_relationships(params["id"], params["user_id"], memory_doc.get("relationships", []) or [])
    except Exception:
        log.exception("relationship projection failed for %s", params["id"])

    return True


def _project_relationships(memory_id: str, user_id: str, relationships: List[Any]) -> None:
    """Project LLM-extracted (subject, predicate, object) triples as Memory→Memory edges
    when the object names an entity owned by another memory of the same user."""
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        predicate = str(rel.get("predicate", "")).strip().lower().replace(" ", "_")
        object_name = str(rel.get("object", "")).strip()
        if not predicate or not object_name:
            continue
        if predicate in CAUSAL_PREDICATES:
            cypher = PROJECT_CAUSES_CYPHER
        elif predicate in DEPENDS_PREDICATES:
            cypher = PROJECT_DEPENDS_ON_CYPHER
        else:
            continue
        try:
            neo4j_client.run_write(
                cypher, id=memory_id, user_id=user_id, object_name=object_name,
            )
        except Exception:
            log.exception("relationship edge write failed for %s -> %s", memory_id, object_name)
