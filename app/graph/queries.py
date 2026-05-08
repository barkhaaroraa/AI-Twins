"""Parametrized read queries against the Neo4j cortex.

`fetch_subgraph` and `causal_walk` are used by the cascade in Phase 4 (PPR + Stage 4).
ACL is enforced inside the MATCH so non-visible nodes are never traversed.
"""
import logging
from typing import Dict, List

from app.graph import neo4j_client

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------

LINEAGE_NEIGHBOURS_CYPHER = """
MATCH (m:Memory {id: $memory_id})
OPTIONAL MATCH (m)-[r:DERIVED_FROM|REFINES|SUPERSEDES|CONTRADICTS|ABSTRACTION_OF|CAUSES|PRECEDES]->(n)
RETURN type(r) AS rel, n.id AS neighbour_id, n.summary AS summary
"""


def lineage_neighbours(memory_id: str) -> List[Dict]:
    try:
        return neo4j_client.run_read(LINEAGE_NEIGHBOURS_CYPHER, memory_id=memory_id)
    except Exception:
        return []


COUNT_MEMORIES_FOR_USER_CYPHER = """
MATCH (m:Memory {user_id: $user_id}) RETURN count(m) AS n
"""


def count_memories_for_user(user_id: str) -> int:
    try:
        rows = neo4j_client.run_read(COUNT_MEMORIES_FOR_USER_CYPHER, user_id=user_id)
        return int(rows[0]["n"]) if rows else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# PPR subgraph fetch (Phase 4)
# ---------------------------------------------------------------------------

# Two-hop expansion across PPR-relevant edges, with ACL enforced inside the MATCH.
# We project relationships into a (src, dst, type, weight) tuple so the Python
# kernel can normalise per-source out-edges.
SUBGRAPH_CYPHER = """
WITH $seed_ids AS seeds, $user_id AS uid, $agent_name AS aname
MATCH (start:Memory {user_id: uid}) WHERE start.id IN seeds
  AND (
    start.visibility = 'public'
    OR (start.visibility = 'shared' AND aname IN start.shared_with)
    OR (start.visibility = 'private' AND start.agent_owner = aname)
  )
CALL {
  WITH start, aname
  MATCH (start)-[r1:CO_ACTIVATED|MENTIONS|DEPENDS_ON|DERIVED_FROM|REFINES|CAUSES|PRECEDES|ABSTRACTION_OF|SUPERSEDES|CONTRADICTS]-(n1:Memory)
  WHERE (
    n1.visibility = 'public'
    OR (n1.visibility = 'shared' AND aname IN n1.shared_with)
    OR (n1.visibility = 'private' AND n1.agent_owner = aname)
  )
  RETURN n1, r1
  UNION
  WITH start, aname
  MATCH (start)-[r1:MENTIONS]-(:Entity)-[r2:MENTIONS]-(n1:Memory)
  WHERE (
    n1.visibility = 'public'
    OR (n1.visibility = 'shared' AND aname IN n1.shared_with)
    OR (n1.visibility = 'private' AND n1.agent_owner = aname)
  )
  RETURN n1, r1
}
WITH collect(DISTINCT start) + collect(DISTINCT n1) AS nodes
UNWIND nodes AS m
WITH DISTINCT m WHERE m IS NOT NULL
WITH collect(m) AS mems
UNWIND mems AS src
OPTIONAL MATCH (src)-[r:CO_ACTIVATED|MENTIONS|DEPENDS_ON|DERIVED_FROM|REFINES|CAUSES|PRECEDES|ABSTRACTION_OF|SUPERSEDES|CONTRADICTS]-(dst:Memory)
WHERE dst IN mems AND src.id < dst.id          // dedupe undirected pairs
RETURN
  [m IN mems | m.id] AS node_ids,
  collect({src: src.id, dst: dst.id, type: type(r), weight: coalesce(r.weight, 0.5)}) AS edges
"""


CONCEPT_BRIDGE_CYPHER = """
WITH $seed_ids AS seeds, $user_id AS uid, $agent_name AS aname
MATCH (s:Memory {user_id: uid})<-[:ABSTRACTION_OF]-(c:Concept)-[:ABSTRACTION_OF]->(n:Memory {user_id: uid})
WHERE s.id IN seeds AND s.id <> n.id
  AND (
    n.visibility = 'public'
    OR (n.visibility = 'shared' AND aname IN n.shared_with)
    OR (n.visibility = 'private' AND n.agent_owner = aname)
  )
RETURN s.id AS src, n.id AS dst
"""

# Two memories sharing an Entity get a synthetic MENTIONS-bridge edge so PPR can
# spread between them. The Cypher subgraph fetch pulls them into the node set but
# the schema has no direct Memory↔Memory MENTIONS edge — without this, PPR would
# see them as disconnected components.
ENTITY_BRIDGE_CYPHER = """
WITH $node_ids AS nids, $user_id AS uid, $agent_name AS aname
MATCH (s:Memory {user_id: uid})-[:MENTIONS]->(e:Entity {user_id: uid})<-[:MENTIONS]-(n:Memory {user_id: uid})
WHERE s.id IN nids AND n.id IN nids AND s.id < n.id
  AND (
    n.visibility = 'public'
    OR (n.visibility = 'shared' AND aname IN n.shared_with)
    OR (n.visibility = 'private' AND n.agent_owner = aname)
  )
RETURN s.id AS src, n.id AS dst, count(DISTINCT e) AS shared
"""


def fetch_subgraph(seed_ids: List[str], user_id: str, agent_name: str) -> Dict:
    """Returns {node_ids: [...], edges: [{src, dst, type, weight}, ...]} or empty on failure."""
    if not seed_ids:
        return {"node_ids": [], "edges": []}
    try:
        rows = neo4j_client.run_read(
            SUBGRAPH_CYPHER,
            seed_ids=list(seed_ids),
            user_id=user_id,
            agent_name=agent_name,
        )
        if not rows:
            node_ids: List[str] = list(seed_ids)
            edges: List[Dict] = []
        else:
            node_ids = rows[0].get("node_ids") or []
            edges = [e for e in (rows[0].get("edges") or []) if e and e.get("src") and e.get("dst")]
            # Always include seeds even if Cypher returned nothing.
            for s in seed_ids:
                if s not in node_ids:
                    node_ids.append(s)

        seen_pairs = {tuple(sorted([e["src"], e["dst"]])) for e in edges}

        # Concept bridge: memories sharing an abstracting Concept get a synthetic ABSTRACTION_OF
        # edge so PPR can spread activation between them. The Concept itself is not in the PPR
        # graph (it has no Memory-style ACL), but its sibling memories do enter as nodes.
        try:
            concept_rows = neo4j_client.run_read(
                CONCEPT_BRIDGE_CYPHER,
                seed_ids=list(seed_ids),
                user_id=user_id,
                agent_name=agent_name,
            )
        except Exception:
            log.exception("concept bridge query failed")
            concept_rows = []

        for row in concept_rows or []:
            src, dst = row["src"], row["dst"]
            if not src or not dst or src == dst:
                continue
            if dst not in node_ids:
                node_ids.append(dst)
            pair = tuple(sorted([src, dst]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({
                "src": src, "dst": dst,
                "type": "ABSTRACTION_OF", "weight": 0.6,
            })

        # Entity bridge: synthesize a MENTIONS-bridge edge between any two memories in
        # the current node set that share at least one Entity. Without this, the SUBGRAPH_CYPHER
        # pulls bridge-connected memories in but the OPTIONAL MATCH at the end only sees direct
        # Memory↔Memory edges (none exist for entity-bridged pairs), so PPR has no path.
        try:
            entity_rows = neo4j_client.run_read(
                ENTITY_BRIDGE_CYPHER,
                node_ids=list(node_ids),
                user_id=user_id,
                agent_name=agent_name,
            )
        except Exception:
            log.exception("entity bridge query failed")
            entity_rows = []

        for row in entity_rows or []:
            src, dst = row["src"], row["dst"]
            if not src or not dst or src == dst:
                continue
            pair = tuple(sorted([src, dst]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # Scale weight by number of shared entities, capped at MENTIONS schema weight.
            shared = int(row.get("shared", 1) or 1)
            weight = min(0.5, 0.3 + 0.1 * shared)
            edges.append({
                "src": src, "dst": dst,
                "type": "MENTIONS", "weight": weight,
            })

        return {"node_ids": node_ids, "edges": edges}
    except Exception:
        log.exception("fetch_subgraph failed")
        return {"node_ids": list(seed_ids), "edges": []}


# ---------------------------------------------------------------------------
# Causal walk (Phase 4)
# ---------------------------------------------------------------------------

CAUSAL_WALK_CYPHER = """
WITH $seed_ids AS seeds, $user_id AS uid, $agent_name AS aname
MATCH (s:Memory) WHERE s.id IN seeds
OPTIONAL MATCH p1 = (cause:Memory)-[:CAUSES*1..3]->(s)
  WHERE cause.user_id = uid AND (
    cause.visibility = 'public'
    OR (cause.visibility = 'shared' AND aname IN cause.shared_with)
    OR (cause.visibility = 'private' AND cause.agent_owner = aname)
  )
OPTIONAL MATCH p2 = (s)-[:PRECEDES*1..3]->(later:Memory)
  WHERE later.user_id = uid AND (
    later.visibility = 'public'
    OR (later.visibility = 'shared' AND aname IN later.shared_with)
    OR (later.visibility = 'private' AND later.agent_owner = aname)
  )
OPTIONAL MATCH p3 = (latest:Memory)-[:SUPERSEDES*1..]->(s)
  WHERE latest.user_id = uid AND (
    latest.visibility = 'public'
    OR (latest.visibility = 'shared' AND aname IN latest.shared_with)
    OR (latest.visibility = 'private' AND latest.agent_owner = aname)
  )
RETURN
  collect(DISTINCT {id: cause.id, hops: length(p1)}) AS causes,
  collect(DISTINCT {id: later.id, hops: length(p2)}) AS laters,
  collect(DISTINCT {id: latest.id, hops: length(p3)}) AS latests
"""


def causal_walk(seed_ids: List[str], user_id: str, agent_name: str) -> Dict[str, List[Dict]]:
    """Returns three buckets, each a list of {id, hops} dicts so the caller can apply
    hop-decayed scoring (a 1-hop predecessor should outweigh a 3-hop one)."""
    empty = {"causes": [], "laters": [], "latests": []}
    if not seed_ids:
        return empty
    try:
        rows = neo4j_client.run_read(
            CAUSAL_WALK_CYPHER,
            seed_ids=list(seed_ids), user_id=user_id, agent_name=agent_name,
        )
    except Exception:
        log.exception("causal_walk failed")
        return empty
    if not rows:
        return empty
    r = rows[0]
    def _clean(items):
        out = []
        for it in items or []:
            if not it:
                continue
            mid = it.get("id") if isinstance(it, dict) else it
            if not mid:
                continue
            hops = int(it.get("hops", 1)) if isinstance(it, dict) else 1
            out.append({"id": mid, "hops": max(1, hops)})
        return out
    return {
        "causes": _clean(r.get("causes")),
        "laters": _clean(r.get("laters")),
        "latests": _clean(r.get("latests")),
    }
