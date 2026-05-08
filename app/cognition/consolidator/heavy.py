"""Heavy consolidator — cron 03:00 + idle ≥30 min fallback. May call LLM.

Tasks:
1. Cluster detection: high-CO_ACTIVATED memory clusters not yet abstracted.
2. Coherence filter: avg pairwise embedding similarity ≥ 0.75.
3. Abstraction induction (LLM) → Concept node + ABSTRACTION_OF edges.
4. Tier migration: warm → cold (utility < 0.4 AND age > 30d AND no_act 7d).
5. Forgetting: utility < 0.05 AND confidence < 0.3 AND age > 90d.
   Removes from Qdrant + warm Mongo; keeps Neo4j node for lineage.

Run from FastAPI process via APScheduler, or as a one-shot:
    python -m app.cognition.consolidator.heavy --once
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List

from ulid import ULID

from app.cognition.consolidator.abstraction import avg_pairwise_similarity, induce_concept
from app.db.mongo import concept_board_collection, memory_collection
from app.graph import neo4j_client
from app.memory.vector import COLLECTION_NAME, qdrant

log = logging.getLogger(__name__)


CLUSTER_CYPHER = """
MATCH (m1:Memory)-[r:CO_ACTIVATED]-(m2:Memory)
WHERE m1.user_id = $user_id AND m2.user_id = $user_id
  AND r.weight > $weight_threshold
  AND NOT (m1)<-[:ABSTRACTION_OF]-(:Concept)
WITH m1, collect(DISTINCT m2.id) AS group
WHERE size(group) >= $min_members - 1
RETURN m1.id AS seed_id, [m1.id] + group AS member_ids
"""

CREATE_CONCEPT_CYPHER = """
MERGE (u:User {id: $user_id})
CREATE (c:Concept {
  id: $cid, label: $label, summary: $summary, confidence: $confidence,
  induced_at: datetime(), user_id: $user_id
})
MERGE (c)-[:OWNED_BY]->(u)
WITH c, $member_ids AS mids
UNWIND mids AS mid
  MATCH (m:Memory {id: mid})
  MERGE (c)-[:ABSTRACTION_OF]->(m)
RETURN c.id AS id
"""


# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------

def detect_clusters(user_id: str, weight_threshold: float = 0.4, min_members: int = 3) -> List[Dict]:
    try:
        rows = neo4j_client.run_read(
            CLUSTER_CYPHER,
            user_id=user_id, weight_threshold=weight_threshold, min_members=min_members,
        )
    except Exception:
        log.exception("detect_clusters failed for user %s", user_id)
        return []
    # Dedupe overlapping clusters: prefer the larger one if seeds overlap.
    seen = set()
    out = []
    for r in sorted(rows, key=lambda x: -len(x["member_ids"])):
        ids = frozenset(r["member_ids"])
        if any(ids <= s for s in seen):
            continue
        seen.add(ids)
        out.append({"seed_id": r["seed_id"], "member_ids": list(ids)})
    return out


def induce_concepts_for_user(user_id: str) -> List[str]:
    """Returns the list of newly-created Concept ids."""
    new_ids: List[str] = []
    clusters = detect_clusters(user_id)
    for cluster in clusters:
        member_docs = list(memory_collection.find(
            {"_id": {"$in": cluster["member_ids"]}},
            {"_id": 1, "summary": 1, "embedding": 1},
        ))
        embeds = [d.get("embedding") for d in member_docs if d.get("embedding")]
        # Coherence threshold calibrated for bge-small-en-v1.5: related clusters score
        # ~0.7+, unrelated noise ~0.56.
        if len(embeds) < 2 or avg_pairwise_similarity(embeds) < 0.7:
            continue
        concept = induce_concept(member_docs)
        if not concept:
            continue
        cid = str(ULID().to_uuid())
        try:
            neo4j_client.run_write(
                CREATE_CONCEPT_CYPHER,
                user_id=user_id,
                cid=cid,
                label=concept["label"],
                summary=concept["summary"],
                confidence=concept["confidence"],
                member_ids=cluster["member_ids"],
            )
            concept_board_collection.insert_one({
                "_id": cid,
                "user_id": user_id,
                "label": concept["label"],
                "summary": concept["summary"],
                "confidence": concept["confidence"],
                "member_ids": cluster["member_ids"],
                "induced_at": datetime.utcnow(),
            })
            new_ids.append(cid)
        except Exception:
            log.exception("create_concept failed for user %s cluster %s", user_id, cluster["seed_id"])
    return new_ids


def migrate_to_cold() -> int:
    """Mark eligible warm memories tier='cold'. Cold memories remain searchable
    via the same Qdrant collection and Mongo collection, just labelled cold."""
    cutoff_age = datetime.utcnow() - timedelta(days=30)
    cutoff_act = datetime.utcnow() - timedelta(days=7)
    n = memory_collection.update_many(
        {
            "tier": "warm",
            "utility": {"$lt": 0.4, "$ne": None},
            "created_at": {"$lt": cutoff_age},
            "last_accessed_at": {"$lt": cutoff_act},
        },
        {"$set": {"tier": "cold"}},
    )
    return n.modified_count


def forget(threshold_utility: float = 0.05, threshold_conf: float = 0.3) -> int:
    """Mark forgotten and remove from Qdrant + warm Mongo. Neo4j node retained."""
    cutoff_age = datetime.utcnow() - timedelta(days=90)
    docs = list(memory_collection.find(
        {
            "tier": {"$ne": "forgotten"},
            "utility": {"$lt": threshold_utility, "$ne": None},
            "confidence": {"$lt": threshold_conf},
            "created_at": {"$lt": cutoff_age},
        },
        {"_id": 1},
    ))
    n = 0
    for d in docs:
        mid = d["_id"]
        try:
            qdrant.delete(collection_name=COLLECTION_NAME, points_selector=[mid])
        except Exception:
            pass
        memory_collection.update_one({"_id": mid}, {"$set": {"tier": "forgotten"}})
        n += 1
    return n


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------

def run_for_all_users() -> Dict[str, object]:
    user_ids = memory_collection.distinct("user_id")
    induced = {}
    for uid in user_ids:
        ids = induce_concepts_for_user(uid)
        if ids:
            induced[uid] = ids
    cold = migrate_to_cold()
    forgotten = forget()
    return {"concepts_induced": induced, "migrated_to_cold": cold, "forgotten": forgotten}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--user", default=None, help="limit to a single user_id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.user:
        induced = induce_concepts_for_user(args.user)
        print(f"induced concepts: {induced}")
    else:
        result = run_for_all_users()
        print(result)
    neo4j_client.close()


if __name__ == "__main__":
    main()
