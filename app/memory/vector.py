import logging
import os
import uuid

# Prevent transformers from importing TensorFlow (which can break on some envs)
os.environ["TRANSFORMERS_NO_TF"] = "1"

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from app.config import QDRANT_HOST, QDRANT_PORT

log = logging.getLogger(__name__)

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "mem_warm"
VECTOR_SIZE = 384


PAYLOAD_INDEXES = [
    ("user_id",      qm.PayloadSchemaType.KEYWORD),
    ("agent_owner",  qm.PayloadSchemaType.KEYWORD),
    ("visibility",   qm.PayloadSchemaType.KEYWORD),
    ("shared_with",  qm.PayloadSchemaType.KEYWORD),
    ("memory_type",  qm.PayloadSchemaType.KEYWORD),
    ("intent",       qm.PayloadSchemaType.KEYWORD),
    ("created_at",   qm.PayloadSchemaType.INTEGER),
]


def init_vector_collection():
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
        )
    for field_name, schema in PAYLOAD_INDEXES:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as e:
            # Index may already exist; not fatal.
            log.debug("payload index %s already present or failed: %s", field_name, e)


def embed_text(text: str):
    return embedder.encode(text).tolist()


def store_memory(
    user_id: str,
    text: str,
    memory_id: str = None,
    memory_type: str = None,
    entities=None,
    confidence: float = None,
    agent_owner: str = None,
    visibility: str = "private",
    shared_with=None,
    intent: str = None,
    created_at_epoch_ms: int = None,
    embedding=None,
):
    if not text or not isinstance(text, str):
        return None

    # Reuse the caller's embedding if it already computed one (memory_updater does this for
    # dedupe). Falls back to embedding here if not provided.
    vector = embedding if embedding is not None else embed_text(text)
    point_id = memory_id or str(uuid.uuid4())

    payload = {
        "user_id": user_id,
        "text": text,
        "visibility": visibility or "private",
        "shared_with": shared_with or [],
    }
    if memory_type:
        payload["memory_type"] = memory_type
    if entities:
        # Qdrant payload supports nested objects, but for indexability we keep names too.
        payload["entities"] = entities
        payload["entity_names"] = [
            (e.get("name") if isinstance(e, dict) else str(e)) for e in entities
        ]
    if confidence is not None:
        payload["confidence"] = confidence
    if agent_owner:
        payload["agent_owner"] = agent_owner
    if intent:
        payload["intent"] = intent
    if created_at_epoch_ms is not None:
        payload["created_at"] = int(created_at_epoch_ms)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{"id": point_id, "vector": vector, "payload": payload}],
    )
    return vector


def _acl_filter(user_id: str, agent_name: str):
    """Native Qdrant filter implementing the same logic as acl_check.

    visibility==public OR
    (visibility==shared AND shared_with contains agent) OR
    (visibility==private AND agent_owner==agent)
    """
    return qm.Filter(
        must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))],
        should=[
            qm.FieldCondition(key="visibility", match=qm.MatchValue(value="public")),
            qm.Filter(
                must=[
                    qm.FieldCondition(key="visibility", match=qm.MatchValue(value="shared")),
                    qm.FieldCondition(key="shared_with", match=qm.MatchValue(value=agent_name)),
                ]
            ),
            qm.Filter(
                must=[
                    qm.FieldCondition(key="visibility", match=qm.MatchValue(value="private")),
                    qm.FieldCondition(key="agent_owner", match=qm.MatchValue(value=agent_name)),
                ]
            ),
        ],
    )


def search_memory(user_id: str, query: str, limit: int = 5, agent_name: str = None):
    """Vector search with native ACL filter (no Python post-filter).

    If agent_name is None, falls back to user_id-only filter (legacy behaviour).
    """
    if agent_name:
        flt = _acl_filter(user_id, agent_name)
    else:
        flt = qm.Filter(must=[
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        ])

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(query),
        query_filter=flt,
        limit=limit,
    )

    out = []
    for hit in results.points:
        if not hit.payload:
            continue
        out.append({
            "memory_id": hit.id,
            "text": hit.payload.get("text"),
            "similarity_score": round(hit.score, 3),
            "memory_type": hit.payload.get("memory_type", "Semantic"),
            "entities": hit.payload.get("entities", []),
            "agent_owner": hit.payload.get("agent_owner"),
            "visibility": hit.payload.get("visibility", "private"),
            "shared_with": hit.payload.get("shared_with", []),
        })
    return out
