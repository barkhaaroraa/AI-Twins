import os
import uuid

# Prevent transformers from importing TensorFlow (which can break on some envs)
os.environ["TRANSFORMERS_NO_TF"] = "1"

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant via Docker (HTTP mode – STABLE)
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "user_memory"
VECTOR_SIZE = 384


def init_vector_collection():
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "size": VECTOR_SIZE,
                "distance": "Cosine"
            }
        )


def embed_text(text: str):
    return embedder.encode(text).tolist()


def batch_embed(texts: list) -> list:
    return embedder.encode(texts).tolist()


def store_memory(user_id: str, text: str, memory_id: str = None,
                 memory_type: str = None, entities: list = None,
                 confidence: float = None, session_id: str = None):
    if not text or not isinstance(text, str):
        return None

    vector = embed_text(text)
    point_id = memory_id or str(uuid.uuid4())

    payload = {
        "user_id": user_id,
        "text": text,
    }
    if memory_type:
        payload["memory_type"] = memory_type
    if entities:
        payload["entities"] = entities
    if confidence is not None:
        payload["confidence"] = confidence
    if session_id:
        payload["session_id"] = session_id

    point = {
        "id": point_id,
        "vector": vector,
        "payload": payload,
    }

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    return vector


def search_memory(user_id: str, query: str, limit: int = 5):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(query),
        limit=limit * 3,
    )

    explainable_results = []

    for hit in results.points:
        if not hit.payload:
            continue
        if hit.payload.get("user_id") != user_id:
            continue

        explainable_results.append({
            "memory_id": hit.id,
            "text": hit.payload.get("text"),
            "similarity_score": round(hit.score, 3),
            "memory_type": hit.payload.get("memory_type", "Semantic"),
            "entities": hit.payload.get("entities", []),
        })

        if len(explainable_results) >= limit:
            break

    return explainable_results


def search_memory_with_filter(user_id: str, query: str,
                              memory_types: list = None, limit: int = 10):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(query),
        limit=limit * 3,
    )

    filtered = []
    for hit in results.points:
        if not hit.payload:
            continue
        if hit.payload.get("user_id") != user_id:
            continue
        if memory_types and hit.payload.get("memory_type") not in memory_types:
            continue

        filtered.append({
            "memory_id": hit.id,
            "text": hit.payload.get("text"),
            "similarity_score": round(hit.score, 3),
            "memory_type": hit.payload.get("memory_type", "Semantic"),
            "entities": hit.payload.get("entities", []),
        })

        if len(filtered) >= limit:
            break

    return filtered

