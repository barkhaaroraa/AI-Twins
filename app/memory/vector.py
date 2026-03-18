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


def store_memory(user_id: str, text: str):
    if not text or not isinstance(text, str):
        return  # 🔒 DO NOTHING if summary is invalid

    vector = embed_text(text)

    point = {
        "id": str(uuid.uuid4()),  # always valid UUID
        "vector": vector,
        "payload": {
            "user_id": user_id,
            "text": text,
        },
    }

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )



def search_memory(user_id: str, query: str, limit: int = 3):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(query),
        limit=limit
    )

    explainable_results = []

    for hit in results.points:
        if not hit.payload:
            continue
        if hit.payload.get("user_id") != user_id:
            continue

        explainable_results.append({
            "text": hit.payload.get("text"),
            "similarity_score": round(hit.score, 3)
        })

    return explainable_results

