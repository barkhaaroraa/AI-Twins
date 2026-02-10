from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid

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
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed_text(query),
        limit=limit
    )

    return [
        hit.payload["text"]
        for hit in results
        if hit.payload and hit.payload.get("user_id") == user_id
    ]
