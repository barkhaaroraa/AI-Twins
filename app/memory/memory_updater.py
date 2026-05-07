from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from app.db.mongo import memory_collection
from app.memory.vector import embed_text, store_memory


def update_memory(
    user_id: str,
    summarized_memory: Optional[Dict],
    source_agent: Optional[str] = None,
) -> Optional[Dict]:

    if summarized_memory is None:
        return None

    memory_id = str(uuid4())
    embedding = embed_text(summarized_memory["summary"])

    memory_doc = {
        "_id": memory_id,
        "user_id": user_id,
        "type": summarized_memory.get("type", "fact"),
        "intent": summarized_memory.get("intent", summarized_memory.get("type", "fact")),
        "memory_type": summarized_memory.get("memory_type", "Semantic"),
        "summary": summarized_memory["summary"],
        "entities": summarized_memory.get("entities", []),
        "relationships": summarized_memory.get("relationships", []),
        "confidence": summarized_memory.get("confidence", 0.8),
        "importance": summarized_memory.get("importance", 0.5),
        "embedding": embedding,
        "source_agent": source_agent,
        "created_at": datetime.utcnow(),
    }

    memory_collection.insert_one(memory_doc)

    store_memory(
        user_id=user_id,
        text=memory_doc["summary"],
        memory_id=memory_id,
        memory_type=memory_doc["memory_type"],
        entities=memory_doc["entities"],
        confidence=memory_doc["confidence"],
    )

    return memory_doc
