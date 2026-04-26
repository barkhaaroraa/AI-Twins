from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4

from app.db.mongo import memory_collection
from app.memory.vector import store_memory, embed_text


def update_memory(
    user_id: str,
    summarized_memory: Optional[Dict],
    session_id: str = None,
    memory_graph=None,
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
        "original_importance": summarized_memory.get("importance", 0.5),
        "embedding": embedding,
        "session_id": session_id,
        "is_consolidated": False,
        "superseded_by": None,
        "created_at": datetime.utcnow(),
        "last_accessed": None,
        "last_updated": datetime.utcnow(),
    }

    # 1. Store in MongoDB
    memory_collection.insert_one(memory_doc)

    # 2. Store in Qdrant with metadata
    store_memory(
        user_id=user_id,
        text=memory_doc["summary"],
        memory_id=memory_id,
        memory_type=memory_doc["memory_type"],
        entities=memory_doc["entities"],
        confidence=memory_doc["confidence"],
        session_id=session_id,
    )

    # 3. Add to knowledge graph and auto-link
    if memory_graph is not None:
        memory_graph.add_memory_node(
            memory_id=memory_id,
            content=memory_doc["summary"],
            memory_type=memory_doc["memory_type"],
            timestamp=memory_doc["created_at"],
            confidence=memory_doc["confidence"],
            embedding=embedding,
            entities=memory_doc["entities"],
            user_id=user_id,
            importance=memory_doc["importance"],
        )
        memory_graph.auto_link_by_entities(memory_id)
        memory_graph.auto_link_by_similarity(memory_id)

    return memory_doc

