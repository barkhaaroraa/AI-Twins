from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4

from app.db.mongo import memory_collection


def update_memory(
    user_id: str,
    summarized_memory: Optional[Dict]
) -> Optional[Dict]:
    """
    Persists summarized memory into MongoDB.
    Returns stored memory document or None.
    """

    if summarized_memory is None:
        return None

    memory_doc = {
        "_id": str(uuid4()),
        "user_id": user_id,
        "type": summarized_memory["type"],
        "summary": summarized_memory["summary"],
        "importance": summarized_memory["importance"],
        "created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow()
    }

    memory_collection.insert_one(memory_doc)

    return memory_doc
