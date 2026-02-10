from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4

from app.db.mongo import memory_collection
from app.memory.vector import store_memory


def update_memory(
    user_id: str,
    summarized_memory: Optional[Dict]
) -> Optional[Dict]:

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

    # 1️⃣ Store in MongoDB
    memory_collection.insert_one(memory_doc)

    # 2️⃣ Store embedding in Qdrant  ← THIS WAS MISSING
    store_memory(
        user_id=memory_doc["user_id"],
        text=memory_doc["summary"]
    )


    return memory_doc

