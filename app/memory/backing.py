from datetime import datetime
from typing import List, Optional

from ulid import ULID

from app.db.mongo import raw_events_collection


def append_event(
    user_id: str,
    agent_name: Optional[str],
    event_type: str,
    payload,
    produced_memory_ids: Optional[List[str]] = None,
) -> str:
    """Append-only autobiographical trace. Returns the event_id (ULID).

    event_type ∈ {user_message, agent_response, extraction_output, consolidation_event}
    """
    event_id = str(ULID())
    raw_events_collection.insert_one({
        "_id": event_id,
        "user_id": user_id,
        "agent_name": agent_name,
        "event_type": event_type,
        "timestamp": datetime.utcnow(),
        "payload": payload,
        "produced_memory_ids": produced_memory_ids or [],
    })
    return event_id


def attach_produced_memory(event_id: str, memory_id: str):
    raw_events_collection.update_one(
        {"_id": event_id},
        {"$addToSet": {"produced_memory_ids": memory_id}},
    )
