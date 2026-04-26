from datetime import datetime
from typing import List, Optional
from app.db.mongo import db

graph_edges_collection = db["graph_edges"]
sessions_collection = db["sessions"]


# --- Edge CRUD ---

def upsert_edge(source: str, target: str, relation: str, weight: float,
                user_id: str, metadata: Optional[dict] = None) -> None:
    graph_edges_collection.update_one(
        {"source": source, "target": target, "relation": relation},
        {"$set": {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": weight,
            "metadata": metadata or {},
            "user_id": user_id,
            "updated_at": datetime.utcnow(),
        },
         "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )


def save_edges(edges: List[dict]) -> None:
    for e in edges:
        upsert_edge(
            source=e["source"],
            target=e["target"],
            relation=e["relation"],
            weight=e.get("weight", 1.0),
            user_id=e.get("user_id", ""),
            metadata=e.get("metadata"),
        )


def load_all_edges(user_id: Optional[str] = None) -> List[dict]:
    query = {"user_id": user_id} if user_id else {}
    return list(graph_edges_collection.find(query, {"_id": 0}))


def delete_edges_for_node(memory_id: str) -> int:
    result = graph_edges_collection.delete_many(
        {"$or": [{"source": memory_id}, {"target": memory_id}]}
    )
    return result.deleted_count


# --- Session CRUD ---

def create_session(session_id: str, user_id: str) -> dict:
    doc = {
        "_id": session_id,
        "user_id": user_id,
        "memory_ids": [],
        "start_time": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "is_active": True,
    }
    sessions_collection.insert_one(doc)
    return doc


def get_active_session(user_id: str) -> Optional[dict]:
    return sessions_collection.find_one(
        {"user_id": user_id, "is_active": True},
        sort=[("last_activity", -1)],
    )


def update_session_activity(session_id: str, memory_id: str) -> None:
    sessions_collection.update_one(
        {"_id": session_id},
        {
            "$push": {"memory_ids": memory_id},
            "$set": {"last_activity": datetime.utcnow()},
        },
    )


def close_session(session_id: str) -> None:
    sessions_collection.update_one(
        {"_id": session_id},
        {"$set": {"is_active": False, "end_time": datetime.utcnow()}},
    )


def get_user_sessions(user_id: str, limit: int = 10) -> List[dict]:
    return list(
        sessions_collection.find({"user_id": user_id})
        .sort("start_time", -1)
        .limit(limit)
    )
