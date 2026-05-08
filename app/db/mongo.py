from datetime import datetime

from pymongo import MongoClient, ASCENDING

from app.config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["ai_twin_db"]

users_collection = db["users"]
memory_collection = db["memories_warm"]                # was "memories"
raw_events_collection = db["raw_events"]
co_activation_log_collection = db["co_activation_log"]
concept_board_collection = db["concept_board"]


def init_indexes():
    """Idempotent index setup. Safe to call on every startup."""
    raw_events_collection.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
    memory_collection.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
    memory_collection.create_index([("user_id", ASCENDING), ("agent_owner", ASCENDING)])
    memory_collection.create_index([("user_id", ASCENDING), ("visibility", ASCENDING)])
    co_activation_log_collection.create_index([("user_id", ASCENDING), ("ts", ASCENDING)])


def get_user(user_id: str):
    return users_collection.find_one({"user_id": user_id})


def create_user_if_not_exists(user_id: str):
    users_collection.update_one(
        {"user_id": user_id},
        {"$setOnInsert": {"user_id": user_id, "created_at": datetime.utcnow()}},
        upsert=True,
    )


def get_user_memories(user_id: str):
    return list(
        memory_collection.find(
            {"user_id": user_id},
            {
                "_id": 1,
                "type": 1,
                "memory_type": 1,
                "intent": 1,
                "summary": 1,
                "entities": 1,
                "source_agent": 1,
                "agent_owner": 1,
                "visibility": 1,
                "shared_with": 1,
                "importance": 1,
                "confidence": 1,
                "tier": 1,
                "lineage": 1,
                "created_at": 1,
            },
        )
    )
