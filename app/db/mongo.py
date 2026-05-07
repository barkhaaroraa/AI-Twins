from datetime import datetime

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["ai_twin_db"]

users_collection = db["users"]
memory_collection = db["memories"]


def get_user(user_id: str):
    return users_collection.find_one({"user_id": user_id})


def create_user_if_not_exists(user_id: str):
    if not get_user(user_id):
        users_collection.insert_one({
            "user_id": user_id,
            "created_at": datetime.utcnow(),
        })


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
                "importance": 1,
                "created_at": 1,
            },
        )
    )
