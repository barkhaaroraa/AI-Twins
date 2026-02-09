from pymongo import MongoClient
from datetime import datetime
import os

client = MongoClient("mongodb://localhost:27017")

db = client["ai_twin_db"]
users = db.users
users_collection = db["users"]

def get_user(user_id: str):
    return users_collection.find_one({"user_id": user_id})

def create_user_if_not_exists(user_id: str):
    user = get_user(user_id)
    if not user:
        users_collection.insert_one({
            "user_id": user_id,
            "preferences": {},
            "tasks": [],
            "created_at": datetime.utcnow()
        })
def update_preference(user_id: str, key: str, value: str):
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {f"preferences.{key}": value}}
    )
    

def get_preferences(user_id: str):
    user = users_collection.find_one({"user_id": user_id})
    if user:
        return user.get("preferences", {})
    return {}




def add_task(user_id: str, task_title: str):
    task_title = task_title.strip().lower()

    users.update_one(
        {
            "user_id": user_id,
            "tasks.title": {"$ne": task_title}
        },
        {
            "$push": {
                "tasks": {
                    "title": task_title,
                    "status": "ongoing",
                    "created_at": datetime.utcnow()
                }
            }
        }
    )




def complete_task(user_id: str, title: str):
    normalized_title = normalize_task_title(title)

    users_collection.update_one(
        {
            "user_id": user_id,
            "tasks.normalized_title": normalized_title
        },
        {
            "$set": {
                "tasks.$.status": "completed"
            }
        }
    )

def get_tasks(user_id: str):
    user = users_collection.find_one({"user_id": user_id})
    if user:
        return user.get("tasks", [])
    return []

def normalize_task_title(title: str) -> str:
    return " ".join(title.lower().strip().split())

def add_to_buffer(user_id: str, text: str):
    users_collection.update_one(
        {"user_id": user_id},
        {
            "$push": {
                "memory_buffer": text
            }
        }
    )


def get_buffer(user_id: str):
    user = users_collection.find_one({"user_id": user_id})
    return user.get("memory_buffer", [])


def clear_buffer(user_id: str):
    users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "memory_buffer": []
            }
        }
    )
