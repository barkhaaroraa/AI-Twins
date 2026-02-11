from pymongo import MongoClient
from datetime import datetime
from difflib import SequenceMatcher
from datetime import datetime, timedelta

# --------------------
# Database Setup
# --------------------

client = MongoClient("mongodb://localhost:27017")
db = client["ai_twin_db"]

users_collection = db["users"]
memory_collection = db["memories"]

# --------------------
# Helpers
# --------------------

def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold


# --------------------
# User Management
# --------------------

def get_user(user_id: str):
    return users_collection.find_one({"user_id": user_id})


def create_user_if_not_exists(user_id: str):
    if not get_user(user_id):
        users_collection.insert_one({
            "user_id": user_id,
            "preferences": {},
            "tasks": [],
            "memory_buffer": [],
            "created_at": datetime.utcnow()
        })


# --------------------
# Preferences
# --------------------

def update_preference(user_id: str, key: str, value: str):
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {f"preferences.{key}": value}}
    )


def get_preferences(user_id: str):
    user = get_user(user_id)
    return user.get("preferences", {}) if user else {}


# --------------------
# Tasks (with merge logic)
# --------------------

def add_task(user_id: str, task_title: str):
    task_title = normalize_text(task_title)

    users_collection.update_one(
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


def add_or_merge_task(user_id: str, new_task: str):
    user = get_user(user_id)
    if not user:
        return

    normalized_new = normalize_text(new_task)

    for task in user.get("tasks", []):
        if is_similar(normalize_text(task["title"]), normalized_new):
            return  # Same task → ignore duplicate

    add_task(user_id, new_task)


def complete_task(user_id: str, title: str):
    normalized_title = normalize_text(title)

    users_collection.update_one(
        {
            "user_id": user_id,
            "tasks.title": normalized_title
        },
        {
            "$set": {
                "tasks.$.status": "completed"
            }
        }
    )


def get_tasks(user_id: str):
    user = get_user(user_id)
    return user.get("tasks", []) if user else []


# --------------------
# Short-term Memory Buffer
# --------------------

def add_to_buffer(user_id: str, text: str):
    users_collection.update_one(
        {"user_id": user_id},
        {"$push": {"memory_buffer": text}}
    )


def get_buffer(user_id: str):
    user = get_user(user_id)
    return user.get("memory_buffer", []) if user else []


def clear_buffer(user_id: str):
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"memory_buffer": []}}
    )
# ... existing code ...

def cleanup_duplicate_tasks(user_id: str):
    user = get_user(user_id)
    if not user:
        return

    unique_tasks = []
    seen = []

    for task in user.get("tasks", []):
        title = normalize_text(task["title"])

        if any(is_similar(title, s) for s in seen):
            continue

        seen.append(title)
        unique_tasks.append(task)

    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"tasks": unique_tasks}}
    )
    
def get_user_memories(user_id: str):
    return list(
        memory_collection.find(
            {"user_id": user_id},
            {"_id": 1, "type": 1, "summary": 1, "importance": 1, "created_at": 1}
        )
    )


def delete_memory(memory_id: str):
    return memory_collection.delete_one({"_id": memory_id})





def decay_memory_importance():
    one_day_ago = datetime.utcnow() - timedelta(days=1)

    memories = memory_collection.find()

    for m in memories:
        days_old = (datetime.utcnow() - m["created_at"]).days

        # Reduce importance gradually
        new_importance = max(0, m["importance"] - (0.01 * days_old))

        memory_collection.update_one(
            {"_id": m["_id"]},
            {"$set": {"importance": new_importance}}
        )
