import math
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from app.config import SESSION_TIMEOUT_MINUTES, DECAY_LAMBDA
from app.db.graph_store import (
    get_active_session,
    create_session,
    update_session_activity,
    close_session,
    get_user_sessions,
)
from app.db.mongo import memory_collection


class EpisodicMemory:
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.decay_lambda = DECAY_LAMBDA

    # ------------------------------------------------------------------
    # Time decay
    # ------------------------------------------------------------------

    def compute_recency_weight(self, created_at: datetime) -> float:
        days_since = (datetime.utcnow() - created_at).total_seconds() / 86400.0
        return math.exp(-self.decay_lambda * max(days_since, 0))

    def compute_time_score(
        self, relevance: float, created_at: datetime, importance: float
    ) -> float:
        recency = self.compute_recency_weight(created_at)
        return relevance * recency * importance

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_or_create_session(self, user_id: str) -> str:
        active = get_active_session(user_id)

        if active:
            last_activity = active.get("last_activity", datetime.utcnow())
            elapsed = (datetime.utcnow() - last_activity).total_seconds() / 60
            if elapsed < SESSION_TIMEOUT_MINUTES:
                return active["_id"]
            # Session timed out — close it
            close_session(active["_id"])

        session_id = str(uuid4())
        create_session(session_id, user_id)
        return session_id

    def add_memory_to_session(
        self, session_id: str, memory_id: str, user_id: str
    ) -> Optional[str]:
        """Add memory to session and create temporal_next edge.
        Returns the previous memory_id if one existed."""
        active = get_active_session(user_id)
        previous_memory_id = None

        if active and active.get("memory_ids"):
            previous_memory_id = active["memory_ids"][-1]

        update_session_activity(session_id, memory_id)

        if previous_memory_id:
            self.memory_graph.auto_link_by_session(
                memory_id, session_id, previous_memory_id
            )

        return previous_memory_id

    # ------------------------------------------------------------------
    # Temporal queries
    # ------------------------------------------------------------------

    def get_memories_in_timerange(
        self, user_id: str, start: datetime, end: datetime
    ) -> List[dict]:
        memories = list(
            memory_collection.find({
                "user_id": user_id,
                "created_at": {"$gte": start, "$lte": end},
            }).sort("created_at", -1)
        )
        for m in memories:
            m["_id"] = str(m["_id"])
        return memories

    def get_last_session_memories(self, user_id: str) -> List[dict]:
        sessions = get_user_sessions(user_id, limit=2)
        # Get the most recent completed (or only) session
        target = None
        for s in sessions:
            if not s.get("is_active", True):
                target = s
                break
        if not target and sessions:
            target = sessions[0]
        if not target:
            return []

        memory_ids = target.get("memory_ids", [])
        if not memory_ids:
            return []

        memories = list(
            memory_collection.find({"_id": {"$in": memory_ids}}).sort("created_at", 1)
        )
        for m in memories:
            m["_id"] = str(m["_id"])
        return memories

    def get_session_timeline(self, user_id: str, limit: int = 10) -> List[dict]:
        sessions = get_user_sessions(user_id, limit=limit)
        timeline = []

        for s in sessions:
            memory_ids = s.get("memory_ids", [])
            memories = []
            if memory_ids:
                raw = list(
                    memory_collection.find(
                        {"_id": {"$in": memory_ids}},
                        {
                            "_id": 1, "summary": 1, "memory_type": 1,
                            "type": 1, "created_at": 1, "importance": 1,
                        },
                    ).sort("created_at", 1)
                )
                for m in raw:
                    m["_id"] = str(m["_id"])
                memories = raw

            timeline.append({
                "session_id": s["_id"],
                "start_time": s.get("start_time", datetime.utcnow()).isoformat(),
                "last_activity": s.get("last_activity", datetime.utcnow()).isoformat(),
                "is_active": s.get("is_active", False),
                "memory_count": len(memory_ids),
                "memories": memories,
            })

        return timeline
