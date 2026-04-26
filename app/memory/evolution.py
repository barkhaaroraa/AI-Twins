import math
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from app.config import DECAY_LAMBDA, DEDUP_SIMILARITY_THRESHOLD
from app.db.mongo import (
    memory_collection,
    update_memory_confidence,
    mark_memory_superseded,
    update_memory_importance,
    get_all_memories_for_user,
)
from app.llm.ollama_client import generate_json
from app.memory.vector import embed_text, store_memory as vector_store


class MemoryEvolution:
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def detect_contradiction(
        self, new_memory: dict, existing_memories: List[dict]
    ) -> List[dict]:
        if not existing_memories:
            return []

        contradictions = []
        new_summary = new_memory.get("summary", "")

        for existing in existing_memories:
            old_summary = existing.get("summary") or existing.get("text", "")
            if not old_summary:
                continue

            prompt = f"""You are a contradiction detector. Determine if these two statements contradict each other.
ONLY output valid JSON: {{"contradicts": true/false, "explanation": "brief reason"}}

Statement A (new): "{new_summary}"
Statement B (existing): "{old_summary}"
"""
            try:
                result = generate_json(prompt)
                if result.get("contradicts"):
                    contradictions.append({
                        "memory_id": existing.get("_id") or existing.get("memory_id"),
                        "text": old_summary,
                        "explanation": result.get("explanation", ""),
                    })
            except Exception:
                continue

        return contradictions

    # ------------------------------------------------------------------
    # Handle contradiction
    # ------------------------------------------------------------------

    def handle_contradiction(
        self, new_memory_id: str, old_memory_id: str, user_id: str
    ) -> None:
        # Add contradicts edge
        self.memory_graph.add_edge(
            new_memory_id, old_memory_id, "contradicts", 1.0,
            metadata={"resolved_at": datetime.utcnow().isoformat()},
            user_id=user_id,
        )

        # Reduce old memory confidence by 50%
        old_node = self.memory_graph.get_node_data(old_memory_id)
        if old_node:
            new_confidence = old_node.get("confidence", 1.0) * 0.5
            update_memory_confidence(old_memory_id, new_confidence)
            if old_memory_id in self.memory_graph.graph:
                self.memory_graph.graph.nodes[old_memory_id]["confidence"] = new_confidence

        mark_memory_superseded(old_memory_id, new_memory_id)

    # ------------------------------------------------------------------
    # Merge overlapping memories
    # ------------------------------------------------------------------

    def merge_overlapping(
        self, memory_id_a: str, memory_id_b: str, user_id: str
    ) -> Optional[str]:
        node_a = self.memory_graph.get_node_data(memory_id_a)
        node_b = self.memory_graph.get_node_data(memory_id_b)
        if not node_a or not node_b:
            return None

        prompt = f"""Merge these two overlapping memories into a single concise memory (max 30 words).
ONLY output valid JSON: {{"summary": "merged memory text"}}

Memory A: "{node_a.get('content', '')}"
Memory B: "{node_b.get('content', '')}"
"""
        try:
            result = generate_json(prompt)
            merged_summary = result.get("summary", "")
            if not merged_summary:
                return None
        except Exception:
            return None

        # Create merged memory
        merged_id = str(uuid4())
        merged_importance = max(
            node_a.get("importance", 0.5), node_b.get("importance", 0.5)
        )
        merged_confidence = max(
            node_a.get("confidence", 0.5), node_b.get("confidence", 0.5)
        )
        merged_entities = list(
            set(node_a.get("entities", []) + node_b.get("entities", []))
        )
        embedding = embed_text(merged_summary)

        # Store in MongoDB
        memory_collection.insert_one({
            "_id": merged_id,
            "user_id": user_id,
            "type": "merged",
            "intent": node_a.get("memory_type", "fact"),
            "memory_type": node_a.get("memory_type", "Semantic"),
            "summary": merged_summary,
            "entities": merged_entities,
            "relationships": [],
            "confidence": merged_confidence,
            "importance": merged_importance,
            "original_importance": merged_importance,
            "embedding": embedding,
            "is_consolidated": True,
            "source_memory_ids": [memory_id_a, memory_id_b],
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        })

        # Store in Qdrant
        vector_store(
            user_id=user_id,
            text=merged_summary,
            memory_id=merged_id,
            memory_type=node_a.get("memory_type", "Semantic"),
            entities=merged_entities,
            confidence=merged_confidence,
        )

        # Add to graph
        self.memory_graph.add_memory_node(
            merged_id, merged_summary,
            node_a.get("memory_type", "Semantic"),
            datetime.utcnow(), merged_confidence, embedding,
            merged_entities, user_id, merged_importance,
        )

        # derived_from edges
        self.memory_graph.add_edge(
            merged_id, memory_id_a, "derived_from", 1.0, user_id=user_id
        )
        self.memory_graph.add_edge(
            merged_id, memory_id_b, "derived_from", 1.0, user_id=user_id
        )

        return merged_id

    # ------------------------------------------------------------------
    # Exponential decay
    # ------------------------------------------------------------------

    def apply_exponential_decay(self, user_id: str = None) -> int:
        query = {"user_id": user_id} if user_id else {}
        memories = list(memory_collection.find(query))
        count = 0

        for m in memories:
            created_at = m.get("created_at", datetime.utcnow())
            original = m.get("original_importance", m.get("importance", 0.5))
            days_old = (datetime.utcnow() - created_at).total_seconds() / 86400.0
            new_importance = original * math.exp(-DECAY_LAMBDA * days_old)
            new_importance = max(0.01, round(new_importance, 4))

            if abs(new_importance - m.get("importance", 0)) > 0.001:
                update_memory_importance(m["_id"], new_importance)
                if m["_id"] in self.memory_graph.graph:
                    self.memory_graph.graph.nodes[m["_id"]]["importance"] = new_importance
                count += 1

        return count

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check_and_evolve(
        self, user_id: str, new_memory: dict, semantic_neighbors: List[dict]
    ) -> dict:
        result = {"contradictions": [], "merges": []}

        # Check contradictions
        contradictions = self.detect_contradiction(new_memory, semantic_neighbors)
        for c in contradictions:
            old_id = c["memory_id"]
            new_id = new_memory.get("_id") or new_memory.get("memory_id")
            if old_id and new_id:
                self.handle_contradiction(new_id, old_id, user_id)
                result["contradictions"].append(c)

        # Check for merge candidates (high similarity but not contradicting)
        contradicted_ids = {c["memory_id"] for c in contradictions}
        new_id = new_memory.get("_id") or new_memory.get("memory_id")

        for neighbor in semantic_neighbors:
            neighbor_id = neighbor.get("_id") or neighbor.get("memory_id")
            if not neighbor_id or neighbor_id in contradicted_ids:
                continue
            sim = neighbor.get("similarity_score", 0)
            if sim >= DEDUP_SIMILARITY_THRESHOLD and new_id:
                merged_id = self.merge_overlapping(new_id, neighbor_id, user_id)
                if merged_id:
                    result["merges"].append({
                        "merged_id": merged_id,
                        "source_ids": [new_id, neighbor_id],
                    })

        return result
