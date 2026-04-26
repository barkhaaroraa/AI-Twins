from datetime import datetime
from typing import Optional

from app.db.mongo import (
    create_user_if_not_exists,
    get_user,
    update_preference,
    add_or_merge_task,
    complete_task,
    get_memory_by_id,
)
from app.llm.ollama_client import generate_response
from app.memory.graph import MemoryGraph
from app.memory.episodic import EpisodicMemory
from app.memory.retrieval import HybridRetriever
from app.memory.consolidation import ConsolidationEngine
from app.memory.evolution import MemoryEvolution
from app.memory.summarizer import summarize_memory
from app.memory.memory_updater import update_memory
from app.memory.vector import init_vector_collection, search_memory
from twin.prompt_builder import PromptBuilder


class TwinOrchestrator:
    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.episodic = EpisodicMemory(self.memory_graph)
        self.retriever = HybridRetriever(self.memory_graph, self.episodic)
        self.consolidation = ConsolidationEngine(self.memory_graph)
        self.evolution = MemoryEvolution(self.memory_graph)
        self.prompt_builder = PromptBuilder()

    def initialize(self):
        init_vector_collection()
        self.memory_graph.load_from_mongo()

    # ------------------------------------------------------------------
    # Main chat pipeline
    # ------------------------------------------------------------------

    def process_message(self, user_id: str, message: str) -> dict:
        try:
            # Apply exponential decay
            self.evolution.apply_exponential_decay(user_id)

            # Ensure user exists
            create_user_if_not_exists(user_id)

            # Session management
            session_id = self.episodic.get_or_create_session(user_id)

            # Preference / task learning (simple keyword-based, kept from original)
            message_lower = message.lower()
            if "simple" in message_lower:
                update_preference(user_id, "tone", "simple")
            if "detailed" in message_lower or "in detail" in message_lower:
                update_preference(user_id, "tone", "detailed")
            if "short" in message_lower:
                update_preference(user_id, "length", "short")
            if "example" in message_lower or "examples" in message_lower:
                update_preference(user_id, "examples", True)
            if "working on" in message_lower:
                task_title = message_lower.split("working on")[-1].strip()
                add_or_merge_task(user_id, task_title)
            if "completed" in message_lower or "finished" in message_lower:
                task_title = message.split(" ")[-1].strip()
                complete_task(user_id, task_title)

            # Memory extraction (intent-aware)
            memory_updates = {"extracted": None, "graph_edges_added": 0,
                              "contradictions": [], "merges": [],
                              "consolidation_triggered": False}

            summarized = summarize_memory(message)
            if summarized:
                stored = update_memory(
                    user_id, summarized, session_id, self.memory_graph
                )
                if stored:
                    memory_id = stored["_id"]
                    memory_updates["extracted"] = {
                        "intent": summarized.get("intent"),
                        "memory_type": summarized.get("memory_type"),
                        "summary": summarized.get("summary"),
                        "entities": summarized.get("entities", []),
                    }

                    # Session tracking
                    self.episodic.add_memory_to_session(session_id, memory_id, user_id)

                    # Evolution check
                    semantic_neighbors = search_memory(user_id, message, limit=5)
                    evolution_result = self.evolution.check_and_evolve(
                        user_id, stored, semantic_neighbors
                    )
                    memory_updates["contradictions"] = evolution_result["contradictions"]
                    memory_updates["merges"] = evolution_result["merges"]

                    # Count edges added
                    memory_updates["graph_edges_added"] = len(
                        self.memory_graph.get_neighbors_with_relations(memory_id)
                    )

                    # Consolidation check
                    self.consolidation.increment_counter()
                    if self.consolidation.should_consolidate():
                        self.consolidation.run_consolidation(user_id)
                        memory_updates["consolidation_triggered"] = True

                    # Invalidate retrieval cache
                    self.retriever.invalidate_cache_for_user(user_id)

            # Hybrid retrieval
            retrieval_result = self.retriever.retrieve(user_id, message, limit=5)

            # Build prompt
            user = get_user(user_id)
            prompt = self.prompt_builder.build_prompt(
                user_profile=user or {},
                retrieved_memories=retrieval_result["results"],
                message=message,
            )

            # Generate response
            response = generate_response(prompt)

            return {
                "response": response,
                "memory_used": retrieval_result["results"],
                "retrieval_trace": retrieval_result,
                "memory_updates": memory_updates,
                "session_id": session_id,
            }

        except Exception as e:
            return {
                "error": str(e),
                "response": "An error occurred while processing your message.",
                "memory_used": [],
                "retrieval_trace": {"query_intent": {}, "results": []},
                "memory_updates": {},
                "session_id": None,
            }

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def get_graph_data(self, user_id: str) -> dict:
        return self.memory_graph.get_graph_json(user_id)

    def get_timeline(self, user_id: str) -> dict:
        return {"sessions": self.episodic.get_session_timeline(user_id)}

    def trigger_consolidation(self, user_id: str) -> dict:
        return self.consolidation.run_consolidation(user_id)

    def get_memory_details(self, memory_id: str) -> dict:
        node_data = self.memory_graph.get_node_data(memory_id)
        connections = self.memory_graph.get_neighbors_with_relations(memory_id)
        mongo_data = get_memory_by_id(memory_id)

        if mongo_data:
            mongo_data["_id"] = str(mongo_data["_id"])

        return {
            "memory": mongo_data,
            "graph_node": node_data,
            "connections": connections,
        }

    def get_topics(self, user_id: str) -> dict:
        return {"topics": self.consolidation.extract_topics(user_id)}

    def get_stats(self, user_id: str) -> dict:
        from app.db.mongo import get_all_memories_for_user
        memories = get_all_memories_for_user(user_id)

        type_breakdown = {}
        for m in memories:
            mtype = m.get("memory_type", m.get("type", "unknown"))
            type_breakdown[mtype] = type_breakdown.get(mtype, 0) + 1

        graph_data = self.memory_graph.get_graph_json(user_id)
        sessions = self.episodic.get_session_timeline(user_id, limit=100)

        return {
            "total_memories": len(memories),
            "graph_nodes": graph_data["stats"]["total_nodes"],
            "graph_edges": graph_data["stats"]["total_edges"],
            "sessions": len(sessions),
            "types_breakdown": type_breakdown,
            "cache_stats": self.retriever.cache.stats(),
        }
