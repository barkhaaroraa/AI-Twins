import logging
import threading
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
from app.llm.ollama_client import generate_response, OllamaUnavailable
from app.memory.graph import MemoryGraph
from app.memory.episodic import EpisodicMemory
from app.memory.retrieval import HybridRetriever
from app.memory.consolidation import ConsolidationEngine
from app.memory.evolution import MemoryEvolution
from app.memory.summarizer import summarize_memory, fast_store_payload
from app.memory.memory_updater import update_memory
from app.memory.vector import init_vector_collection, search_memory
from app.memory.intent_classifier import get_intent_classifier
from app.agents import AGENTS, AgentSpec
from twin.prompt_builder import PromptBuilder

log = logging.getLogger(__name__)


class TwinOrchestrator:
    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.episodic = EpisodicMemory(self.memory_graph)
        self.retriever = HybridRetriever(self.memory_graph, self.episodic)
        self.consolidation = ConsolidationEngine(self.memory_graph)
        self.evolution = MemoryEvolution(self.memory_graph)
        self.prompt_builder = PromptBuilder()
        self.intent_classifier = get_intent_classifier()
        # Track latest background results per user for the UI to poll
        self._bg_results: dict = {}

    def initialize(self):
        init_vector_collection()
        self.memory_graph.load_from_mongo()

    # ------------------------------------------------------------------
    # Main chat pipeline — optimised for speed
    #
    # FAST PATH (blocks the response):
    #   1. Ensure user exists + get profile (DB)
    #   2. Session management (DB)
    #   3. Quick keyword prefs/tasks (no LLM)
    #   4. Hybrid retrieval — rule-based intent + vector search + graph
    #   5. Build prompt + generate_response (single LLM call)
    #   6. Return immediately
    #
    # BACKGROUND (after response is sent):
    #   7. Memory extraction via LLM
    #   8. Store to MongoDB + Qdrant + Graph
    #   9. Evolution checks (contradiction LLM calls)
    #  10. Consolidation (if threshold met)
    #  11. Decay
    # ------------------------------------------------------------------

    def process_message(
        self,
        user_id: str,
        message: str,
        agent: Optional[AgentSpec] = None,
    ) -> dict:
        create_user_if_not_exists(user_id)
        session_id = self.episodic.get_or_create_session(user_id)
        self._apply_keyword_learning(user_id, message)

        # ML routing: should we even hit the memory pipeline?
        routing = self.intent_classifier.classify(message)
        needs_retrieval = routing["needs_retrieval"]
        should_store = routing["should_store"]
        if agent and agent.force_store:
            should_store = True
            routing["should_store"] = True
            routing["forced_by_agent"] = agent.name

        # Retrieval (only if the classifier thinks it'll help)
        if needs_retrieval:
            try:
                retrieval_result = self.retriever.retrieve(user_id, message, limit=4)
            except Exception:
                log.exception("retrieval failed; falling back to LLM-only")
                retrieval_result = {"query_intent": {}, "results": []}
        else:
            retrieval_result = {
                "query_intent": {"intent": "direct", "skipped": True},
                "results": [],
            }

        retrieval_result.setdefault("query_intent", {}).update(
            {
                "needs_retrieval": needs_retrieval,
                "should_store": should_store,
                "retrieval_score": routing["retrieval_score"],
                "store_score": routing["store_score"],
            }
        )

        # Build prompt + LLM call. Tolerate Ollama failures.
        user = get_user(user_id)
        prompt = self.prompt_builder.build_prompt(
            user_profile=user or {},
            retrieved_memories=retrieval_result["results"],
            message=message,
            agent_role=agent.role_prompt if agent else None,
        )

        ollama_error = None
        try:
            response = generate_response(prompt)
        except OllamaUnavailable as e:
            log.warning("Ollama unavailable: %s", e)
            ollama_error = str(e)
            response = (
                "I can't reach the local LLM right now, so I can't answer this turn. "
                "Your message will still be saved to memory if it contains anything noteworthy."
            )
        except Exception as e:
            log.exception("generate_response failed")
            ollama_error = str(e)
            response = "Something went wrong generating a reply, but your message was processed."

        # Background memory work (decay, store, evolution, consolidation)
        bg_thread = threading.Thread(
            target=self._background_memory_work,
            args=(
                user_id,
                message,
                session_id,
                should_store,
                needs_retrieval and not retrieval_result["results"],
                agent.name if agent else None,
            ),
            daemon=True,
        )
        bg_thread.start()

        prev_updates = self._bg_results.pop(user_id, None)

        out = {
            "response": response,
            "memory_used": retrieval_result["results"],
            "retrieval_trace": retrieval_result,
            "memory_updates": prev_updates or {
                "extracted": None,
                "graph_edges_added": 0,
                "contradictions": [],
                "merges": [],
                "consolidation_triggered": False,
                "status": "processing",
            },
            "session_id": session_id,
            "routing": routing,
            "agent": agent.name if agent else None,
        }
        if ollama_error:
            out["error"] = ollama_error
        return out

    def process_agent_message(
        self, user_id: str, message: str, agent_name: str
    ) -> dict:
        spec = AGENTS.get(agent_name)
        if spec is None:
            raise ValueError(
                f"Unknown agent '{agent_name}'. Available: {sorted(AGENTS)}"
            )
        return self.process_message(user_id, message, agent=spec)

    # ------------------------------------------------------------------
    # Background memory processing (runs after response is sent)
    # ------------------------------------------------------------------

    def _background_memory_work(
        self,
        user_id: str,
        message: str,
        session_id: str,
        should_store: bool,
        force_topic_node: bool,
        source_agent: Optional[str] = None,
    ):
        """Heavy/LLM work that doesn't block the user response.

        force_topic_node=True means retrieval was attempted but found nothing;
        we still record this query as a new topic node so the graph grows.
        """
        updates = {
            "extracted": None,
            "graph_edges_added": 0,
            "contradictions": [],
            "merges": [],
            "consolidation_triggered": False,
            "status": "completed",
        }
        try:
            summarized = None
            if should_store:
                # LLM extraction is best-effort; fall back to a minimal payload
                # so the message still becomes a node.
                try:
                    summarized = summarize_memory(message)
                except Exception:
                    log.exception("summarize_memory failed; using fast payload")
                if not summarized:
                    summarized = fast_store_payload(message)
            elif force_topic_node:
                summarized = fast_store_payload(message, memory_type="Episodic")

            if summarized:
                stored = update_memory(
                    user_id,
                    summarized,
                    session_id,
                    self.memory_graph,
                    source_agent=source_agent,
                )
                if stored:
                    memory_id = stored["_id"]
                    updates["extracted"] = {
                        "intent": summarized.get("intent"),
                        "memory_type": summarized.get("memory_type"),
                        "summary": summarized.get("summary"),
                        "entities": summarized.get("entities", []),
                    }
                    self.episodic.add_memory_to_session(session_id, memory_id, user_id)

                    # Cheap embedding-based evolution check (no LLM):
                    # only trigger LLM contradiction detection on near-duplicates.
                    semantic_neighbors = search_memory(user_id, message, limit=3)
                    near_dups = [
                        n for n in semantic_neighbors
                        if n.get("similarity_score", 0) >= 0.85
                        and n.get("memory_id") != memory_id
                    ]
                    if near_dups:
                        try:
                            evolution_result = self.evolution.check_and_evolve(
                                user_id, stored, near_dups
                            )
                            updates["contradictions"] = evolution_result["contradictions"]
                            updates["merges"] = evolution_result["merges"]
                        except Exception:
                            log.exception("evolution failed (non-fatal)")

                    updates["graph_edges_added"] = len(
                        self.memory_graph.get_neighbors_with_relations(memory_id)
                    )

                    self.consolidation.increment_counter()
                    if self.consolidation.should_consolidate():
                        try:
                            self.consolidation.run_consolidation(user_id)
                            updates["consolidation_triggered"] = True
                        except Exception:
                            log.exception("consolidation failed (non-fatal)")

                    self.retriever.invalidate_cache_for_user(user_id)

            self.evolution.apply_exponential_decay(user_id)

        except Exception:
            log.exception("Background memory work failed for user %s", user_id)
            updates["status"] = "error"

        self._bg_results[user_id] = updates

    # ------------------------------------------------------------------
    # Keyword-based preference/task learning (instant, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_keyword_learning(user_id: str, message: str):
        ml = message.lower()
        if "simple" in ml:
            update_preference(user_id, "tone", "simple")
        if "detailed" in ml or "in detail" in ml:
            update_preference(user_id, "tone", "detailed")
        if "short" in ml:
            update_preference(user_id, "length", "short")
        if "example" in ml or "examples" in ml:
            update_preference(user_id, "examples", True)
        if "working on" in ml:
            task_title = ml.split("working on")[-1].strip()
            add_or_merge_task(user_id, task_title)
        if "completed" in ml or "finished" in ml:
            task_title = message.split(" ")[-1].strip()
            complete_task(user_id, task_title)

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
