from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from app.config import W_SEMANTIC, W_GRAPH, W_RECENCY, W_IMPORTANCE
from app.llm.ollama_client import generate_json
from app.memory.vector import search_memory_with_filter
from app.memory.cache import MemoryCache


class HybridRetriever:
    def __init__(self, memory_graph, episodic):
        self.memory_graph = memory_graph
        self.episodic = episodic
        self.w_semantic = W_SEMANTIC
        self.w_graph = W_GRAPH
        self.w_recency = W_RECENCY
        self.w_importance = W_IMPORTANCE
        self.cache = MemoryCache()

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_query_intent(self, query: str) -> dict:
        prompt = f"""Classify the intent of this user message for memory retrieval.
ONLY output valid JSON:
{{
  "intent": "question | continuation | recall | correction | exploration",
  "relevant_memory_types": ["Semantic", "Episodic", "Procedural", "Preference"],
  "temporal_hint": "recent | last_session | specific_date | null"
}}

User message: "{query}"
"""
        try:
            result = generate_json(prompt)
            return {
                "intent": result.get("intent", "question"),
                "relevant_memory_types": result.get(
                    "relevant_memory_types",
                    ["Semantic", "Episodic", "Procedural", "Preference"],
                ),
                "temporal_hint": result.get("temporal_hint"),
            }
        except Exception:
            return {
                "intent": "question",
                "relevant_memory_types": ["Semantic", "Episodic", "Procedural", "Preference"],
                "temporal_hint": None,
            }

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> dict:
        # Check cache
        cache_key = f"{user_id}:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Step 1: Classify query intent
        query_intent = self.classify_query_intent(query)

        # Step 2: Semantic search (broader pool)
        semantic_results = search_memory_with_filter(
            user_id, query,
            memory_types=query_intent["relevant_memory_types"],
            limit=10,
        )

        if not semantic_results:
            result = {
                "query_intent": query_intent,
                "results": [],
            }
            self.cache.put(cache_key, result)
            return result

        # Step 3: Graph expansion — BFS from top semantic seeds
        seed_ids = [r["memory_id"] for r in semantic_results[:5] if r.get("memory_id")]
        graph_neighbors = set()
        for seed in seed_ids:
            neighbors = self.memory_graph.bfs_neighbors(seed, max_hops=2)
            graph_neighbors.update(neighbors)

        # Step 4: Personalized PageRank
        pagerank_scores = self.memory_graph.personalized_pagerank(seed_ids)

        # Collect all candidate IDs
        all_candidate_ids = set(r["memory_id"] for r in semantic_results if r.get("memory_id"))
        all_candidate_ids.update(graph_neighbors)

        # Step 5: Score each candidate
        scored_results = []
        semantic_map = {r["memory_id"]: r for r in semantic_results if r.get("memory_id")}

        for cid in all_candidate_ids:
            node_data = self.memory_graph.get_node_data(cid)

            # Semantic score
            semantic_score = 0.0
            if cid in semantic_map:
                semantic_score = semantic_map[cid].get("similarity_score", 0.0)

            # Graph rank (normalized later)
            graph_rank = pagerank_scores.get(cid, 0.0)

            # Recency
            recency = 0.5
            if node_data and node_data.get("timestamp"):
                recency = self.episodic.compute_recency_weight(node_data["timestamp"])

            # Importance
            importance = 0.5
            if node_data:
                importance = node_data.get("importance", 0.5)

            scored_results.append({
                "memory_id": cid,
                "text": (node_data.get("content", "") if node_data
                         else semantic_map.get(cid, {}).get("text", "")),
                "memory_type": (node_data.get("memory_type", "Semantic") if node_data
                                else semantic_map.get(cid, {}).get("memory_type", "Semantic")),
                "entities": (node_data.get("entities", []) if node_data
                             else semantic_map.get(cid, {}).get("entities", [])),
                "raw_scores": {
                    "semantic": semantic_score,
                    "graph_rank": graph_rank,
                    "recency": recency,
                    "importance": importance,
                },
            })

        # Normalize graph_rank across candidates
        max_gr = max((r["raw_scores"]["graph_rank"] for r in scored_results), default=1.0)
        if max_gr > 0:
            for r in scored_results:
                r["raw_scores"]["graph_rank"] /= max_gr

        # Compute final scores
        for r in scored_results:
            s = r["raw_scores"]
            r["final_score"] = round(
                self.w_semantic * s["semantic"]
                + self.w_graph * s["graph_rank"]
                + self.w_recency * s["recency"]
                + self.w_importance * s["importance"],
                4,
            )
            r["score_breakdown"] = {
                "semantic": round(s["semantic"], 3),
                "graph_rank": round(s["graph_rank"], 3),
                "recency": round(s["recency"], 3),
                "importance": round(s["importance"], 3),
            }
            r["explanation"] = self._generate_explanation(r)
            del r["raw_scores"]

        # Sort and limit
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        scored_results = scored_results[:limit]

        result = {
            "query_intent": query_intent,
            "results": scored_results,
        }

        self.cache.put(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def _generate_explanation(self, result: dict) -> str:
        parts = []
        s = result["raw_scores"]

        if s["semantic"] >= 0.8:
            parts.append(f"Strong semantic match ({s['semantic']:.2f})")
        elif s["semantic"] >= 0.5:
            parts.append(f"Moderate semantic match ({s['semantic']:.2f})")

        if s["graph_rank"] >= 0.7:
            parts.append("Well-connected in memory graph")
        elif s["graph_rank"] >= 0.3:
            parts.append("Connected via graph relationships")

        if s["recency"] >= 0.9:
            parts.append("Very recent memory")
        elif s["recency"] >= 0.5:
            parts.append("Recent memory")
        else:
            parts.append("Older memory")

        if s["importance"] >= 0.8:
            parts.append("High importance")

        if result.get("entities"):
            ents = ", ".join(result["entities"][:3])
            parts.append(f"Entities: {ents}")

        return ". ".join(parts) + "." if parts else "Retrieved via hybrid search."

    def invalidate_cache_for_user(self, user_id: str) -> None:
        self.cache.invalidate_for_user(user_id)
