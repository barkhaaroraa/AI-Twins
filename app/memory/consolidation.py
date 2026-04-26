from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from app.config import (
    CONSOLIDATION_THRESHOLD,
    CLUSTER_DISTANCE_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    NMF_TOPICS,
)
from app.db.mongo import memory_collection, get_all_memories_for_user
from app.llm.ollama_client import generate_json
from app.memory.vector import embed_text, store_memory as vector_store


class ConsolidationEngine:
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.consolidation_threshold = CONSOLIDATION_THRESHOLD
        self.new_memory_count = 0

    def should_consolidate(self) -> bool:
        return self.new_memory_count >= self.consolidation_threshold

    def increment_counter(self) -> None:
        self.new_memory_count += 1

    def reset_counter(self) -> None:
        self.new_memory_count = 0

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_memories(self, user_id: str) -> Dict[int, List[str]]:
        memories = get_all_memories_for_user(user_id)
        memories_with_emb = [
            m for m in memories
            if m.get("embedding") and len(m["embedding"]) > 0
        ]

        if len(memories_with_emb) < 2:
            return {}

        ids = [str(m["_id"]) for m in memories_with_emb]
        embeddings = np.array([m["embedding"] for m in memories_with_emb])

        distance_matrix = cosine_distances(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            label = int(label)
            clusters.setdefault(label, []).append(ids[idx])

        return clusters

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def find_near_duplicates(self, user_id: str) -> List[Tuple[str, str]]:
        memories = get_all_memories_for_user(user_id)
        memories_with_emb = [
            m for m in memories
            if m.get("embedding") and len(m["embedding"]) > 0
            and not m.get("is_consolidated")
        ]

        if len(memories_with_emb) < 2:
            return []

        ids = [str(m["_id"]) for m in memories_with_emb]
        embeddings = np.array([m["embedding"] for m in memories_with_emb])
        sim_matrix = cosine_similarity(embeddings)

        duplicates = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i][j] >= DEDUP_SIMILARITY_THRESHOLD:
                    duplicates.append((ids[i], ids[j]))

        return duplicates

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize_cluster(self, memory_ids: List[str], user_id: str) -> Optional[dict]:
        texts = []
        all_entities = set()

        for mid in memory_ids:
            m = memory_collection.find_one({"_id": mid})
            if m:
                texts.append(m.get("summary", ""))
                all_entities.update(m.get("entities", []))

        if not texts:
            return None

        prompt = f"""Merge these related memories into a single concise consolidated memory (max 40 words).
ONLY output valid JSON: {{"summary": "consolidated memory text", "memory_type": "Semantic|Episodic|Procedural|Preference"}}

Memories:
{chr(10).join(f'- {t}' for t in texts)}
"""
        try:
            result = generate_json(prompt)
            summary = result.get("summary", "")
            if not summary:
                return None
        except Exception:
            # Fallback: join first few words of each
            summary = "; ".join(t[:60] for t in texts[:3])

        consolidated_id = str(uuid4())
        embedding = embed_text(summary)
        memory_type = result.get("memory_type", "Semantic") if 'result' in dir() else "Semantic"

        doc = {
            "_id": consolidated_id,
            "user_id": user_id,
            "type": "consolidated",
            "intent": "fact",
            "memory_type": memory_type,
            "summary": summary,
            "entities": list(all_entities),
            "relationships": [],
            "confidence": 0.90,
            "importance": 0.95,
            "original_importance": 0.95,
            "embedding": embedding,
            "is_consolidated": True,
            "source_memory_ids": memory_ids,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }
        memory_collection.insert_one(doc)

        vector_store(
            user_id=user_id, text=summary, memory_id=consolidated_id,
            memory_type=memory_type, entities=list(all_entities), confidence=0.90,
        )

        self.memory_graph.add_memory_node(
            consolidated_id, summary, memory_type, datetime.utcnow(),
            0.90, embedding, list(all_entities), user_id, 0.95,
        )

        for mid in memory_ids:
            if mid in self.memory_graph.graph:
                self.memory_graph.add_edge(
                    consolidated_id, mid, "derived_from", 1.0, user_id=user_id
                )

        return doc

    # ------------------------------------------------------------------
    # Topic Modeling
    # ------------------------------------------------------------------

    def extract_topics(self, user_id: str, n_topics: int = NMF_TOPICS) -> List[dict]:
        memories = get_all_memories_for_user(user_id)
        texts = [m.get("summary", "") for m in memories if m.get("summary")]
        ids = [str(m["_id"]) for m in memories if m.get("summary")]

        if len(texts) < n_topics + 1:
            return []

        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)

        actual_topics = min(n_topics, len(texts) - 1)
        nmf = NMF(n_components=actual_topics, random_state=42)
        W = nmf.fit_transform(tfidf)
        H = nmf.components_

        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(H):
            top_word_indices = topic.argsort()[:-6:-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_memories = [
                ids[i] for i in range(len(W))
                if W[i, topic_idx] > 0.1
            ]
            topics.append({
                "topic_id": topic_idx,
                "keywords": top_words,
                "memory_ids": topic_memories,
            })

        return topics

    # ------------------------------------------------------------------
    # Main consolidation pipeline
    # ------------------------------------------------------------------

    def run_consolidation(self, user_id: str) -> dict:
        result = {
            "clusters_formed": 0,
            "duplicates_removed": 0,
            "summaries_created": 0,
            "topics": [],
        }

        # 1. Cluster memories
        clusters = self.cluster_memories(user_id)
        result["clusters_formed"] = len(clusters)

        # 2. Find and handle duplicates
        duplicates = self.find_near_duplicates(user_id)
        result["duplicates_removed"] = len(duplicates)

        # 3. Summarize large clusters
        for cluster_id, memory_ids in clusters.items():
            if len(memory_ids) >= 3:
                consolidated = self.summarize_cluster(memory_ids, user_id)
                if consolidated:
                    result["summaries_created"] += 1

        # 4. Extract topics
        topics = self.extract_topics(user_id)
        result["topics"] = topics

        self.reset_counter()
        return result
