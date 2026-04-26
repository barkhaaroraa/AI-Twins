from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from app.config import GRAPH_SIMILARITY_THRESHOLD, GRAPH_MAX_HOPS, PAGERANK_ALPHA
from app.db.mongo import get_all_memories_for_user, memory_collection
from app.db.graph_store import load_all_edges, upsert_edge, delete_edges_for_node


class MemoryGraph:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_memory_node(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        timestamp: datetime,
        confidence: float,
        embedding: List[float],
        entities: List[str],
        user_id: str,
        importance: float = 0.5,
    ) -> None:
        clean_entities = []
        for e in (entities or []):
            if isinstance(e, str):
                clean_entities.append(e.lower())

        self.graph.add_node(
            memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            confidence=confidence,
            embedding=embedding,
            entities=clean_entities,
            user_id=user_id,
            importance=importance,
        )

    def remove_memory_node(self, memory_id: str) -> None:
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)
            delete_edges_for_node(memory_id)

    def get_node_data(self, memory_id: str) -> Optional[dict]:
        if memory_id not in self.graph:
            return None
        return dict(self.graph.nodes[memory_id])

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
        user_id: str = "",
        persist: bool = True,
    ) -> None:
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )
        if persist:
            upsert_edge(source_id, target_id, relation, weight, user_id, metadata)

    # ------------------------------------------------------------------
    # Auto-linking
    # ------------------------------------------------------------------

    def auto_link_by_entities(self, memory_id: str) -> List[str]:
        node = self.graph.nodes.get(memory_id)
        if not node or not node.get("entities"):
            return []

        node_entities = set(node["entities"])
        user_id = node["user_id"]
        linked = []

        for other_id, other_data in self.graph.nodes(data=True):
            if other_id == memory_id:
                continue
            if other_data.get("user_id") != user_id:
                continue
            other_entities = set(other_data.get("entities", []))
            shared = node_entities & other_entities
            if shared:
                weight = len(shared) / max(len(node_entities | other_entities), 1)
                self.add_edge(
                    memory_id, other_id, "related_to", weight,
                    metadata={"shared_entities": list(shared)},
                    user_id=user_id,
                )
                linked.append(other_id)

        return linked

    def auto_link_by_similarity(
        self, memory_id: str, threshold: float = GRAPH_SIMILARITY_THRESHOLD
    ) -> List[str]:
        node = self.graph.nodes.get(memory_id)
        if not node or not node.get("embedding"):
            return []

        user_id = node["user_id"]
        node_emb = np.array(node["embedding"]).reshape(1, -1)
        linked = []

        candidates = [
            (nid, ndata)
            for nid, ndata in self.graph.nodes(data=True)
            if nid != memory_id
            and ndata.get("user_id") == user_id
            and ndata.get("embedding")
        ]
        if not candidates:
            return []

        ids = [c[0] for c in candidates]
        embs = np.array([c[1]["embedding"] for c in candidates])
        sims = cosine_similarity(node_emb, embs)[0]

        for idx, sim in enumerate(sims):
            if sim >= threshold:
                other_id = ids[idx]
                if not self.graph.has_edge(memory_id, other_id):
                    self.add_edge(
                        memory_id, other_id, "related_to", float(sim),
                        user_id=user_id,
                    )
                    linked.append(other_id)

        return linked

    def auto_link_by_session(
        self, memory_id: str, session_id: str, previous_memory_id: Optional[str] = None
    ) -> None:
        if previous_memory_id and previous_memory_id in self.graph:
            node = self.graph.nodes.get(memory_id)
            user_id = node["user_id"] if node else ""
            self.add_edge(
                previous_memory_id, memory_id, "temporal_next", 1.0,
                metadata={"session_id": session_id},
                user_id=user_id,
            )

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def bfs_neighbors(self, memory_id: str, max_hops: int = GRAPH_MAX_HOPS) -> List[str]:
        if memory_id not in self.graph:
            return []
        visited = set()
        queue = deque([(memory_id, 0)])
        visited.add(memory_id)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        visited.discard(memory_id)
        return list(visited)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def personalized_pagerank(
        self, seed_ids: List[str], alpha: float = PAGERANK_ALPHA
    ) -> Dict[str, float]:
        if not seed_ids or self.graph.number_of_nodes() == 0:
            return {}

        valid_seeds = [s for s in seed_ids if s in self.graph]
        if not valid_seeds:
            return {}

        personalization = {
            node: (1.0 / len(valid_seeds) if node in valid_seeds else 0.0)
            for node in self.graph.nodes()
        }
        try:
            return nx.pagerank(
                self.graph, alpha=alpha, personalization=personalization,
                max_iter=100, tol=1e-06,
            )
        except nx.PowerIterationFailedConvergence:
            return {s: 1.0 / len(valid_seeds) for s in valid_seeds}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_from_mongo(self, user_id: Optional[str] = None) -> None:
        self.graph.clear()

        query = {"user_id": user_id} if user_id else {}
        memories = list(memory_collection.find(query))

        for m in memories:
            self.add_memory_node(
                memory_id=m["_id"],
                content=m.get("summary", ""),
                memory_type=m.get("memory_type", m.get("type", "Semantic")),
                timestamp=m.get("created_at", datetime.utcnow()),
                confidence=m.get("confidence", 1.0),
                embedding=m.get("embedding", []),
                entities=m.get("entities", []),
                user_id=m.get("user_id", ""),
                importance=m.get("importance", 0.5),
            )

        edges = load_all_edges(user_id)
        for e in edges:
            src, tgt = e["source"], e["target"]
            if src in self.graph and tgt in self.graph:
                self.graph.add_edge(
                    src, tgt,
                    relation=e["relation"],
                    weight=e.get("weight", 1.0),
                    metadata=e.get("metadata", {}),
                )

    def save_to_mongo(self) -> None:
        from app.db.graph_store import save_edges as _save
        edges = []
        for u, v, data in self.graph.edges(data=True):
            src_node = self.graph.nodes.get(u, {})
            edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "related_to"),
                "weight": data.get("weight", 1.0),
                "metadata": data.get("metadata", {}),
                "user_id": src_node.get("user_id", ""),
            })
        _save(edges)

    # ------------------------------------------------------------------
    # Export for frontend
    # ------------------------------------------------------------------

    def get_graph_json(self, user_id: str) -> dict:
        nodes = []
        for nid, ndata in self.graph.nodes(data=True):
            if ndata.get("user_id") != user_id:
                continue
            nodes.append({
                "id": nid,
                "label": (ndata.get("content", "")[:50] + "...")
                         if len(ndata.get("content", "")) > 50
                         else ndata.get("content", ""),
                "type": ndata.get("memory_type", "Semantic"),
                "confidence": ndata.get("confidence", 1.0),
                "importance": ndata.get("importance", 0.5),
                "entities": ndata.get("entities", []),
                "timestamp": ndata.get("timestamp", datetime.utcnow()).isoformat(),
            })

        node_ids = {n["id"] for n in nodes}
        edges = []
        for u, v, edata in self.graph.edges(data=True):
            if u in node_ids and v in node_ids:
                edges.append({
                    "from": u,
                    "to": v,
                    "relation": edata.get("relation", "related_to"),
                    "weight": edata.get("weight", 1.0),
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

    def get_neighbors_with_relations(self, memory_id: str) -> List[dict]:
        if memory_id not in self.graph:
            return []
        result = []
        for neighbor in self.graph.successors(memory_id):
            edata = self.graph.edges[memory_id, neighbor]
            ndata = self.graph.nodes[neighbor]
            result.append({
                "memory_id": neighbor,
                "content": ndata.get("content", ""),
                "relation": edata.get("relation", "related_to"),
                "weight": edata.get("weight", 1.0),
                "direction": "outgoing",
            })
        for neighbor in self.graph.predecessors(memory_id):
            edata = self.graph.edges[neighbor, memory_id]
            ndata = self.graph.nodes[neighbor]
            result.append({
                "memory_id": neighbor,
                "content": ndata.get("content", ""),
                "relation": edata.get("relation", "related_to"),
                "weight": edata.get("weight", 1.0),
                "direction": "incoming",
            })
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
