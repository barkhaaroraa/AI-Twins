"""Personalized PageRank — Python power-method kernel.

Used by cascade Stage 3 over a Neo4j-fetched subgraph. The subgraph is small
(≤ a few hundred nodes for a reasonable user), so we keep the implementation
simple: build a sparse-ish adjacency in numpy, run 8 iterations.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from app.graph.schema import EDGE_WEIGHTS

ALPHA_RESTART = 0.15
MAX_ITER = 8
EPS = 1e-3


def _edge_weight(edge_type: str, stored_weight: float) -> float:
    """CO_ACTIVATED uses stored learned weight; others fall back to schema constants."""
    if edge_type == "CO_ACTIVATED":
        return float(stored_weight or 0.5)
    return float(EDGE_WEIGHTS.get(edge_type, stored_weight or 0.5))


def power_method_ppr(
    nodes: List[str],
    edges: List[Dict],
    seeds: Dict[str, float],
    alpha: float = ALPHA_RESTART,
    max_iter: int = MAX_ITER,
    eps: float = EPS,
) -> Dict[str, float]:
    """Iterative weighted PPR. Returns {memory_id: activation}."""
    if not nodes:
        return dict(seeds)

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build weighted out-edge map: src_idx -> list of (dst_idx, weight, sign).
    out_edges: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for e in edges:
        src = idx.get(e.get("src"))
        dst = idx.get(e.get("dst"))
        if src is None or dst is None or src == dst:
            continue
        w = _edge_weight(e.get("type", ""), e.get("weight", 0.5))
        # Treat as undirected by adding both directions (the schema already encodes
        # directed semantics as separate edge types).
        out_edges[src].append((dst, w))
        out_edges[dst].append((src, w))

    # Seed vector (normalised).
    seed_vec = np.zeros(n, dtype=np.float64)
    for mid, s in seeds.items():
        if mid in idx:
            seed_vec[idx[mid]] = max(seed_vec[idx[mid]], float(s))
    if seed_vec.sum() <= 0:
        return dict(seeds)
    seed_vec /= seed_vec.sum()

    activation = seed_vec.copy()
    for _ in range(max_iter):
        new_act = alpha * seed_vec
        for src in range(n):
            val = activation[src]
            if val <= 0:
                continue
            outs = out_edges.get(src)
            if not outs:
                # Sink: route restart probability fully to seed.
                new_act += (1.0 - alpha) * val * seed_vec
                continue
            total_w = sum(abs(w) for _, w in outs) or 1.0
            for dst, w in outs:
                # Negative edges (CONTRADICTS) suppress activation.
                new_act[dst] += (1.0 - alpha) * val * (w / total_w)
        # Clamp negatives to 0 — activation never goes below baseline.
        np.clip(new_act, 0.0, None, out=new_act)
        if np.max(np.abs(new_act - activation)) < eps:
            activation = new_act
            break
        activation = new_act

    return {nodes[i]: float(activation[i]) for i in range(n) if activation[i] > 0}
