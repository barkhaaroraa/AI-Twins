"""LLM-driven concept induction.

Decoupled from Mongo/scheduler so it can be tested with a hand-rolled cluster.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from app.llm.ollama_client import OllamaUnavailable, generate_json

log = logging.getLogger(__name__)


ABSTRACTION_PROMPT = """You are a memory-consolidation system. Given the user memories below, induce a single concept that generalises them.

Output ONLY valid JSON, no commentary.

Schema:
{{
  "label": "short noun phrase (≤6 words)",
  "summary": "1-2 sentence summary of the concept",
  "confidence": 0.0 to 1.0
}}

Memories:
{members}
"""


def avg_pairwise_similarity(embeddings: List[List[float]]) -> float:
    if len(embeddings) < 2:
        return 0.0
    arr = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    arr = arr / np.where(norms == 0, 1, norms)[:, None]
    sim = arr @ arr.T
    n = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    return float(upper.mean()) if upper.size else 0.0


def induce_concept(members: List[Dict]) -> Optional[Dict]:
    """Return {label, summary, confidence} or None on LLM failure."""
    if not members:
        return None
    bullet_lines = "\n".join(f"- {m.get('summary', '')}" for m in members)
    try:
        result = generate_json(
            ABSTRACTION_PROMPT.format(members=bullet_lines),
            timeout=120,
        )
    except OllamaUnavailable:
        log.info("Ollama unavailable; skipping abstraction induction")
        return None
    except Exception:
        log.exception("induce_concept LLM call failed")
        return None

    label = (result.get("label") or "").strip()
    summary = (result.get("summary") or "").strip()
    if not label or not summary:
        return None
    return {
        "label": label,
        "summary": summary,
        "confidence": float(result.get("confidence", 0.7)),
    }
