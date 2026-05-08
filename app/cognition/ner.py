"""Rule-based entity extraction. Shared between summariser write-side and cascade entity-seed."""
import re
from typing import Dict, List


_TECH_TERMS = re.compile(
    r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|React|Vue|Angular|"
    r"FastAPI|Django|Flask|Node\.?js|PyTorch|TensorFlow|MongoDB|PostgreSQL|"
    r"Docker|Kubernetes|AWS|GCP|Azure|Redis|GraphQL|REST|Neo4j|Qdrant|Ollama)\b",
    re.I,
)
_QUOTED_DOUBLE = re.compile(r'"([^"]+)"')
_QUOTED_SINGLE = re.compile(r"'([^']+)'")
_PASCAL_OR_TITLE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_CAMEL = re.compile(r"\b([A-Za-z]+[A-Z][a-z]+[A-Za-z]*)\b")


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Returns list[{name, kind}]. Kind is heuristic: tool|topic."""
    out: Dict[str, str] = {}

    for m in _TECH_TERMS.finditer(text):
        out[m.group(1)] = "tool"
    for m in _QUOTED_DOUBLE.finditer(text):
        out.setdefault(m.group(1), "topic")
    for m in _QUOTED_SINGLE.finditer(text):
        out.setdefault(m.group(1), "topic")
    for m in _PASCAL_OR_TITLE.finditer(text):
        word = m.group(1)
        if m.start() > 0 and text[m.start() - 1] not in ".!?\n":
            out.setdefault(word, "topic")
    for m in _CAMEL.finditer(text):
        out.setdefault(m.group(1), "tool")

    return [{"name": n, "kind": k} for n, k in out.items()]


def entity_names(text: str) -> List[str]:
    return [e["name"] for e in extract_entities(text)]
