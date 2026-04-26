import re
from typing import Optional, Dict, List
from app.llm.ollama_client import generate_json


MAX_SUMMARY_LENGTH = 200

# Intent -> (memory_type, default_importance)
INTENT_DEFAULTS = {
    "task": ("Procedural", 0.9),
    "preference": ("Preference", 0.8),
    "fact": ("Semantic", 0.7),
    "correction": ("Semantic", 0.95),
    "goal": ("Procedural", 0.85),
    "contextual_reference": ("Episodic", 0.6),
}

# Rule-based intent patterns
INTENT_PATTERNS = {
    "correction": re.compile(
        r"\b(actually|no,|that'?s wrong|i meant|correction|not .+, but)\b", re.I
    ),
    "goal": re.compile(
        r"\b(i want to|my goal|planning to|aiming to|hope to|going to)\b", re.I
    ),
    "task": re.compile(
        r"\b(working on|building|developing|implementing|need to|todo|creating)\b", re.I
    ),
    "preference": re.compile(
        r"\b(i prefer|i like|i want|i enjoy|favou?rite|i love)\b", re.I
    ),
    "fact": re.compile(
        r"\b(i am a|i'm a|i have|i know|i studied|my .+ is|i work at|i live)\b", re.I
    ),
    "contextual_reference": re.compile(
        r"\b(last time|earlier|before|continue|as i said|remember when)\b", re.I
    ),
}


def summarize_memory(user_message: str) -> Optional[Dict]:
    """
    Intent-aware memory extraction pipeline.
    1. Try LLM-based structured extraction
    2. Fallback to rule-based extraction
    """
    llm_result = _llm_extract(user_message)
    if llm_result:
        return llm_result

    return _rule_based_extract(user_message)


# ------------------------------------------------------------------
# LLM-based extraction
# ------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an advanced memory extraction system for an AI Twin.
Analyze the user message and extract structured memory worth remembering long-term.

ONLY output valid JSON. DO NOT explain anything.

Schema:
{{
  "store": true or false,
  "intent": "task | preference | fact | correction | goal | contextual_reference",
  "memory_type": "Semantic | Episodic | Procedural | Preference",
  "summary": "abstracted memory in <= 30 words",
  "entities": ["entity1", "entity2"],
  "relationships": [{{"subject": "...", "predicate": "...", "object": "..."}}],
  "confidence": 0.0 to 1.0,
  "importance": 0.0 to 1.0
}}

Rules:
- Do NOT store raw conversation or greetings
- Do NOT store questions unless they reveal intent
- Store stable user facts, preferences, tasks, goals, corrections
- Extract ALL named entities (tools, languages, projects, people, topics)
- If nothing is worth storing, set store=false

User message:
\"\"\"{message}\"\"\"
"""


def _llm_extract(user_message: str) -> Optional[Dict]:
    try:
        result = generate_json(EXTRACTION_PROMPT.format(message=user_message))

        if not result.get("store"):
            return None

        intent = result.get("intent", "fact")
        default_type, default_imp = INTENT_DEFAULTS.get(intent, ("Semantic", 0.7))

        summary = (result.get("summary") or "").strip()
        if not summary:
            return None
        if len(summary) > MAX_SUMMARY_LENGTH:
            summary = summary[:MAX_SUMMARY_LENGTH] + "..."

        return {
            "type": intent,
            "intent": intent,
            "memory_type": result.get("memory_type", default_type),
            "summary": summary,
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", []),
            "confidence": float(result.get("confidence", 0.8)),
            "importance": float(result.get("importance", default_imp)),
        }
    except Exception:
        return None


# ------------------------------------------------------------------
# Rule-based fallback
# ------------------------------------------------------------------

def _rule_based_extract(user_message: str) -> Optional[Dict]:
    message_lower = user_message.lower().strip()

    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(message_lower):
            default_type, default_imp = INTENT_DEFAULTS[intent]
            summary = _clean_summary(user_message, intent)
            if not summary:
                continue
            entities = _extract_entities(user_message)
            return {
                "type": intent,
                "intent": intent,
                "memory_type": default_type,
                "summary": summary[:MAX_SUMMARY_LENGTH],
                "entities": entities,
                "relationships": [],
                "confidence": 0.6,
                "importance": default_imp,
            }

    return None


def _clean_summary(text: str, intent: str) -> str:
    """Strip common prefixes to get a cleaner summary."""
    removals = {
        "task": ["I am working on", "I'm working on", "working on"],
        "preference": ["I prefer", "i prefer", "I like", "i like"],
        "fact": ["I am a", "I'm a"],
        "correction": ["Actually,", "actually,", "No,", "no,"],
        "goal": ["I want to", "i want to", "My goal is to", "I'm planning to"],
        "contextual_reference": [],
    }
    result = text
    for prefix in removals.get(intent, []):
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
            break
    return result.strip()


def _extract_entities(text: str) -> List[str]:
    """Simple entity extraction: capitalized words, quoted strings, technical terms."""
    entities = set()

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"', text):
        entities.add(match.group(1))
    for match in re.finditer(r"'([^']+)'", text):
        entities.add(match.group(1))

    # Capitalized multi-word names (e.g. "Machine Learning", "FastAPI")
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        word = match.group(1)
        # Skip sentence-starting words by checking position
        if match.start() > 0 and text[match.start() - 1] not in ".!?\n":
            entities.add(word)

    # Technical terms (camelCase, PascalCase, contains digits/special)
    for match in re.finditer(r"\b([A-Za-z]+[A-Z][a-z]+[A-Za-z]*)\b", text):
        entities.add(match.group(1))

    # Known tech patterns
    for match in re.finditer(
        r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|React|Vue|Angular|"
        r"FastAPI|Django|Flask|Node\.?js|PyTorch|TensorFlow|MongoDB|PostgreSQL|"
        r"Docker|Kubernetes|AWS|GCP|Azure|Redis|GraphQL|REST)\b",
        text, re.I,
    ):
        entities.add(match.group(1))

    return list(entities)
