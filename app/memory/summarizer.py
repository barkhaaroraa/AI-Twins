from typing import Optional, Dict
from app.llm.ollama_client import generate_response
import json


MAX_SUMMARY_LENGTH = 120


def summarize_memory(user_message: str):
    """
    Hybrid memory summarizer:
    1. Try LLM-based summarization
    2. Fallback to rule-based summarization
    """

    # 1️⃣ Try LLM-based memory extraction
    llm_memory = llm_summarize_memory(user_message)
    if llm_memory:
        return llm_memory

    # 2️⃣ Fallback to rule-based logic
    message = user_message.lower().strip()

    if "i prefer" in message or "i like" in message:
        return build_memory("preference", extract_preference(user_message), 0.8)

    if "i am working on" in message or "working on" in message:
        return build_memory("task", extract_task(user_message), 0.9)

    if "don't" in message or "do not" in message:
        return build_memory("constraint", extract_constraint(user_message), 0.7)

    if "i am a" in message or "i'm a" in message:
        return build_memory("background", extract_background(user_message), 0.6)

    return None



# -------------------------------
# Helper functions (internal use)
# -------------------------------

def build_memory(mem_type: str, summary: str, importance: float) -> Optional[Dict]:
    if not summary:
        return None

    summary = summary.strip()

    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH] + "..."

    return {
        "type": mem_type,
        "summary": summary,
        "importance": importance
    }


def extract_preference(text: str) -> str:
    return text.replace("I prefer", "").replace("i prefer", "").strip()


def extract_task(text: str) -> str:
    return text.replace("I am working on", "").replace("I'm working on", "").strip()


def extract_constraint(text: str) -> str:
    return text.replace("I don't", "").replace("I do not", "").strip()


def extract_background(text: str) -> str:
    return text.replace("I am a", "").replace("I'm a", "").strip()

def llm_summarize_memory(user_message: str) -> dict | None:
    """
    Uses LLM to decide whether the message should be stored as long-term memory.
    Returns structured memory or None.
    """

    prompt = f"""
You are a memory extraction system for an AI Twin.

Your job is to decide whether the user's message reveals
long-term information worth remembering.

ONLY output valid JSON.
DO NOT explain anything.

Memory schema:
{{
  "store": true or false,
  "type": "preference | task | constraint | background | none",
  "summary": "abstracted memory in <= 20 words",
  "importance": number between 0.0 and 1.0
}}

Rules:
- Do NOT store raw conversation
- Do NOT store questions
- Store only stable user facts
- If nothing is worth storing, set store=false

User message:
\"\"\"{user_message}\"\"\"
"""

    try:
        response = generate_response(prompt)
        memory = json.loads(response)

        if not memory.get("store"):
            return None

        return {
            "type": memory["type"],
            "summary": memory["summary"],
            "importance": float(memory["importance"])
        }

    except Exception as e:
        # Fallback safety
        return None
