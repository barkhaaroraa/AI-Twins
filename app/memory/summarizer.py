from typing import Optional, Dict

MAX_SUMMARY_LENGTH = 120


def summarize_memory(
    user_message: str,
    ai_response: Optional[str] = None
) -> Optional[Dict]:
    """
    Converts raw interaction into structured, long-term memory.
    Stores abstracted user facts, not raw conversation.
    """

    message = user_message.lower().strip()

    if "i prefer" in message or "i like" in message:
        summary = extract_preference(user_message)
        return build_memory("preference", summary, 0.8)

    if "i am working on" in message or "i'm working on" in message:
        summary = extract_task(user_message)
        return build_memory("task", summary, 0.9)

    if "don't" in message or "do not" in message:
        summary = extract_constraint(user_message)
        return build_memory("constraint", summary, 0.7)

    if "i am a" in message or "i'm a" in message:
        summary = extract_background(user_message)
        return build_memory("background", summary, 0.6)

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

