from app.llm.ollama_client import generate_response

def summarize_memories(messages: list[str]) -> str:
    joined = "\n".join(messages)

    prompt = f"""
Summarize the following user interactions into ONE concise memory.
Focus on:
- goals
- ongoing projects
- preferences
- important facts

Ignore greetings and filler.

Interactions:
{joined}

Summary:
"""

    summary = generate_response(prompt)
    return summary.strip()
