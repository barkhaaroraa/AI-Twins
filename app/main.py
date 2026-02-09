from fastapi import FastAPI
from pydantic import BaseModel
from app.memory.mongo import add_to_buffer, get_buffer, clear_buffer
from app.memory.summarizer import summarize_memories
from app.llm.ollama_client import generate_response
from app.memory.mongo import (
    create_user_if_not_exists,
    get_user,
    update_preference,
    add_task,
    complete_task
)
from app.memory.vector import (
    init_vector_collection,
    store_memory,
    search_memory
)

app = FastAPI(title="AI Twin System")

# Initialize Qdrant collection once at startup
init_vector_collection()


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    # Ensure user exists
    create_user_if_not_exists(request.user_id)

    message_lower = request.message.lower()

    # --------------------
    # Preference Learning
    # --------------------
    if "simple" in message_lower:
        update_preference(request.user_id, "tone", "simple")

    if "detailed" in message_lower or "in detail" in message_lower:
        update_preference(request.user_id, "tone", "detailed")

    if "short" in message_lower:
        update_preference(request.user_id, "length", "short")

    if "example" in message_lower or "examples" in message_lower:
        update_preference(request.user_id, "examples", True)

    # --------------------
    # Task Memory
    # --------------------
    if "working on" in message_lower:
        task_title = request.message.lower().split("working on")[-1].strip()
        add_task(request.user_id, task_title)

    if "completed" in message_lower or "finished" in message_lower:
        task_title = request.message.split(" ")[-1].strip()
        complete_task(request.user_id, task_title)

    # --------------------
    # Fetch Memory
    # --------------------
    user = get_user(request.user_id)
    preferences = user.get("preferences", {})
    tasks = user.get("tasks", [])

    semantic_memories = search_memory(
        request.user_id,
        request.message
    )

    # --------------------
    # Prompt Construction
    # --------------------
    prompt = f"""
You are an AI assistant with persistent memory.

User Preferences:
{preferences}

Active Tasks:
{tasks}

Relevant Past Context:
{semantic_memories}

Instructions:
- Use past context only if relevant.
- Do not repeat old information unnecessarily.
- Consider active tasks when responding.
- If the user asks for help, relate it to ongoing tasks if relevant.

User Message:
{request.message}
""".strip()

    # --------------------
    # Generate Response
    # --------------------
    response = generate_response(prompt)

    # --------------------
    # Add message to short-term buffer
    add_to_buffer(request.user_id, request.message)

    buffer = get_buffer(request.user_id)

# If buffer reaches threshold → summarize
    if len(buffer) >= 5:
        summary = summarize_memories(buffer)

        store_memory(
            request.user_id,
            summary
        )

        clear_buffer(request.user_id)

    return {"response": response}
