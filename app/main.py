from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.db.mongo import (
    create_user_if_not_exists,
    get_user,
    update_preference,
    complete_task,
    add_or_merge_task,
)

from app.memory.memory_updater import update_memory
from app.memory.summarizer import summarize_memory
from app.db.mongo import get_user_memories, delete_memory, decay_memory_importance
from app.llm.ollama_client import generate_response
from app.memory.vector import (
    init_vector_collection,
    search_memory
)

API_TOKEN = "supersecrettoken123"

def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

app = FastAPI(title="AI Twin System")

# Static / UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize Qdrant collection once
init_vector_collection()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ChatRequest(BaseModel):
    user_id: str
    message: str


def filter_relevant_memory(memories, threshold=0.75, max_items=2):
    return [
        m for m in memories
        if m["similarity_score"] >= threshold
    ][:max_items]


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        decay_memory_importance()

        # --------------------
        # Ensure user exists
        # --------------------
        create_user_if_not_exists(request.user_id)

        message_lower = request.message.lower()

        # --------------------
        # Memory Summarization (LONG-TERM)
        # --------------------
        summarized = summarize_memory(request.message)
        if summarized:
            update_memory(request.user_id, summarized)

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
            task_title = message_lower.split("working on")[-1].strip()
            add_or_merge_task(request.user_id, task_title)

        if "completed" in message_lower or "finished" in message_lower:
            task_title = request.message.split(" ")[-1].strip()
            complete_task(request.user_id, task_title)

        # --------------------
        # Fetch Persistent Memory
        # --------------------
        user = get_user(request.user_id)
        preferences = user.get("preferences", {})
        tasks = user.get("tasks", [])

        semantic_memories = search_memory(
            request.user_id,
            request.message
        )

        relevant_memories = filter_relevant_memory(semantic_memories)

        # --------------------
        # STRUCTURED PROMPT INJECTION
        # --------------------
        user_profile_block = []
        if preferences:
            user_profile_block.append(f"Preferences: {preferences}")
        if tasks:
            user_profile_block.append(
                "Active Tasks: " + ", ".join(t["title"] for t in tasks)
            )

        history_block = [
            m["text"] for m in relevant_memories
        ]

        prompt = f"""
User Profile:
{chr(10).join(f"- {p}" for p in user_profile_block) or "- None"}

Relevant History:
{chr(10).join(f"- {h}" for h in history_block) or "- None"}

User Message:
"{request.message}"

AI Response:
""".strip()

        # --------------------
        # Generate Response
        # --------------------
        response = generate_response(prompt)

        # --------------------
        # RETURN WITH EXPLAINABILITY
        # --------------------
        return {
            "response": response,
            "memory_used": relevant_memories
        }
    except Exception as e:
        return {"error": str(e), "response": "An error occurred", "memory_used": []}
    
@app.get("/memory/{user_id}")
def view_memory(user_id: str, auth: str = Depends(verify_token)):
    memories = get_user_memories(user_id)

    for m in memories:
        m["_id"] = str(m["_id"])

    return {"memories": memories}


@app.delete("/memory/{memory_id}")
def remove_memory(memory_id: str, auth: str = Depends(verify_token)):
    result = delete_memory(memory_id)

    if result.deleted_count == 0:
        return {"message": "Memory not found"}

    return {"message": "Memory deleted successfully"}
