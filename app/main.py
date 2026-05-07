from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.db.mongo import get_user_memories, delete_memory
from app.agents import AGENTS
from twin.orchestrator import TwinOrchestrator

API_TOKEN = "supersecrettoken123"


def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="AI Twin - Cognitive Memory Engine")
orchestrator = TwinOrchestrator()

# Static / UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def startup():
    orchestrator.initialize()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    return orchestrator.process_message(request.user_id, request.message)


# ------------------------------------------------------------------
# Multi-agent shared-memory endpoints
# ------------------------------------------------------------------

@app.get("/agents")
def list_agents():
    return {
        "agents": [
            {"name": a.name, "role": a.role_prompt.splitlines()[0]}
            for a in AGENTS.values()
        ]
    }


@app.post("/agent/{agent_name}")
def agent_chat(agent_name: str, request: ChatRequest):
    if agent_name not in AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent '{agent_name}'. Available: {sorted(AGENTS)}",
        )
    return orchestrator.process_agent_message(
        request.user_id, request.message, agent_name
    )


@app.get("/api/agent-memories/{user_id}")
def agent_memories(user_id: str):
    """Lean memory list for the Agents tab — no embeddings, sorted newest-first."""
    memories = get_user_memories(user_id)
    out = []
    for m in memories:
        out.append({
            "id": str(m.get("_id", "")),
            "summary": m.get("summary", ""),
            "memory_type": m.get("memory_type", m.get("type", "Semantic")),
            "intent": m.get("intent", m.get("type", "")),
            "entities": m.get("entities", []),
            "source_agent": m.get("source_agent"),
            "importance": m.get("importance", 0.5),
            "created_at": (
                m["created_at"].isoformat()
                if m.get("created_at") and hasattr(m["created_at"], "isoformat")
                else m.get("created_at")
            ),
        })
    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"memories": out}


# ------------------------------------------------------------------
# Memory CRUD (legacy, kept for compatibility)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# New API endpoints
# ------------------------------------------------------------------

@app.get("/api/graph/{user_id}")
def get_graph(user_id: str):
    return orchestrator.get_graph_data(user_id)


@app.get("/api/timeline/{user_id}")
def get_timeline(user_id: str):
    return orchestrator.get_timeline(user_id)


@app.post("/api/consolidate/{user_id}")
def consolidate(user_id: str):
    return orchestrator.trigger_consolidation(user_id)


@app.get("/api/memory/{memory_id}/details")
def memory_details(memory_id: str):
    return orchestrator.get_memory_details(memory_id)


@app.get("/api/topics/{user_id}")
def get_topics(user_id: str):
    return orchestrator.get_topics(user_id)


@app.get("/api/stats/{user_id}")
def get_stats(user_id: str):
    return orchestrator.get_stats(user_id)


@app.get("/api/memory-updates/{user_id}")
def get_memory_updates(user_id: str):
    """Poll for background memory processing results."""
    updates = orchestrator._bg_results.pop(user_id, None)
    if updates:
        return {"ready": True, "updates": updates}
    return {"ready": False}
