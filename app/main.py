from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agents import AGENTS
from app.db.mongo import get_user_memories
from twin.orchestrator import TwinOrchestrator


app = FastAPI(title="AI Twin - Agentic Memory")
orchestrator = TwinOrchestrator()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def startup():
    orchestrator.initialize()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


class ChatRequest(BaseModel):
    user_id: str
    message: str


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
    """Lean memory list for agents — no embeddings, sorted newest-first."""
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
