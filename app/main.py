from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.db.mongo import get_user_memories, delete_memory
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
