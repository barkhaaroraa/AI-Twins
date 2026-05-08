import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agents import AGENTS
from app.cognition import blackboard_registry
from app.cognition.consolidator import heavy as heavy_consolidator
from app.cognition.consolidator.light import (
    LightConsolidator, get_consolidator, idle_for_seconds, is_idle, IDLE_THRESHOLD_SEC,
)
from app.db.mongo import concept_board_collection, get_user_memories, init_indexes
from app.graph import neo4j_client
from app.graph.queries import lineage_neighbours
from twin.orchestrator import TwinOrchestrator

log = logging.getLogger(__name__)


app = FastAPI(title="AI Twin - HSC Memory")
orchestrator = TwinOrchestrator()
scheduler = BackgroundScheduler(daemon=True)
light: LightConsolidator = get_consolidator()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def _heavy_run_unconditional():
    log.info("heavy consolidator: triggered (cron 03:00)")
    try:
        heavy_consolidator.run_for_all_users()
    except Exception:
        log.exception("heavy consolidator failed")


def _heavy_run_if_idle_30min():
    if idle_for_seconds() < 1800:
        return
    log.info("heavy consolidator: triggered (idle ≥30min)")
    try:
        heavy_consolidator.run_for_all_users()
    except Exception:
        log.exception("heavy consolidator failed")


@app.on_event("startup")
def startup():
    init_indexes()
    neo4j_client.init_schema()
    orchestrator.initialize()
    blackboard_registry.start_decay_tick()
    light.start()
    scheduler.add_job(_heavy_run_unconditional, CronTrigger(hour=3),
                      coalesce=True, max_instances=1, id="heavy-cron")
    scheduler.add_job(_heavy_run_if_idle_30min, IntervalTrigger(minutes=30),
                      coalesce=True, max_instances=1, id="heavy-idle")
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    light.stop()
    blackboard_registry.stop_decay_tick()
    neo4j_client.close()


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
    """Lean memory list — no embeddings, newest-first, includes ACL/tier/lineage."""
    memories = get_user_memories(user_id)
    out = []
    for m in memories:
        out.append({
            "id": str(m.get("_id", "")),
            "summary": m.get("summary", ""),
            "memory_type": m.get("memory_type", m.get("type", "Semantic")),
            "intent": m.get("intent", m.get("type", "")),
            "entities": m.get("entities", []),
            "source_agent": m.get("source_agent") or m.get("agent_owner"),
            "agent_owner": m.get("agent_owner"),
            "visibility": m.get("visibility", "private"),
            "shared_with": m.get("shared_with", []),
            "tier": m.get("tier", "warm"),
            "importance": m.get("importance", 0.5),
            "confidence": m.get("confidence", 0.8),
            "lineage": m.get("lineage", {}),
            "created_at": (
                m["created_at"].isoformat()
                if m.get("created_at") and hasattr(m["created_at"], "isoformat")
                else m.get("created_at")
            ),
        })
    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"memories": out}


@app.get("/api/concepts/{user_id}")
def concepts(user_id: str):
    docs = list(concept_board_collection.find(
        {"user_id": user_id},
        {"_id": 1, "label": 1, "summary": 1, "confidence": 1, "member_ids": 1, "induced_at": 1},
    ).sort("induced_at", -1))
    out = []
    for d in docs:
        out.append({
            "id": d["_id"],
            "label": d.get("label", ""),
            "summary": d.get("summary", ""),
            "confidence": d.get("confidence", 0.7),
            "member_ids": d.get("member_ids", []),
            "induced_at": d["induced_at"].isoformat() if hasattr(d.get("induced_at"), "isoformat") else d.get("induced_at"),
        })
    return {"concepts": out}


@app.get("/api/lineage/{memory_id}")
def lineage(memory_id: str):
    return {"memory_id": memory_id, "neighbours": lineage_neighbours(memory_id)}


@app.post("/admin/forget")
def admin_forget(threshold: float = Query(0.05, ge=0.0, le=1.0)):
    n = heavy_consolidator.forget(threshold_utility=threshold, threshold_conf=0.3)
    return {"forgotten": n}


@app.post("/admin/consolidate")
def admin_consolidate():
    """Dev-only: run light consolidator immediately, then heavy."""
    light.run_once()
    result = heavy_consolidator.run_for_all_users()
    return {"light": "ran", "heavy": result}
