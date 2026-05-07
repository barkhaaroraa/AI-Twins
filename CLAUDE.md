# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Twins is a FastAPI-based **agentic memory** system. Multiple specialized agents (logger, nutritionist, trainer) share **one** memory pool keyed by `user_id` — anything one agent writes, the others can read on retrieval. Memory is summarized by an LLM, stored in MongoDB (structured) + Qdrant (vector), and pulled back via cosine similarity into the next agent's prompt.

There is no UI and no generic `/chat` endpoint — the only entry points are the agent endpoints.

## Running the Application

```bash
pip install -r requirements.txt

# Required services (must be running):
# - MongoDB on localhost:27017
# - Qdrant on localhost:6333
# - Ollama on localhost:11434 with the model from .env pulled

uvicorn app.main:app --reload
```

In this dev environment the three services run as Docker containers `ai_twin_mongo`, `ai_twin_qdrant`, `ai_twin_ollama` — start them with `docker start ai_twin_mongo ai_twin_qdrant ai_twin_ollama`.

## Environment Variables

`.env` (loaded by `app/config.py`):
- `OLLAMA_URL` — Ollama generate endpoint (e.g. `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL` — Ollama model name (e.g. `llama3.2:1b`)
- `APP_NAME`

## Endpoints

- `GET /agents` — list registered agents (name + first-line role)
- `POST /agent/{agent_name}` — body `{"user_id": "...", "message": "..."}`. Runs the full pipeline as that agent.
- `GET /api/agent-memories/{user_id}` — newest-first list of stored memories (no embeddings) for the shared pool.

No auth.

## Architecture

### Request flow (`POST /agent/{agent_name}`)

1. `app/main.py` looks the agent up in `AGENTS` and delegates to `TwinOrchestrator.process_agent_message`.
2. `twin/orchestrator.py`:
   - ensures the user exists in Mongo,
   - runs `search_memory` (Qdrant cosine similarity, top 4) for the message,
   - builds the prompt via `PromptBuilder` with the agent's `role_prompt` prepended,
   - calls Ollama (`generate_response`) — synchronous, blocks the response,
   - kicks off a daemon thread to summarize + store the user message (LLM extraction with rule-based fallback),
   - returns `{response, memory_used, agent}`.
3. Background thread (`_store_memory`):
   - `summarize_memory` (LLM JSON extraction → rule-based regex fallback). If both fail and the agent has `force_store=True`, falls back to `fast_store_payload` (no-LLM payload with heuristic entity extraction).
   - `update_memory` dual-writes to Mongo `memories` and Qdrant `user_memory`.

### Agents (`app/agents/__init__.py`)

`AgentSpec(name, role_prompt, force_store)`. Three are registered:
- `logger` — `force_store=True`, every user message becomes a memory.
- `nutritionist` — only LLM-judged "worth storing" messages persist.
- `trainer` — same.

Add an agent by appending an `AgentSpec` to the `AGENTS` dict.

### Data stores

- **MongoDB `ai_twin_db`**:
  - `users` — `{user_id, created_at}`. Minimal; no preferences/tasks.
  - `memories` — `{_id, user_id, type, intent, memory_type, summary, entities, relationships, confidence, importance, embedding, source_agent, created_at}`.
- **Qdrant `user_memory`** — 384-dim `all-MiniLM-L6-v2` embeddings, payload carries `user_id`, `text`, `memory_type`, `entities`, `confidence`. The `user_id` is filtered post-query (no Qdrant payload index).

### Memory extraction (`app/memory/summarizer.py`)

`summarize_memory(message)` tries `_llm_extract` (Ollama JSON) first; on parse failure or `store=false` it falls back to `_rule_based_extract` (regex against `task | preference | fact | correction | goal | contextual_reference`). Each intent maps to a `(memory_type, default_importance)` pair via `INTENT_DEFAULTS`. `fast_store_payload` is the LLM-free fast path used only by `force_store` agents when both extractors fail.

### Prompt construction (`twin/prompt_builder.py`)

`{agent_role}\n\nRelevant Memories:\n- [type, score=…] text…\n\nUser Message:\n"…"\n\nAI Response:`. Memories are grouped by `memory_type` to give the LLM a hint at hierarchy.

## Directories

- `app/main.py` — FastAPI app, 3 routes
- `app/agents/` — `AgentSpec` definitions; the only place to register a new agent
- `app/memory/` — `summarizer.py`, `memory_updater.py`, `vector.py`
- `app/db/mongo.py` — minimal users + memories CRUD
- `app/llm/ollama_client.py` — Ollama wrapper with `OllamaUnavailable` exception
- `twin/orchestrator.py` — pipeline glue
- `twin/prompt_builder.py` — prompt assembly
