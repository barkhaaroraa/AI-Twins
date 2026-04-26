# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Twins is a FastAPI-based "AI Twin" system that learns user preferences, tracks tasks, and maintains long-term memory to personalize LLM responses. It uses Ollama for local LLM inference, MongoDB for persistent storage, and Qdrant for semantic vector search.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Required services (must be running):
# - MongoDB on localhost:27017
# - Qdrant on localhost:6333
# - Ollama with a model configured via .env

# Start the server
uvicorn app.main:app --reload
```

## Environment Variables

Configured via `.env` file (loaded by `app/config.py`):
- `OLLAMA_URL` - Ollama API endpoint
- `OLLAMA_MODEL` - Model name for Ollama
- `APP_NAME` - Application name

## Architecture

### Request Flow

1. **`app/main.py`** - FastAPI app with `/chat` (POST) and `/memory` (GET/DELETE) endpoints. The `/chat` endpoint orchestrates the full pipeline: decay old memories, extract/store new memories, fetch context, build prompt, generate response.

2. **Memory Pipeline** (on each `/chat` request):
   - `app/memory/summarizer.py` - Attempts LLM-based memory extraction first (asks the LLM to output structured JSON), falls back to rule-based keyword matching (e.g., "I prefer", "working on")
   - `app/memory/memory_updater.py` - Dual-writes extracted memories to both MongoDB (structured doc) and Qdrant (vector embedding)
   - `app/memory/vector.py` - Manages Qdrant vector store using `all-MiniLM-L6-v2` embeddings (384 dimensions). Handles `store_memory` and `search_memory`

3. **`app/db/mongo.py`** - MongoDB operations: user CRUD, preferences, tasks (with fuzzy dedup via `SequenceMatcher`), memory buffer, and memory importance decay

4. **`app/llm/ollama_client.py`** - Thin wrapper around Ollama's `/api/generate` endpoint (non-streaming)

5. **`app/static/` + `app/templates/`** - Simple chat UI served at `/`

### Data Stores

- **MongoDB (`ai_twin_db`)**: `users` collection (preferences, tasks, memory buffer) and `memories` collection (summarized memories with importance scores)
- **Qdrant (`user_memory`)**: Vector embeddings of memory summaries for semantic retrieval

### Key Design Patterns

- Memory retrieval uses cosine similarity with a 0.75 threshold and max 2 results (`filter_relevant_memory` in main.py)
- Memory importance decays by 0.01 per day since creation (`decay_memory_importance`)
- Structured prompt injection: user profile (preferences + tasks) and relevant history are injected into the LLM prompt
- Response includes `memory_used` for explainability

### Auth

The `/memory` endpoints require a Bearer token via `Authorization` header. The token is hardcoded in `app/main.py` as `API_TOKEN`. The `/chat` endpoint has no auth.

## Directories

- `app/` - Main application (FastAPI server, DB, memory pipeline, LLM client, UI)
- `twin/` - Contains `orchestrator.py` and `prompt_builder.py` (currently empty files)
