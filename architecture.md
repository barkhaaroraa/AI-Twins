# Architecture

## What this is

An **agentic memory** server. Three specialized agents (`logger`, `nutritionist`, `trainer`) talk to the user and **share a single memory pool keyed by `user_id`**. Anything one agent writes, the next one reads. Storage is MongoDB (structured) + Qdrant (vector). Generation is local Ollama. There is no `/chat` route, no per-agent memory partition, no graph / consolidation / timeline machinery — the project is deliberately scoped to "shared agentic memory and nothing else."

## Components

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser  ─────────────────  app/templates/index.html  + static/*    │
│            POST /agent/{name}                                        │
│            GET  /api/agent-memories/{user_id}                        │
└────────────┬─────────────────────────────────────────────────────────┘
             │ HTTP (JSON)
┌────────────▼─────────────────────────────────────────────────────────┐
│  FastAPI  app/main.py                                                │
│  ── routes ────                                                      │
│   GET  /agents                       list_agents()                   │
│   POST /agent/{agent_name}           agent_chat()                    │
│   GET  /api/agent-memories/{user_id} agent_memories()                │
│                                                                      │
│  Singleton: TwinOrchestrator (twin/orchestrator.py)                  │
└────────────┬─────────────────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────────────────┐
│  Pipeline (sync path, blocks the response)                           │
│   1. create_user_if_not_exists       app/db/mongo.py                 │
│   2. search_memory  (top-4 cosine)   app/memory/vector.py → Qdrant   │
│   3. build_prompt                    twin/prompt_builder.py          │
│   4. generate_response               app/llm/ollama_client.py        │
│   5. spawn daemon thread → returns response immediately              │
│                                                                      │
│  Pipeline (background, daemon thread)                                │
│   6. summarize_memory   (LLM JSON → regex → fast_payload)            │
│   7. update_memory      writes to Mongo + Qdrant                     │
└──────────────────────────────────────────────────────────────────────┘
             │                                  │
   ┌─────────▼──────────┐             ┌─────────▼─────────────┐
   │  MongoDB           │             │  Qdrant               │
   │  ai_twin_db        │             │  user_memory          │
   │  ├ users           │             │  └ 384-dim cosine     │
   │  └ memories        │             │    payload: user_id,  │
   │                    │             │    text, memory_type, │
   │                    │             │    entities, conf.    │
   └────────────────────┘             └───────────────────────┘

   ┌────────────────────┐
   │  Ollama (Docker)   │  ← called twice per turn:
   │  llama3.2:1b       │     once for the agent reply (sync)
   │  port 11434        │     once for memory extraction (bg)
   └────────────────────┘
```

## Logical data flow — `POST /agent/{agent_name}`

```
client message
  │
  ▼
main.agent_chat
  │ resolve AgentSpec from app/agents/__init__.py
  ▼
TwinOrchestrator.process_agent_message
  │
  ▼
TwinOrchestrator._process    ──────────── SYNCHRONOUS ────────────
  │
  ├─ create_user_if_not_exists(user_id)         [Mongo: users]
  │
  ├─ retrieved = search_memory(user_id, msg, 4)
  │     │
  │     ├─ embed_text(msg)                      [SentenceTransformer]
  │     ├─ qdrant.query_points(top=12)          [Qdrant: user_memory]
  │     └─ post-filter by user_id, take top 4
  │
  ├─ build_prompt(user_profile, retrieved, msg, agent.role_prompt)
  │     │
  │     └─ {role_prompt}\n\nRelevant Memories:\n- [type, score=…] text…
  │        \n\nUser Message: "…"\n\nAI Response:
  │
  ├─ generate_response(prompt)                  [Ollama /api/generate]
  │     OllamaUnavailable → degraded "I can't reach the LLM" response
  │
  ├─ spawn daemon thread → _store_memory(...)
  │
  └─ return {response, memory_used, agent}      ◄── client unblocked

_store_memory                ──────────── BACKGROUND ─────────────
  │
  ├─ summarize_memory(message)
  │     │
  │     ├─ _llm_extract  →  Ollama JSON  → INTENT_DEFAULTS lookup
  │     │      on parse failure or store=false  ↓
  │     └─ _rule_based_extract  → regex over INTENT_PATTERNS
  │                                       (task | preference | fact |
  │                                        correction | goal | contextual_reference)
  │
  ├─ if no extraction AND agent.force_store:
  │     summarized = fast_store_payload(message)   ← never drops a logger msg
  │
  └─ update_memory(user_id, summarized, source_agent)
        │
        ├─ embed_text(summary)
        ├─ memory_collection.insert_one({...})    [Mongo: memories]
        └─ qdrant.upsert(point)                   [Qdrant: user_memory]
```

### Why two LLM calls per turn

1. **Reply call** is on the critical path. The user blocks on it.
2. **Extraction call** runs after the response is sent. Latency and Ollama outages are absorbed there — the user never waits, and a failed extraction silently falls back to regex (and to `fast_store_payload` for `force_store=True` agents).

### Shared memory key

There is **no per-agent partitioning**. Every memory document has a `source_agent` field for display/debug, but `search_memory` filters only on `user_id`. That's the central design choice: log a workout via `logger`, then ask `nutritionist` what to eat, and the workout fact is in scope.

### Agent specifications (`app/agents/__init__.py`)

`AgentSpec(name, role_prompt, force_store)`:

| name | force_store | meaning |
|------|-------------|---------|
| `logger` | True | every user message persists, even if the LLM/regex extractors return nothing |
| `nutritionist` | False | only LLM-judged "worth storing" messages persist |
| `trainer` | False | same as nutritionist |

Add an agent: append a new `AgentSpec` to the `AGENTS` dict. No other wiring needed — `main.py` looks agents up by name and `TwinOrchestrator` is agent-agnostic.

## Memory document schema

`memories` collection (Mongo):

```jsonc
{
  "_id":           "<uuid4>",
  "user_id":       "demo_health",
  "type":          "task",                    // alias of intent, kept for compat
  "intent":        "task",                    // task|preference|fact|correction|goal|contextual_reference
  "memory_type":   "Procedural",              // Semantic|Episodic|Procedural|Preference
  "summary":       "Completed a 30-minute morning run",
  "entities":      ["run", "morning"],
  "relationships": [],                        // optional {subject, predicate, object}
  "confidence":    0.8,
  "importance":    0.7,
  "embedding":     [<384 floats>],
  "source_agent":  "logger",                  // null if not from an agent
  "created_at":    "<utc datetime>"
}
```

Qdrant payload mirrors a subset (`user_id`, `text`, `memory_type`, `entities`, `confidence`) — the embedding is the vector, not in the payload.

## Intent classification

`summarizer.py` tries the LLM first, then a regex fallback. Each intent maps to a `(memory_type, default_importance)` pair via `INTENT_DEFAULTS`:

| intent | memory_type | importance |
|--------|-------------|------------|
| task | Procedural | 0.9 |
| preference | Preference | 0.8 |
| fact | Semantic | 0.7 |
| correction | Semantic | 0.95 |
| goal | Procedural | 0.85 |
| contextual_reference | Episodic | 0.6 |

Importance is currently descriptive (stored on the doc, surfaced in the UI). It is **not** decayed or used for retrieval ranking — retrieval is pure cosine similarity.

## Failure modes

- **Ollama down** → reply is a static degradation message, memory still gets stored via the regex fallback in the background thread.
- **Mongo down** → request fails with a 500 (no graceful degradation, ensure container is up).
- **Qdrant down** → `init_vector_collection()` fails on startup; mid-flight failures bubble through `search_memory` and `update_memory`.
- **Extraction returns nothing** → message is dropped unless the agent has `force_store=True`.

## File map

```
app/
  main.py                     FastAPI app, 3 routes, mounts /static + /
  agents/__init__.py          AgentSpec + AGENTS registry
  config.py                   .env loader (OLLAMA_URL/MODEL, APP_NAME)
  db/mongo.py                 users + memories CRUD (minimal)
  llm/ollama_client.py        generate_response / generate_json + OllamaUnavailable
  memory/
    summarizer.py             LLM JSON → regex → INTENT_DEFAULTS; fast_store_payload
    memory_updater.py         embed → insert(Mongo) → upsert(Qdrant)
    vector.py                 SentenceTransformer + Qdrant client + search_memory
  static/{app.js,style.css}   agent-grid UI + shared-memory feed
  templates/index.html        single-page agents view

twin/
  orchestrator.py             pipeline glue (sync reply + bg memory write)
  prompt_builder.py           role + retrieved memories + user message → prompt

requirements.txt              direct deps only (no networkx, no scikit-learn)
.env                          OLLAMA_URL, OLLAMA_MODEL, APP_NAME
CLAUDE.md                     project notes for Claude Code
architecture.md               this file
```

## Run

```bash
docker start ai_twin_mongo ai_twin_qdrant ai_twin_ollama
source venv/bin/activate
uvicorn app.main:app --reload
# UI:  http://localhost:8000/
# API: GET /agents · POST /agent/{name} · GET /api/agent-memories/{user_id}
```
