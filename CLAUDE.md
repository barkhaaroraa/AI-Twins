# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Twins is a **Hierarchical Semantic Cognition (HSC)** memory system: a four-tier cognitive substrate where multiple agents reason on top of a shared, ACL-aware memory graph. Agents are partitioned by default (per-agent isolation) and share memories only via explicit policy, type-level rules (Preferences, Identity), or entity-bridged discovery.

Tiers:

- **L1** — per-`(user, agent)` in-process working-memory blackboard (activation map, working set, scratchpad, decay).
- **L2** — collaborative cognitive workspace: Mongo `memories_warm` + Qdrant `mem_warm` + `co_activation_log`.
- **L3** — semantic cortex: Neo4j graph + cold-tier docs/vectors. Concepts, Episodes, lineage edges live here.
- **Backing** — Mongo `raw_events`, append-only.

Retrieval is a **6-stage cascade**: seed (vector + entity + working-set, parallel) → working-set expansion → Personalized PageRank (PPR) over Neo4j → causal/temporal walks → ACL-aware synthesis → reflective update.

Background cognition: a **light consolidator** (idle ≥60 s) reinforces co-activation edges, decays old edges, recomputes utility; a **heavy consolidator** (cron 03:00 + idle ≥30 min fallback) detects clusters, induces `Concept` nodes via LLM abstraction, migrates to cold tier, and forgets low-utility memories.

See `architecture.md` for the full specification.

## Running the Application

```bash
pip install -r requirements.txt

# Required services (must be running):
# - MongoDB on localhost:27017
# - Qdrant on localhost:6333
# - Ollama on localhost:11434 with the model from .env pulled
# - Neo4j on localhost:7474 (browser) / 7687 (bolt)

uvicorn app.main:app --reload
```

In this dev environment the four services run as Docker containers:

```bash
docker start ai_twin_mongo ai_twin_qdrant ai_twin_ollama ai_twin_neo4j
```

Heavy consolidator can be invoked manually:

```bash
python -m app.cognition.consolidator.heavy --once
```

## Environment Variables

`.env` (loaded by `app/config.py`):

- `OLLAMA_URL` — Ollama generate endpoint (e.g. `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL` — Ollama model name (e.g. `qwen2.5:3b`). Used for all LLM paths: agent replies, intent extraction, contradiction detection, concept abstraction.
- `APP_NAME`
- `NEO4J_URI` — bolt URI (e.g. `bolt://localhost:7687`)
- `NEO4J_USER` — defaults to `neo4j`
- `NEO4J_PASSWORD` — dev password (`neo4j_dev`)

## Endpoints

- `GET /agents` — list registered agents (name + first-line role)
- `POST /agent/{agent_name}` — body `{"user_id": "...", "message": "..."}`. Runs the cascade as that agent.
- `GET /api/agent-memories/{user_id}` — newest-first list of stored memories (no embeddings).
- `GET /api/concepts/{user_id}` — induced Concept nodes for the user.
- `GET /api/lineage/{memory_id}` — 1-hop lineage subgraph as JSON.
- `POST /admin/forget?threshold=0.1` — dev-only forgetting trigger.

No auth.

## Architecture (one-liner)

`POST /agent/{name}` → orchestrator appends to `raw_events`, runs the 6-stage cascade (with ACL filter), builds a prompt, calls Ollama, returns the response, then async-writes the user message as a memory (extract → dedupe → triple-write Mongo+Qdrant+Neo4j) and feeds the cascade trace back into the agent's blackboard. See `architecture.md` for the full pipeline.

## Agents (`app/agents/__init__.py`)

`AgentSpec(name, role_prompt, force_store, default_visibility)`. Six are registered:

- Health family: `logger` (`force_store=True`), `nutritionist`, `trainer`.
- Productivity family: `project`, `school`, `research`.

Sharing: `app/agents/sharing_policy.py` holds `SHARING_POLICY` (auto-public types, bridge entities, family groupings) and the canonical `acl_check(memory, agent)` function used by every retrieval path.

Add an agent: append an `AgentSpec` to the `AGENTS` dict; optionally add to a family in `SHARING_POLICY.families`.

## Data stores

- **MongoDB `ai_twin_db`** — collections: `users`, `memories_warm`, `raw_events`, `co_activation_log`, `coherence_log`, `concept_board` (when populated by heavy consolidator).
- **Qdrant** — collection `mem_warm` (384-dim `all-MiniLM-L6-v2`); payload-indexed on `user_id`, `agent_owner`, `visibility`, `shared_with`, `memory_type`, `created_at`.
- **Neo4j `ai_twin`** — graph: `User`, `Agent`, `Memory`, `Entity`, `Concept`, `Episode` nodes; `OWNED_BY`, `AUTHORED_BY`, `MENTIONS`, `SHARED_WITH`, `CO_ACTIVATED`, `CAUSES`, `PRECEDES`, `SUPERSEDES`, `CONTRADICTS`, `ABSTRACTION_OF`, `REFINES`, `DERIVED_FROM`, `DEPENDS_ON` edges.

## Directories

```
app/
  main.py                FastAPI app, routes
  config.py              .env loader
  agents/                AgentSpec + sharing policy + acl_check
  cognition/             blackboard, activation, cascade, NER, utility, consolidator/
  graph/                 Neo4j client, schema, projections, queries
  memory/                summarizer, memory_updater, vector, backing, contradiction
  db/mongo.py            users + memories + raw_events + co_activation_log accessors
  llm/ollama_client.py   generate_response / generate_json + OllamaUnavailable
twin/
  orchestrator.py        request handler glue (cascade-driven)
  prompt_builder.py      tier-aware + concept-aware + contradiction-aware prompt assembly
tests/
  e2e/test_hsc.py        verification harness (8 e2e checks)
```
