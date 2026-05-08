# Hierarchical Semantic Cognition — Architecture

> A semantic operating system for collaborative cognition, not a chatbot memory store.

---

## Context

The current AI-Twins system was a flat memory pool: every agent (logger, nutritionist, trainer) read from and wrote to a single Mongo + Qdrant store keyed only by `user_id`. Retrieval was pure cosine top-k. Memories carried a `source_agent` field but it was never used for filtering. Relationships, importance, and confidence were stored but inert. The system performed adequately as a retrieval engine but could not support the cognitive properties the next generation of agents needs: per-agent isolation, selective sharing, evolving abstractions, attention-driven retrieval, causal reasoning, and offline consolidation.

This document specifies the redesign that treats memory as a living adaptive system rather than static storage. It draws from CPU cache hierarchy (locality, eviction, coherence), operating systems (scheduling, ACL, paging), hippocampus/cortex models (replay, consolidation, abstraction), and cognitive architectures (spreading activation, blackboards, working memory). The output is a four-tier cognitive substrate that supports two existing agent families (logger / nutritionist / trainer — health) and three new ones (project / school / research — productivity), with isolation by default and explicit, policy-driven sharing.

Locked decisions (informing this design):
1. Both agent families coexist on the new substrate.
2. Neo4j as the L3 graph engine, deployed as a Docker service.
3. Activation state lives in-process only, rebuilt on restart from L2/L3 snapshots + recent backing-store events.
4. Consolidation runs hybrid: light idle-triggered (every ~60 s of inactivity) + heavy scheduled (03:00 daily, with fallback on >30 min idle).
5. Wipe-and-start-fresh for existing memories.

---

## A. Problem Framing

A flat shared pool fails as soon as the system has more than one cognitive concern at once:

1. **No isolation.** Anything written by `logger` can leak into `research` retrieval. A workout note can pollute a thesis-related query if its embedding happens to be close.
2. **No selective sharing.** Cross-agent context is all-or-nothing.
3. **Vector cosine alone is brittle.** It cannot answer "what depends on this?", "what did I conclude after X?", "what contradicts Y?", or "what abstractions have emerged from these episodes?".
4. **Memories are anonymous chunks.** No lineage, evidence, contradictions, or version history.
5. **Storage-shaped, not cognition-shaped.** No priming, spreading activation, working set, offline consolidation, or abstraction induction. The system never "thinks" between requests.

The hierarchy below is *both* a storage hierarchy (latency, capacity, eviction) *and* a cognitive hierarchy (each tier has a distinct role).

| Tier    | Storage view              | Cognitive view                                    |
|---------|---------------------------|---------------------------------------------------|
| L1      | per-agent hot cache       | working-memory blackboard                         |
| L2      | shared warm cache         | collaborative cognitive workspace                 |
| L3      | persistent semantic store | semantic cortex (concepts, episodes, lineage)     |
| Backing | append-only event log     | hippocampal raw record / autobiographical trace   |

---

## B. Design Principles

Drawn from research into spreading activation (Collins & Loftus), hippocampus/cortex consolidation (McClelland et al.), working memory (Baddeley), blackboard architectures (HEARSAY-II), Personalized PageRank (Haveliwala), HippoRAG, GraphRAG, MemGPT, generative-agent reflection (Park et al.), and CPU cache coherence (MESI):

- Structured memory objects over raw text blobs.
- Hybrid retrieval (vector + graph + temporal + structured) over pure embedding search.
- Explicit relations over implicit similarity alone.
- Tiered storage with cognitive roles over a single flat pool.
- Policy-aware sharing over unrestricted global access.
- Consolidation, abstraction, and forgetting over unbounded accumulation.
- Provenance, versioning, and lineage on every memory object.
- Activation-driven retrieval over pure query-driven retrieval.

---

## C. Memory Hierarchy

```
╔════════════════════════════════════════════════╗
║  L1 — Working-Memory Blackboard (per agent)    ║
║  In-process Python. ~50 items per agent.       ║
║  activation map, working set, goals,           ║
║  contradictions, hypotheses, scratchpad.       ║
║  Latency: <100 µs. Volatile.                   ║
╚────────────────────┬───────────────────────────╝
                     │ miss / cold
╔════════════════════▼═══════════════════════════╗
║  L2 — Collaborative Cognitive Workspace        ║
║  Qdrant warm + Mongo (last ~30 d) + tagged     ║
║  shared concepts, transient summaries,         ║
║  active cross-agent goals, attention pool.     ║
║  Latency: <50 ms. Coherent across agents.      ║
╚════════════════════┬═══════════════════════════╝
                     │ insufficient → expand
╔════════════════════▼═══════════════════════════╗
║  L3 — Semantic Cortex                          ║
║  Neo4j (graph) + cold doc/vector entries       ║
║  concepts, episodes, lineage, causal chains,   ║
║  temporal narratives, induced abstractions.    ║
║  Latency: <300 ms. Source of truth.            ║
╚════════════════════┬═══════════════════════════╝
                     │ replay / audit only
╔════════════════════▼═══════════════════════════╗
║  Backing Store — Immutable Event Log           ║
║  Mongo `raw_events` (append-only).             ║
║  raw user messages, extraction outputs,        ║
║  agent responses, full provenance.             ║
║  Never deleted.                                ║
╚════════════════════════════════════════════════╝
```

### L1 — Working-Memory Blackboard (per agent)

Per `(user_id, agent_name)`. Holds:
- `activation_map: Dict[memory_id, float]`
- `working_set: OrderedDict[memory_id, Memory]` (cap ~50)
- `attention_weights: Dict[entity_name, float]`
- `active_goals`, `open_hypotheses`, `contradictions`, `scratchpad`
- `last_cascade: RetrievalTrace`

Decay: exponential, half-life ~7 minutes (λ = 1/600). 1 Hz tick. Items below threshold drop out.

### L2 — Collaborative Cognitive Workspace

Per `user_id`. Shared across all agents for that user, with ACL applied at read time.
- Qdrant warm collection `mem_warm` (embeddings, payload-filtered).
- Mongo `memories_warm` (full docs).
- Mongo `co_activation_log` (pairs of memory_ids touched together — light consolidator drains it).

Eviction to L3 cold on age > 30 d AND low activation; promotion back on rehit.

### L3 — Semantic Cortex

Per `user_id`. Source of truth.
- Neo4j graph: `User`, `Agent`, `Memory`, `Entity`, `Episode`, `Concept`, `Decision`, `Goal`, `Task`, `Topic` nodes; weighted and unweighted edges (see §G).
- Cold doc/vector entries (in this implementation, same collection with `tier=cold` flag, queried on demand).

Mostly written by the consolidators; new `Memory` nodes also projected at write time.

### Backing — Immutable Event Log

Mongo `raw_events`, append-only. Holds `event_id` (ULID), `user_id`, `agent_name`, `timestamp`, `event_type` ∈ {`user_message`, `agent_response`, `extraction_output`, `consolidation_event`}, `payload`, `produced_memory_ids`. Never deleted.

---

## D. Retrieval — 6-Stage Activation Cascade

Retrieval is no longer "embed query, top-k, return." It is a six-stage cascade:

```
Stage 1: Seed activation         (cheap, parallel)
Stage 2: Working-set expansion   (L1 only, instant)
Stage 3: Spreading activation    (L3 graph, PPR)
Stage 4: Causal/temporal walk    (L3 graph, directed)
Stage 5: Filter + synthesis      (ACL, dedupe, rank)
Stage 6: Reflective update       (post-response)
```

### Stage 1 — Seed Activation (parallel, ~20 ms)

```python
async def seed(query, agent, user_id):
    return await asyncio.gather(
        vector_seed(query, user_id, agent, k=5),     # Qdrant warm, ACL via payload filter
        entity_seed(query, user_id, agent),           # NER → Entity nodes → MENTIONS
        working_set_seed(agent, user_id),             # L1 lookup
    )
```
Three `Dict[memory_id, strength]` maps merged by max.

### Stage 2 — Working-Set Expansion

```python
def expand_working_set(seeds, blackboard):
    expanded = dict(seeds)
    for mid, s in seeds.items():
        for nb in blackboard.recently_co_activated(mid):
            expanded[nb] = max(expanded.get(nb, 0), 0.5 * s)
    return expanded
```
Items already on the agent's mind get a head start (priming).

### Stage 3 — Spreading Activation (Personalized PageRank)

Iterative weighted PPR over the user's Neo4j subgraph, restricted to ACL-visible nodes via Cypher `MATCH` filter. α = 0.15 restart probability, 8 iterations, ε = 1e-3 convergence. Edge weights:

| Edge          | Weight                                                  |
|---------------|---------------------------------------------------------|
| CO_ACTIVATED  | learned (light consolidator), capped at 1.0             |
| DEPENDS_ON    | 0.9 (static)                                            |
| MENTIONS      | 0.5 (static)                                            |
| CONTRADICTS   | -0.7 (suppression)                                      |
| SUPERSEDES    | 0.0 (dampens)                                           |

### Stage 4 — Causal/Temporal Traversal

For top-N activated memories, walk:
- backward along `CAUSES`, `DERIVED_FROM` (provenance / why),
- forward along `PRECEDES` (what next),
- outward along `REFINES`, `SUPERSEDES` (latest version).

### Stage 5 — Filter + Synthesis

ACL filter (default-deny on missing visibility), drop superseded, drop contradicted unless query asks about conflicts, lineage dedupe (prefer Concept over members for general queries), rank by `activation * confidence * recency_decay`.

### Stage 6 — Reflective Update (post-response, async)

Bump activation +0.3 for each used memory; log every used pair to `co_activation_log` for the light consolidator.

### Latency budget

| Stage              | Target | Worst case |
|--------------------|--------|------------|
| 1 Seed             | 20 ms  | 80 ms      |
| 2 Working set      | 1 ms   | 5 ms       |
| 3 PPR              | 80 ms  | 200 ms     |
| 4 Causal           | 50 ms  | 150 ms     |
| 5 Synthesis        | 10 ms  | 30 ms      |
| 6 Reflect (async)  | 0      | —          |
| **Total**          | ~160 ms| ~465 ms    |

---

## E. Write Pipeline

```python
async def write_memory(user_msg, user_id, agent):
    event_id = await raw_events.insert({...})       # backing, sync, fast
    asyncio.create_task(_extract_and_persist(event_id, user_msg, user_id, agent))
    return event_id

async def _extract_and_persist(event_id, msg, user_id, agent):
    extraction = await extract(msg, agent)
    if not extraction and not agent.force_store: return
    if not extraction: extraction = fast_store_payload(msg)
    candidate = build_memory(extraction, user_id, agent, event_id)
    duplicate = await find_duplicate(candidate, user_id, sim=0.92)
    if duplicate:
        await reinforce(duplicate, candidate)       # bump confidence + evidence
        return
    candidate.id = ulid()
    candidate.lineage = Lineage(source_event_id=event_id, evidence=[event_id], version=1)
    async with memory_lock(candidate.id):
        await asyncio.gather(
            mongo.memories_warm.insert(candidate.to_doc()),
            qdrant.upsert("mem_warm", candidate.to_point()),
            neo4j_project(candidate),
        )
    blackboard_for(user_id, agent).activate(candidate.id, +1.0)
```

### Memory schema (Mongo `memories_warm`)

```jsonc
{
  "_id":              "01HMZK...",          // ULID
  "user_id":          "user_abc",
  "version":          1,
  "tier":             "warm",
  "intent":           "task",
  "memory_type":      "Procedural",
  "summary":          "...",
  "raw_text_event_id":"01HMZJ...",
  "entities":         [{"name": "thesis", "kind": "topic"}, ...],
  "lineage": {
    "source_event_id":"01HMZJ...",
    "derived_from":   [],
    "supersedes":     [],
    "abstraction_of": [],
    "evidence":       ["01HMZJ..."],
    "contradicted_by":[],
    "version":        1
  },
  "confidence":       0.8,
  "importance":       0.7,
  "novelty":          0.6,
  "utility":          null,
  "last_activation":  0.0,
  "last_accessed_at": "2026-05-07T12:00:00Z",
  "access_count":     0,
  "agent_owner":      "project",
  "visibility":       "private",            // private | shared | public
  "shared_with":      [],
  "created_at":       "...",
  "updated_at":       "..."
}
```

### Qdrant payload

`user_id`, `agent_owner`, `visibility`, `shared_with`, `memory_type`, `intent`, `created_at`. Native Qdrant payload indexes — no Python post-filter.

### Neo4j projection (per memory)

```cypher
MERGE (u:User {id: $user_id})
MERGE (a:Agent {name: $agent_owner})
CREATE (m:Memory {
  id: $id, type: $memory_type, intent: $intent,
  importance: $importance, confidence: $confidence,
  visibility: $visibility, created_at: $created_at
})
MERGE (m)-[:OWNED_BY]->(u)
MERGE (m)-[:AUTHORED_BY]->(a)
WITH m
UNWIND $entities AS ent
  MERGE (e:Entity {name: ent.name, kind: ent.kind, user_id: $user_id})
  MERGE (m)-[:MENTIONS]->(e)
WITH m
UNWIND $shared_with AS sw
  MERGE (sa:Agent {name: sw})
  MERGE (m)-[:SHARED_WITH]->(sa)
```

`CO_ACTIVATED`, `CAUSES`, `SUPERSEDES`, `CONTRADICTS`, `ABSTRACTION_OF`, `PRECEDES`, `REFINES` are written by the consolidators, not at write time.

### Promotion / demotion

| Transition       | Trigger                                                     | Action                                                                |
|------------------|-------------------------------------------------------------|-----------------------------------------------------------------------|
| L2 → L1          | retrieval cascade selects it                                | load doc into agent blackboard; bump activation                       |
| L1 refresh       | re-used in subsequent turn                                  | bump activation; reset decay timer                                    |
| L1 drop          | activation < ε                                              | drop from working set; snapshot to L2                                 |
| L2 → L3 (cold)   | utility < 0.4 AND age > 30 d AND no activation 7 d          | tier="cold"; vector demoted; Neo4j unchanged                          |
| L3 → L2          | retrieval pulls cold memory N times in 24 h                 | tier="warm" again                                                     |
| → Concept        | heavy consolidator finds ≥3 co-activated similar memories   | create Concept node + ABSTRACTION_OF edges                            |
| → Forgotten      | utility < 0.05 AND confidence < 0.3 AND age > 90 d          | mark forgotten=true; remove from Qdrant + warm Mongo; keep Neo4j node |

---

## F. Cache Replacement & Eviction (Semantic Economics)

### Utility scoring

```python
def utility(m):
    return (
        0.30 * recency_score(m)
      + 0.25 * frequency_score(m)
      + 0.20 * importance(m)
      + 0.15 * centrality(m)        # PageRank on CO_ACTIVATED subgraph
      + 0.10 * lineage_value(m)     # 1 if other memories derive from it
      - 0.20 * staleness_penalty(m)
      - 0.30 * conflict_penalty(m)
    )
```
Recomputed by the light consolidator hourly.

### Decay

L1 activation: λ = 1/600 (~7 min half-life), 1 Hz tick. Drop below 1e-3.

CO_ACTIVATED edge decay: 0.95 per hour of no reinforcement, delete < 0.05.

### Abstraction induction (heavy consolidator)

```python
def induce_abstractions(user_id):
    clusters = neo4j_high_co_activation_clusters(user_id, threshold=0.7)
    for cluster in clusters:
        if avg_pairwise_similarity(cluster) < 0.75: continue
        summary = await llm.abstract(cluster.members)
        concept_id = await neo4j.create_concept(summary, source_ids=...)
```

---

## G. Relationship Modeling

### Node types

| Label    | Purpose                          | Key fields                                                   |
|----------|----------------------------------|--------------------------------------------------------------|
| User     | partition root                   | id                                                           |
| Agent    | author / sharing target          | name                                                         |
| Memory   | atomic memory object             | id, type, intent, confidence, importance, visibility, created_at |
| Entity   | named thing referenced           | name, kind, user_id                                          |
| Episode  | bounded experience               | id, time_window, narrative                                   |
| Concept  | induced abstraction              | id, label, summary, induced_at, confidence                   |
| Decision | recorded choice                  | id, made_at, rationale                                       |
| Goal     | active objective                 | id, status, due_at                                           |
| Task     | actionable item                  | id, status, due_at                                           |
| Topic    | broad theme                      | name                                                         |

### Edge types

| Type           | Direction        | Weight       | Meaning                                |
|----------------|------------------|--------------|----------------------------------------|
| OWNED_BY       | Memory → User    | —            | partition                              |
| AUTHORED_BY    | Memory → Agent   | —            | who wrote it                           |
| SHARED_WITH    | Memory → Agent   | —            | ACL grant                              |
| MENTIONS       | Memory → Entity  | static 0.5   | named-entity reference                 |
| PART_OF        | Memory → Episode | —            | episodic grouping                      |
| DEPENDS_ON     | Memory → Memory  | static 0.9   | logical dependency                     |
| DERIVED_FROM   | Memory → Memory  | —            | provenance                             |
| SUPERSEDES     | Memory → Memory  | —            | replaces                               |
| CONTRADICTS    | Memory ↔ Memory  | -0.7         | conflict marker                        |
| REFINES        | Memory → Memory  | —            | non-replacing improvement              |
| ABSTRACTION_OF | Concept → Memory | —            | concept generalizes memory             |
| CAUSES         | Memory → Memory  | static       | causal claim                           |
| PRECEDES       | Memory → Memory  | static       | temporal order                         |
| CO_ACTIVATED   | Memory ↔ Memory  | learned      | reinforced by retrieval co-occurrence  |

`CO_ACTIVATED` is the only edge type the system learns automatically.

---

## H. Consistency & Access Control

### ACL

Every memory has `agent_owner`, `visibility ∈ {private, shared, public}`, `shared_with: List[agent_name]`.

```python
def acl_check(memory, agent):
    v = memory.get("visibility", "private")  # default-deny
    if v == "public":  return True
    if v == "shared":  return agent.name in memory.get("shared_with", [])
    if v == "private": return memory.get("agent_owner") == agent.name
    return False
```

### Sharing protocols

1. **Explicit share** — `share_memory(memory_id, target_agents, reason)`.
2. **Type-level policy** — Preference and Identity memories auto-public; Goal memories with `cross_agent: true` auto-public; Procedural/Episodic/most Semantic stay private.
3. **Entity-bridged** — entities flagged in `bridge_entities` (calendar, schedule, deadline, energy_level, sleep) make summary discoverable across families with owner notification.

### Coherence

For cross-agent reads of shared memories, a simple invalidation protocol: when an agent refines a memory, all blackboards that have it loaded drop their copy and re-fetch on next access. (MESI-style ledger is encoded in the `coherence_log` collection but is mostly cosmetic at single-process scale.)

### Conflict resolution

When a new memory contradicts an existing one:
1. Both kept. New one gets `CONTRADICTS` edge to old.
2. Confidence drift: new side ×1.05, contradicted side ×0.95 (asymptotic toward 1.0/0.0).
3. Synthesis stage surfaces both with timestamps.
4. After 3+ retrievals where the new side wins, light consolidator introduces `SUPERSEDES`.

---

## I. Concurrency & Performance

```
                ┌──────────────────────────────┐
                │  FastAPI (asyncio loop)      │
                │   - request handlers         │
                │   - L1 blackboards           │
                │   - retrieval cascade        │
                └──────┬───────────────────────┘
                       │ submit
        ┌──────────────┼─────────────────┐
        │              │                 │
        ▼              ▼                 ▼
   ┌────────┐  ┌──────────────┐  ┌──────────────────┐
   │Extract │  │Light         │  │Heavy             │
   │tail    │  │consolidator  │  │consolidator      │
   │(per req)│ │(idle ≥60s)   │  │(cron 03:00 +     │
   │        │  │              │  │ idle ≥30 min)    │
   └────────┘  └──────────────┘  └──────────────────┘
```

- Per-memory `asyncio.Lock` keyed by `memory_id` during the triple-write.
- Single-writer per blackboard (one event loop, one process).
- Heavy consolidator may run in the same process (APScheduler `coalesce=True, max_instances=1`) or be invoked manually with `python -m app.cognition.consolidator.heavy --once`.

---

## J. Use Case Customization

### Agents (`app/agents/__init__.py`)

```python
AGENTS = {
    # health
    "logger":       AgentSpec("logger",       "...", force_store=True,  default_visibility="private"),
    "nutritionist": AgentSpec("nutritionist", "...", force_store=False, default_visibility="private"),
    "trainer":      AgentSpec("trainer",      "...", force_store=False, default_visibility="private"),
    # productivity
    "project":      AgentSpec("project",      "...", force_store=False, default_visibility="private"),
    "school":       AgentSpec("school",       "...", force_store=False, default_visibility="private"),
    "research":     AgentSpec("research",     "...", force_store=False, default_visibility="private"),
}
```

### Sharing policy (`app/agents/sharing_policy.py`)

```python
SHARING_POLICY = {
    "auto_public_types":  {"Preference", "Identity"},
    "bridge_entities":    {"calendar", "schedule", "deadline", "energy_level", "sleep"},
    "families": {
        "health":       {"logger", "nutritionist", "trainer"},
        "productivity": {"project", "school", "research"},
    },
}
```

---

## K. Stack

| Concern         | Tech                                                      |
|-----------------|-----------------------------------------------------------|
| Web framework   | FastAPI (async)                                           |
| Doc store       | MongoDB                                                   |
| Vector index    | Qdrant                                                    |
| Graph store     | Neo4j 5.x + APOC                                          |
| LLM             | Ollama (local) — `qwen2.5:3b` (single model, all paths)   |
| Embeddings      | `BAAI/bge-small-en-v1.5` (384 dim)                        |
| Scheduler       | APScheduler                                               |
| Activation      | In-process Python (numpy + dict)                          |
| Backing store   | Mongo `raw_events`                                        |

### Package layout

```
app/
  cognition/
    blackboard.py             per-agent L1 working memory
    activation.py             decay + spreading-activation kernel
    cascade.py                6-stage retrieval cascade
    ner.py                    rule-based entity extraction
    utility.py                utility scoring
    consolidator/
      light.py                idle-triggered worker
      heavy.py                scheduled worker
      abstraction.py          LLM-driven concept induction
  graph/
    neo4j_client.py
    schema.py
    projections.py            memory→graph projection
    queries.py                parametrized Cypher
  memory/
    summarizer.py             evolved (lineage scaffolding, kind on entities)
    memory_updater.py         evolved (triple-write + ACL + dedupe)
    vector.py                 evolved (Qdrant payload indexes, mem_warm)
    backing.py                raw_events accessors
    contradiction.py          contradiction detection
  agents/
    __init__.py               AgentSpec + AGENTS (extended)
    sharing_policy.py         SHARING_POLICY + acl_check
twin/
  orchestrator.py             evolved (cascade-driven)
  prompt_builder.py           evolved (tier + concept preference + contradictions)
tests/
  e2e/test_hsc.py             verification harness
```

---

## L. Failure Modes & Safeguards

| Failure                              | Symptom                          | Safeguard                                                                                  |
|--------------------------------------|----------------------------------|--------------------------------------------------------------------------------------------|
| Neo4j down at request time           | Stage 3/4 fail                   | Cascade degrades to L1+L2 only. Response carries `degraded_retrieval: true`.               |
| Qdrant down                          | Stage 1 vector seed empty        | Fall back to entity + working-set seeds.                                                   |
| Mongo down                           | Doc fetches fail                 | 503. Mongo is the truth store.                                                             |
| Ollama down                          | Abstraction induction fails      | Heavy consolidator skips abstraction; queues for next cycle. Light unaffected.             |
| Activation lost on crash             | Cold start                       | Lazy rebuild: blackboards re-seed on first request after restart.                          |
| Memory bloat                         | L3 grows unbounded               | Forgetting policy in §F. Manual `forget_below_utility()` admin endpoint.                   |
| Stale abstractions                   | Wrong retrievals                 | Heavy consolidator re-evaluates Concept→source coherence; flags for refresh.               |
| Conflict drift                       | Mixed signals                    | Reflective synthesis surfaces both with timestamps; `SUPERSEDES` after consistent winner.  |
| Over-sharing                         | ACL bug                          | Single chokepoint at Stage 5; default-deny on missing visibility; integration test.        |
| Retrieval pollution                  | Bad answers                      | PPR ACL-restricted at Cypher level — non-visible nodes never traversed.                    |
| Hot-loop activation positive feedback| Same memory always wins          | Decay + edge cap (≤1.0) + diversity-aware reranker in Stage 5.                             |
| Heavy consolidator during user activity | Latency spike                | Idle detector + APScheduler `coalesce=True, max_instances=1`; yields if request flag set.  |
| Backing store unbounded growth       | Disk fill                        | Mongo TTL after 1 yr inactivity; Concepts and lineage preserve what matters.               |

---

## M. Final Blueprint

```
        ┌────────────────────────────────────────────────┐
        │                  USER MESSAGE                  │
        └───────────────────────┬────────────────────────┘
                                ▼
       ┌────────────────────────────────────────────────────┐
       │  FastAPI request handler (asyncio)                 │
       │  • append to raw_events  (backing store)           │
       │  • spawn extract+persist task  (async tail)        │
       │  • run RETRIEVAL CASCADE                           │
       └───────────────────────┬────────────────────────────┘
                               ▼
   ┌──── 6-STAGE RETRIEVAL CASCADE (ACL-aware throughout) ───────┐
   │ 1 Seed (parallel):  vector  +  entity NER  +  working set   │
   │ 2 Working-set expand (L1 priming)                           │
   │ 3 Spreading activation: PPR over Neo4j with edge weights    │
   │ 4 Causal/temporal traversal: CAUSES, PRECEDES, SUPERSEDES   │
   │ 5 Filter + synthesis: ACL, lineage dedupe, rank             │
   │ 6 Reflect (post-response): bump activation, log pairs       │
   └───────────────────────┬─────────────────────────────────────┘
                           ▼
       ┌────────────────────────────────────────────────────┐
       │  Prompt builder · Ollama · response → user         │
       └───────────────────────┬────────────────────────────┘
                               ▼
       ┌────────────────────────────────────────────────────┐
       │  Async write tail:                                 │
       │   extract → dedupe → triple-write (Mongo+Qdrant+   │
       │   Neo4j); bump writer's L1 activation              │
       │   queue light-consolidator tasks                   │
       └────────────────────────────────────────────────────┘
```

```
       ┌─────────────────────────────────┐
       │   BACKGROUND COGNITION CYCLES   │
       │                                 │
       │  Light  (asyncio, idle ≥60 s):  │
       │   • edge re-weighting           │
       │   • activation decay            │
       │   • dedupe sweep                │
       │   • utility recompute           │
       │                                 │
       │  Heavy  (cron 03:00 + ≥30 min): │
       │   • cluster detection           │
       │   • abstraction induction (LLM) │
       │   • episodic → semantic         │
       │   • L2 ↔ L3 migration           │
       │   • concept hierarchy           │
       └─────────────────────────────────┘
```

Memory at a glance:
```
L1  per-agent working memory blackboard   (in-process, volatile, ~50 items)
        ↕ on retrieval / write
L2  collaborative cognitive workspace     (Mongo warm + Qdrant warm, 30d window)
        ↕ on age / activation / consolidation
L3  semantic cortex                       (Neo4j graph + cold doc/vector)
        ↕ replay / audit only
backing  immutable event log               (Mongo raw_events, append-only)
```

Invariants:
- Every memory has an `agent_owner` and explicit `visibility`. Default is private.
- Every memory has a `lineage` (source event, evidence, derived-from, supersedes, contradicted-by).
- Every retrieval is a cascade, not a single query.
- Every co-activation is learned; edges in the graph reflect what actually got used together.
- Every consolidation is traceable; Concepts know their source memories.

---

## Verification — End-to-End Checks

1. **Isolation.** Memory written via `project` (default private) is not retrieved through `school`.
2. **Selective sharing.** `visibility=shared, shared_with=[research]` → visible to `research`, not to `school`.
3. **Spreading activation.** Two memories sharing an entity but with low cosine similarity — query matching one surfaces the other via the graph.
4. **Abstraction induction.** ≥4 co-activated memories on a topic → after heavy consolidator, a `Concept` node exists with `ABSTRACTION_OF` edges. General queries surface the Concept.
5. **Forgetting.** `utility<0.05 AND confidence<0.3 AND age>90d` → marked forgotten, removed from Qdrant + warm Mongo; Neo4j node remains.
6. **Coherence.** Two agents load the same shared memory; one refines; other's next access fetches fresh.
7. **Provenance.** Every L3 Memory traces back to ≥1 `raw_events` entry. No orphans.
8. **Conflict survival.** Two contradictory memories both stored; `CONTRADICTS` edge exists; retrieval surfaces both with timestamps.

(Items 6 and 8 from the original spec — warm restart and latency benchmark — are implementation-quality concerns rather than functional invariants and are deferred from the canonical check list.)
