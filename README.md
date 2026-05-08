     Hierarchical Semantic Cognition — Architecture Proposal                     
                                                                                                                                                                                         
     ▎ A semantic operating system for collaborative cognition, not a chatbot memory store.                          
                                                                                                                                                                                         
     ---                                                                                                       
     Context                                                                                                                                                                             
                                                                                                                                                                                         
     The current AI-Twins system is a flat memory pool: every agent (logger, nutritionist, trainer) reads from and writes to a single Mongo + Qdrant store keyed only by user_id.        
     Retrieval is pure cosine top-k. Memories carry a source_agent field but it is never used for filtering. Relationships, importance, and confidence are stored but inert. The system  
     performs adequately as a retrieval engine but cannot support the cognitive properties the next generation of agents needs: per-agent isolation, selective sharing, evolving         
     abstractions, attention-driven retrieval, causal reasoning, and offline consolidation.                                                                                              
                                                                                                                                                                                         
     This document specifies a redesign that treats memory as a living adaptive system rather than static storage. It draws from CPU cache hierarchy (locality, eviction, coherence),    
     operating systems (scheduling, ACL, paging), hippocampus/cortex models (replay, consolidation, abstraction), and cognitive architectures (spreading activation, blackboards, working
      memory). The output is a four-tier cognitive substrate that supports two existing agent families (logger / nutritionist / trainer — health) and three new ones (project / school / 
     research — productivity), with isolation by default and explicit, policy-driven sharing.                                                                                            
                                                                                                                                                                                         
     Locked decisions (informing this design):                                                                                                                                           
     1. Both agent families coexist on the new substrate.                                                                                                                                
     2. Neo4j as the L3 graph engine, deployed as a Docker service.                                                                                                                      
     3. Plan file is the architecture spec; implementation roadmap is a separate later document.                                                                                         
     4. Activation state lives in-process only, rebuilt on restart from L2/L3 snapshots + recent backing-store events.                                                                   
     5. Consolidation runs hybrid: light idle-triggered (every ~60 s of inactivity) + heavy scheduled (03:00 daily, with fallback on >30 min idle).

     ---
     A. Problem Framing

     A flat shared pool fails as soon as the system has more than one cognitive concern at once. Concretely:

     1. No isolation. Anything written by logger can leak into research retrieval. A workout note can pollute a thesis-related query if its embedding happens to be close. There is no
     way to keep school context out of project context.
     2. No selective sharing. When cross-agent context is needed (e.g. "remind me what I told the project agent about my schedule when planning my study time"), there is no protocol —
     sharing is all-or-nothing.
     3. Vector cosine alone is brittle. It ranks by surface similarity. It cannot answer "what depends on this?", "what did I conclude after X?", "what contradicts Y?", or "what
     abstractions have emerged from these episodes?".
     4. Memories are anonymous chunks. No lineage, no evidence, no contradictions, no version history. The system cannot explain why it surfaced a memory, or whether the memory has been
      refined or superseded.
     5. Storage-shaped, not cognition-shaped. Memories sit passively until queried. There is no priming, no spreading activation, no working set, no offline consolidation, no
     abstraction induction. The system never "thinks" between requests.

     A pure storage hierarchy (L1/L2/L3 with eviction) fixes problem 1–3. It does not fix 4–5. To fix 4–5 the hierarchy must also be a cognitive hierarchy: each tier has not just
     different latency and capacity, but different cognitive role.

     ┌─────────┬───────────────────────────┬─────────────────────────────────────────────────┐
     │  Tier   │       Storage view        │                 Cognitive view                  │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L1      │ per-agent hot cache       │ working memory blackboard                       │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L2      │ shared warm cache         │ collaborative cognitive workspace               │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L3      │ persistent semantic store │ semantic cortex (concepts, episodes, lineage)   │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ Backing │ append-only event log     │ hippocampal raw record / autobiographical trace │
     └─────────┴───────────────────────────┴─────────────────────────────────────────────────┘

     The two views are complementary. Storage gives us latency budgets, eviction policies, locality. Cognition gives us what each tier is for and how the tiers should interact
     dynamically.

     ---
     B. Research-Backed Design Principles

     The design borrows from the following, with specific reuse called out:

     - Spreading activation networks (Collins & Loftus 1975; Anderson's ACT-R 1993): activation spreads from queried nodes to associatively connected nodes with decay. Reused as: the L3
      retrieval engine, implemented as Personalized PageRank over Neo4j with edge weights derived from co-activation history.
     - Hippocampus / neocortex consolidation (McClelland, McNaughton & O'Reilly 1995): hippocampus stores episodic specifics; cortex slowly extracts semantics through replay. Reused as:
      L2 holds episodic/recent; L3 holds semantic/consolidated; the heavy consolidation cycle is the "replay" that promotes episodes to semantics.
     - Working memory (Baddeley 1992): a small, attention-bound workspace with executive control, separate from long-term store. Reused as: L1 is per-agent working memory holding
     activation state, working set, active goals, contradictions, hypotheses.
     - Blackboard architectures (HEARSAY-II 1980): agents post and read partial results on a shared structure. Reused as: L2 acts as the collaborative blackboard for inter-agent
     coordination.
     - Personalized PageRank (Haveliwala 2002): teleport-biased PageRank seeded on query-relevant nodes. Reused as: the activation-spreading kernel in the retrieval cascade.
     - HippoRAG (Gutierrez et al. 2024): combines PPR over a knowledge graph with vector search to mimic hippocampal indexing. Reused as: the L2→L3 escalation pattern.
     - GraphRAG (Microsoft 2024): community detection + hierarchical summarization over a knowledge graph. Reused as: the abstraction-induction step in the heavy consolidation cycle.
     - MemGPT (Packer et al. 2023): virtual memory model with paging between fast and slow tiers. Reused as: the L1↔L2 paging discipline.
     - Generative Agents reflection (Park et al. 2023): agents periodically synthesize higher-order insights from raw observations. Reused as: the heavy consolidation cycle's reflective
      synthesis step.
     - CPU cache coherence (MESI): shared/exclusive/invalid states for cache lines. Reused as: the L2 coherence model when multiple agents read the same memory simultaneously and one
     rewrites it.

     Design principles distilled from the above:
     - Prefer structured memory objects over raw text blobs.
     - Prefer hybrid retrieval (vector + graph + temporal + structured) over pure embedding search.
     - Prefer explicit relations over implicit similarity alone.
     - Prefer tiered storage with cognitive roles over a single flat pool.
     - Prefer policy-aware sharing over unrestricted global access.
     - Prefer consolidation, abstraction, and forgetting over unbounded accumulation.
     - Prefer provenance, versioning, and lineage for every memory object.
     - Prefer activation-driven retrieval over query-driven retrieval.

     ---
     C. Memory Hierarchy Design

                        ╔════════════════════════════════════════════════╗
                        ║  L1 — Working Memory Blackboard (per agent)    ║
                        ║  In-process Python. ~100 items per agent.      ║
                        ║  activation map, working set, goals,           ║
                        ║  contradictions, hypotheses, scratchpad.       ║
                        ║  Latency: <100 µs. Volatile.                   ║
                        ╚────────────────────┬───────────────────────────╝
                                             │ miss / cold
                        ╔════════════════════▼═══════════════════════════╗
                        ║  L2 — Collaborative Cognitive Workspace        ║
                        ║  Qdrant warm + Mongo (last ~30 d) + ttl-cache  ║
                        ║  shared concepts, transient summaries, active  ║
                        ║  cross-agent goals, attention pool.            ║
                        ║  Latency: <50 ms. Coherent across agents.      ║
                        ╚────────────────────┬───────────────────────────╝
                                             │ insufficient → expand
                        ╔════════════════════▼═══════════════════════════╗
                        ║  L3 — Semantic Cortex                          ║
                        ║  Neo4j (graph) + Mongo cold + Qdrant cold      ║
                        ║  concepts, episodes, lineage, causal chains,   ║
                        ║  temporal narratives, induced abstractions.    ║
                        ║  Latency: <300 ms. Source of truth.            ║
                        ╚════════════════════┬═══════════════════════════╝
                                             │ replay / audit only
                        ╔════════════════════▼═══════════════════════════╗
                        ║  Backing Store — Immutable Event Log           ║
                        ║  Mongo `raw_events` (append-only) + JSONL      ║
                        ║  raw user messages, extraction outputs,        ║
                        ║  agent responses, full provenance.             ║
                        ║  Latency: irrelevant. Never deleted.           ║
                        ╚════════════════════════════════════════════════╝

     L1 — Working Memory Blackboard (per agent)

     Cognitive role. The agent's current thought space. Not "the last few user messages" — the active reasoning state.

     Contents (per agent):
     - activation_map: Dict[memory_id, float] — current activation level of memories the agent has touched recently.
     - working_set: OrderedDict[memory_id, Memory] — capacity ~50, items resident from the most recent cascade.
     - attention_weights: Dict[entity_name, float] — what the agent is "attending to" right now (e.g. {"thesis": 0.9, "Prof. Liu": 0.7}).
     - active_goals: List[Goal] — goals the agent is currently advancing.
     - open_hypotheses: List[Hypothesis] — tentative claims not yet confirmed.
     - contradictions: List[ContradictionEvent] — detected conflicts between current input and existing memory.
     - scratchpad: List[str] — short-term notes the agent generates during reasoning.
     - last_cascade: RetrievalTrace — the multi-stage retrieval result of the last user turn (kept for reflection).

     Granularity. One blackboard per (user_id, agent_name) pair. Lives in a single FastAPI process.

     Update policy. Every read promotes touched memories into the working set with bumped activation. Every write seeds activation. A 1 Hz background tick decays activation
     exponentially (a ← a · exp(-λ·dt)). Items below a threshold drop out of the working set.

     Why per-agent, not global at L1. Agents must stay isolated by default. A single L1 would erase the isolation guarantee. Each agent has its own attentional state.

     L2 — Collaborative Cognitive Workspace

     Cognitive role. A shared blackboard where agents post and read recent results. Not just cache — the place where collaboration happens.

     Contents.
     - Vector index (Qdrant warm collection mem_warm): embeddings of memories created in the last ~30 days. ACL-aware via payload filters.
     - Doc store (Mongo memories_warm): full memory documents for the same window. Read-through cache for L1.
     - Shared concept board (Mongo concept_board): active cross-agent concepts and transient summaries. Agents post here when they detect a concept worth surfacing to others.
     - Active goals registry (Mongo active_goals): unresolved cross-agent goals (e.g., user said "remind me about X", any agent may resolve it).
     - Attention pool (in-process, replicated lazily across workers if multi-worker): a small pub/sub channel where agents broadcast "I am paying attention to entity Y" — other agents
     can opt to spread their own activation toward Y.
     - Coherence ledger (Mongo coherence_log): MESI-style state per memory_id when agents have it loaded into their L1.

     Granularity. Per user_id, shared across all agents within that user. ACL applied at read time.

     Update policy. Memory enters L2 the moment it is written. Eviction to L3 happens on age (>30 d) AND low recent activation, OR on explicit consolidation. A memory in L3 may be
     promoted back to L2 if activation crosses a threshold (e.g. user starts mentioning an old project again).

     L3 — Semantic Cortex

     Cognitive role. Long-term semantic memory. Holds what is known and how it connects, not what was just said.

     Contents.
     - Neo4j graph (semantic_cortex database):
       - Node labels: Memory, Entity, Episode, Concept, Decision, Goal, Task, Topic, Agent, User.
       - Edge types: MENTIONS, DEPENDS_ON, DERIVED_FROM, SUPERSEDES, CONTRADICTS, ABSTRACTION_OF, PART_OF, AUTHORED_BY, SHARED_WITH, CO_ACTIVATED (weighted), CAUSES, PRECEDES
     (temporal), REFINES.
       - Weighted edges carry weight and last_reinforced_at.
     - Cold doc store (Mongo memories_cold): full memory docs older than 30 d.
     - Cold vector index (Qdrant mem_cold): embeddings for cold memories, kept for long-tail semantic recall.
     - Concept hierarchies (Neo4j Concept nodes with ABSTRACTION_OF edges to source memories): induced by the heavy consolidation cycle.
     - Temporal narratives (Episode nodes linked by PRECEDES edges): consolidated story arcs ("the spring-2026 thesis review cycle").

     Granularity. Per user_id, partitioned in Neo4j by :User {id: $user_id} root with all derived nodes attached.

     Update policy. Mostly written by the consolidation workers, not by the request path. Memory-node creation can happen synchronously on write (cheap), but edge induction
     (CO_ACTIVATED weights, CAUSES, ABSTRACTION_OF) is the consolidator's job.

     Backing Store — Immutable Event Log

     Cognitive role. Autobiographical trace. The ground-truth record of everything that happened. Never deleted. Used for replay during heavy consolidation, for audit, and for
     explainability.

     Contents.
     - raw_events collection in Mongo, append-only, with fields:
       - event_id (ULID, sortable)
       - user_id, agent_name, timestamp
       - event_type: user_message | agent_response | extraction_output | consolidation_event
       - payload: full message / response / extraction JSON
       - produced_memory_ids: list (for traceability)

     Update policy. Append-only, no updates, no deletes. Optional periodic export to JSONL on disk for offline replay.

     ---
     D. Retrieval Architecture (Multi-Stage Activation Cascade)

     Retrieval is no longer "embed query, top-k, return." It is a six-stage cascade in which memory is activated through progressive expansion, then synthesized.

     Stage 1: Seed activation         (cheap, parallel)
     Stage 2: Working-set expansion   (L1 only, instant)
     Stage 3: Spreading activation    (L3 graph, PPR)
     Stage 4: Causal/temporal traversal (L3 graph, directed walks)
     Stage 5: Filter + synthesis      (ACL, dedupe, rank)
     Stage 6: Reflective update       (writes back to L1 + co-activation log)

     Stage 1 — Seed Activation (parallel, ~20 ms)

     Three seed sources are computed in parallel:

     async def seed(query, agent, user_id):
         return await asyncio.gather(
             vector_seed(query, user_id, agent_visible_only=True, k=5),  # Qdrant warm
             entity_seed(query, user_id),                                  # NER → Neo4j entity match
             working_set_seed(agent, user_id),                             # L1 lookup
         )

     Each returns Dict[memory_id, seed_strength]. The three are merged by max.

     Stage 2 — Working-Set Expansion (~1 ms)

     def expand_working_set(seeds, agent_blackboard):
         expanded = dict(seeds)
         for mid, strength in seeds.items():
             # Pull in items already co-resident in this agent's working set
             for neighbor_id in agent_blackboard.recently_co_activated(mid):
                 expanded[neighbor_id] = max(expanded.get(neighbor_id, 0), 0.5 * strength)
         return expanded

     This implements priming: items already on the agent's mind get a head start.

     Stage 3 — Spreading Activation via Personalized PageRank (~50–150 ms)

     def spread_activation(seeds, user_id, agent, max_iter=8, alpha=0.15, eps=1e-3):
         # Personalized PageRank over Neo4j, restricted to agent-visible nodes.
         # alpha = restart probability (teleport back to seeds)
         activation = dict(seeds)
         for _ in range(max_iter):
             new_act = defaultdict(float)
             for node, val in activation.items():
                 if val < eps: continue
                 new_act[node] += alpha * val            # restart component
                 edges = neo4j_neighbors(node, user_id, agent_visibility=agent)
                 total_w = sum(e.weight for e in edges) or 1.0
                 for e in edges:
                     new_act[e.target] += (1 - alpha) * val * (e.weight / total_w)
             if converged(activation, new_act, eps): break
             activation = new_act
         return activation

     Edge weights come from CO_ACTIVATED strength (reinforced by consolidation), DEPENDS_ON (high, fixed), MENTIONS (medium), CONTRADICTS (negative — propagation actually suppresses the
      target), SUPERSEDES (zero — superseded memories are dampened).

     This is implemented in Cypher with APOC's apoc.algo.pageRankWithWeights or a Python-side Power-method over an extracted subgraph for tighter control.

     Stage 4 — Causal / Temporal Traversal (~30–100 ms)

     For the top-N activated nodes, traverse directed edges:
     - Backward along CAUSES, DERIVED_FROM to recover provenance / why.
     - Forward along PRECEDES to recover what happened next.
     - Outward along REFINES, SUPERSEDES to find the latest version.

     Cypher example:
     MATCH (seed:Memory {id: $mid})
     OPTIONAL MATCH path1 = (cause:Memory)-[:CAUSES*1..3]->(seed)
     OPTIONAL MATCH path2 = (seed)-[:PRECEDES*1..3]->(later:Memory)
     OPTIONAL MATCH (latest:Memory)-[:SUPERSEDES*0..]->(seed)
     RETURN seed, collect(DISTINCT cause), collect(DISTINCT later), latest

     Stage 5 — Filter + Synthesis (~10 ms)

     def synthesize(activated, agent, query):
         # ACL filter: only memories visible to this agent
         visible = [m for m in activated if acl_check(m, agent)]
         # Drop superseded
         visible = drop_superseded(visible)
         # Drop contradicted (unless query is asking about contradictions)
         visible = drop_contradicted(visible) if not query.asks_about_conflicts() else visible
         # Lineage dedupe: when an abstraction and its source memories both made it,
         # keep the abstraction unless the query needs specifics
         visible = lineage_dedupe(visible, prefer=abstraction_or_specifics(query))
         # Final rank
         ranked = sorted(
             visible,
             key=lambda m: (
                 m.activation
                 * m.confidence
                 * recency_decay(m.last_accessed_at)
             ),
             reverse=True,
         )
         return ranked[:top_n]

     Stage 6 — Reflective Update (~5 ms, async)

     After the agent generates its response, the cascade trace is fed back:

     async def reflect(cascade_trace, agent_blackboard):
         # Bump activation of every memory used
         for mid in cascade_trace.used_memory_ids:
             agent_blackboard.activate(mid, +0.3)
         # Log co-activations: every pair (a,b) used together gains weight
         pairs = combinations(cascade_trace.used_memory_ids, 2)
         co_activation_log.append({"pairs": pairs, "ts": now()})
         # If query revealed a contradiction, queue it for the consolidator
         if cascade_trace.contradictions_detected:
             consolidator_queue.put(ContradictionTask(cascade_trace.contradictions_detected))

     The co-activation log is what the light consolidation cycle uses to reinforce CO_ACTIVATED edges in Neo4j.

     End-to-end latency budget

     ┌─────────────────────────────┬───────────────────┬────────────┐
     │            Stage            │      Target       │ Worst case │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 1 Seed                      │ 20 ms             │ 80 ms      │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 2 Working set               │ 1 ms              │ 5 ms       │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 3 Spreading activation      │ 80 ms             │ 200 ms     │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 4 Causal traversal          │ 50 ms             │ 150 ms     │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 5 Synthesis                 │ 10 ms             │ 30 ms      │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 6 Reflect (async, off-path) │ 0 (post-response) │ —          │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ Retrieval total             │ ~160 ms           │ ~465 ms    │
     └─────────────────────────────┴───────────────────┴────────────┘

     LLM generation time is unchanged from today (Ollama bound).

     ---
     E. Memory Write/Update Architecture

     Write pipeline (sync portion + async tail)

     async def write_memory(user_msg, user_id, agent):
         # 1. Backing store FIRST — immutable, sync, fast
         event_id = await raw_events.insert({
             "user_id": user_id,
             "agent_name": agent.name,
             "timestamp": now(),
             "event_type": "user_message",
             "payload": user_msg,
         })
         # 2. Off the request path: schedule extraction
         asyncio.create_task(_extract_and_persist(event_id, user_msg, user_id, agent))
         return event_id

     The async tail:

     async def _extract_and_persist(event_id, msg, user_id, agent):
         # 3. Extract: LLM JSON → rule-based → fast_payload (existing path, evolved)
         extraction = await extract(msg, agent)
         if not extraction and not agent.force_store: return
         if not extraction: extraction = fast_store_payload(msg)
         # 4. Resolve duplicates
         candidate = build_memory(extraction, user_id, agent, event_id)
         duplicate = await find_duplicate(candidate, user_id, sim=0.92)
         if duplicate:
             await reinforce(duplicate, candidate)  # bump evidence, increment confidence,
                                                     # add provenance link, do NOT create new
             return
         # 5. Mint memory_id, set lineage scaffold
         candidate.id = ulid()
         candidate.lineage = Lineage(
             source_event_id=event_id,
             derived_from=[],
             evidence=[event_id],
             version=1,
         )
         # 6. Triple-write: Mongo doc, Qdrant vector, Neo4j node + initial edges
         await asyncio.gather(
             mongo.memories_warm.insert(candidate.to_doc()),
             qdrant.upsert(collection="mem_warm", point=candidate.to_point()),
             neo4j_project(candidate),  # creates Memory node + AUTHORED_BY + MENTIONS edges
         )
         # 7. Update agent blackboard (seed activation in writer's L1)
         blackboard_for(user_id, agent).activate(candidate.id, +1.0)
         # 8. Queue consolidation candidates
         light_worker_queue.put(DedupeSweep(user_id, window="1h"))

     Memory schema (Mongo memories_warm / memories_cold)

     {
       "_id": "01HMZK...",                    // ULID, sortable by creation time
       "user_id": "user_abc",
       "version": 1,
       "tier": "warm",                         // warm | cold
       "intent": "task",                       // task|preference|fact|correction|goal|contextual_reference
       "memory_type": "Procedural",            // Semantic|Episodic|Procedural|Preference|Concept|Decision
       "summary": "…",
       "raw_text_event_id": "01HMZJ...",       // pointer into raw_events
       "entities": [{"name": "thesis", "kind": "topic"}, …],
       "embedding_id": "01HMZK...",            // matches Qdrant point id (== _id by convention)

       // Lineage / provenance
       "lineage": {
         "source_event_id": "01HMZJ...",
         "derived_from": [],                   // memory_ids this was derived from
         "supersedes":   [],                   // memory_ids this replaces
         "abstraction_of": [],                 // memory_ids this generalizes
         "evidence":      ["01HMZJ..."],       // event_ids supporting this memory
         "contradicted_by": [],                // memory_ids contradicting this
       },

       // Trust + value
       "confidence": 0.8,                      // updated on reinforcement
       "importance": 0.7,                      // from INTENT_DEFAULTS, may drift
       "novelty":    0.6,                      // 1 - max_sim_at_creation
       "utility":    null,                     // computed by light consolidator (co-activation centrality)

       // Activation (snapshotted for warm restart; live state in L1)
       "last_activation":   0.0,
       "last_accessed_at":  "2026-05-07T12:00:00Z",
       "access_count":      0,

       // Authorship + ACL
       "agent_owner": "project",
       "visibility":  "private",               // private | shared | public
       "shared_with": [],                      // list of agent names if visibility=shared

       // Timestamps
       "created_at": "...", "updated_at": "..."
     }

     Qdrant payload

     Mirrors a subset for filterable retrieval: user_id, agent_owner, visibility, shared_with, memory_type, intent, created_at (epoch ms). All filters native Qdrant payload indexes — no
      Python post-filter (fixes the current vector.py:70 post-filter bug).

     Neo4j projection (per memory)

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
     UNWIND $shared_with AS shared_agent
       MERGE (sa:Agent {name: shared_agent})
       MERGE (m)-[:SHARED_WITH]->(sa)

     CO_ACTIVATED, CAUSES, DEPENDS_ON, SUPERSEDES, CONTRADICTS, ABSTRACTION_OF edges are written by the consolidation workers, not at write time.

     Promotion / demotion

     ┌─────────────────┬──────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
     │   Transition    │                         Trigger                          │                                               Action                                               │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L2 → L1         │ retrieval cascade selects it                             │ load doc into agent blackboard; bump activation                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L1 → L1         │ re-used in subsequent turn                               │ bump activation; reset decay timer                                                                 │
     │ (refresh)       │                                                          │                                                                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L1 → (drop)     │ activation < ε                                           │ drop from working set; activation snapshot persisted to L2 doc                                     │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L2 → L3         │ age > 30 d AND activation < threshold for 7 d            │ move doc warm → cold collection; vector warm → cold; Neo4j node unchanged                          │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L3 → L2         │ retrieval pulls cold memory N times in 24 h              │ promote: copy doc back to warm, vector to mem_warm, mark tier: warm                                │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L3 → Concept    │ heavy consolidator detects ≥3 similar memories with      │ induce Concept node; create ABSTRACTION_OF edges to source memories                                │
     │                 │ co-activation                                            │                                                                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Any → Forgotten │ utility < 0.05 AND confidence < 0.3 AND age > 90 d       │ mark forgotten: true (do not delete; remove from L2 + Qdrant warm + cold; Neo4j node + edges       │
     │                 │                                                          │ retained for lineage)                                                                              │
     └─────────────────┴──────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

     ---
     F. Cache Replacement & Eviction Policies — Semantic Economics

     Eviction is not LRU. It is value-based, with multiple signals.

     Utility scoring

     def utility(memory):
         return (
             0.30 * recency_score(memory)            # exp(-(now - last_access)/τ_r)
           + 0.25 * frequency_score(memory)          # log(1 + access_count)
           + 0.20 * importance(memory)               # from INTENT_DEFAULTS, may drift
           + 0.15 * centrality(memory)               # PageRank in CO_ACTIVATED subgraph
           + 0.10 * lineage_value(memory)            # 1 if other memories derive from it, else 0
           - 0.20 * staleness_penalty(memory)        # contradicted, superseded, untouched-and-old
           - 0.30 * conflict_penalty(memory)         # contradictions outweigh evidence
         )

     utility is recomputed by the light consolidator hourly. It drives:
     - L2 → L3 demotion: utility < 0.4 AND age > 30 d.
     - L3 → forgotten: utility < 0.05 AND age > 90 d AND no recent retrieval.
     - L1 capacity eviction: when working set exceeds 50, drop lowest-utility item.

     Decay

     Activation decays exponentially in L1:

     λ = 1 / 600  # half-life ≈ 7 minutes
     def decay_tick(blackboard, dt):
         for mid in list(blackboard.activation_map):
             a = blackboard.activation_map[mid] * exp(-λ * dt)
             if a < 1e-3:
                 del blackboard.activation_map[mid]
                 blackboard.working_set.pop(mid, None)
             else:
                 blackboard.activation_map[mid] = a

     Edge decay in Neo4j (light consolidator, hourly):

     MATCH ()-[r:CO_ACTIVATED]->()
     WHERE r.last_reinforced_at < datetime() - duration('PT1H')
     SET r.weight = r.weight * 0.95
     WITH r WHERE r.weight < 0.05
     DELETE r

     Abstraction induction (heavy consolidator)

     def induce_abstractions(user_id):
         # 1. Find candidate clusters: memories with high pairwise CO_ACTIVATED weight
         #    that are NOT already abstracted.
         clusters = neo4j.query("""
             MATCH (m1:Memory)-[r:CO_ACTIVATED]-(m2:Memory)
             WHERE r.weight > 0.7 AND NOT (m1)<-[:ABSTRACTION_OF]-(:Concept)
             WITH m1, collect(DISTINCT m2) AS group
             WHERE size(group) >= 2
             RETURN m1, group
         """)
         for cluster in clusters:
             members = cluster.group + [cluster.m1]
             if avg_pairwise_similarity(members) < 0.75: continue   # incoherent cluster
             # 2. LLM call: induce a concept summary
             summary = await llm.abstract(members)
             confidence = lineage_confidence(members)
             # 3. Create Concept node + edges
             concept_id = await neo4j.create_concept(summary, confidence, source_ids=[m.id for m in members])
             # 4. Provenance
             log_event("consolidation", produced=concept_id, sources=[m.id for m in members])

     ---
     G. Relationship Modeling

     Node types

     ┌──────────┬────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
     │  Label   │            Purpose             │                            Key fields                            │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ User     │ partition root                 │ id                                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Agent    │ author / sharing target        │ name                                                             │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Memory   │ atomic memory object           │ id, type, intent, confidence, importance, visibility, created_at │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Entity   │ named thing referenced         │ name, kind (person/place/topic/artifact)                         │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Episode  │ bounded experience             │ id, time_window, narrative                                       │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Concept  │ induced abstraction            │ id, label, summary, induced_at, confidence                       │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Decision │ recorded choice                │ id, made_at, rationale                                           │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Goal     │ active objective               │ id, status, due_at                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Task     │ actionable item                │ id, status, due_at                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Topic    │ broad theme (thesis, fitness…) │ name                                                             │
     └──────────┴────────────────────────────────┴──────────────────────────────────────────────────────────────────┘

     Edge types

     ┌────────────────┬──────────────────┬─────────────┬───────────────────────────────────────┐
     │      Type      │    Direction     │   Weight    │                Meaning                │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ OWNED_BY       │ Memory → User    │ —           │ partition                             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ AUTHORED_BY    │ Memory → Agent   │ —           │ who wrote it                          │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ SHARED_WITH    │ Memory → Agent   │ —           │ ACL grant                             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ MENTIONS       │ Memory → Entity  │ static      │ named-entity reference                │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ PART_OF        │ Memory → Episode │ —           │ episodic grouping                     │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ DEPENDS_ON     │ Memory → Memory  │ static high │ logical dependency                    │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ DERIVED_FROM   │ Memory → Memory  │ —           │ provenance                            │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ SUPERSEDES     │ Memory → Memory  │ —           │ replaces                              │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CONTRADICTS    │ Memory ↔ Memory  │ —           │ conflict marker                       │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ REFINES        │ Memory → Memory  │ —           │ non-replacing improvement             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ ABSTRACTION_OF │ Concept → Memory │ —           │ concept generalizes memory            │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CAUSES         │ Memory → Memory  │ static      │ causal claim                          │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ PRECEDES       │ Memory → Memory  │ static      │ temporal order                        │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CO_ACTIVATED   │ Memory ↔ Memory  │ learned     │ reinforced by retrieval co-occurrence │
     └────────────────┴──────────────────┴─────────────┴───────────────────────────────────────┘

     CO_ACTIVATED is the only edge type the system learns automatically; everything else is asserted at write time, by the consolidator on explicit triggers, or by user feedback.

     Hyperedge / N-ary modeling

     Some relations are inherently n-ary: "in episode E, agent A made decision D supported by memories M1, M2, contradicted by M3." Represent these as first-class nodes:

     (:Decision {id, made_at})
        -[:IN_EPISODE]->(:Episode)
        -[:BY_AGENT]->(:Agent)
        -[:SUPPORTED_BY]->(:Memory)+
        -[:OPPOSED_BY]->(:Memory)*

     This is the "reified relationship" pattern — preserves the graph's queryability while modeling complex relations.

     ---
     H. Consistency & Access Control

     ACL model

     Every memory has:
     - agent_owner — the agent that wrote it.
     - visibility ∈ {private, shared, public}:
       - private (default): only agent_owner can read.
       - shared: read access granted to agents in shared_with list.
       - public: any agent for the same user_id may read.
     - shared_with: List[agent_name] — populated when visibility = shared.

     Sharing protocols

     Three ways a memory can become shared:

     1. Explicit share. An agent calls share_memory(memory_id, target_agents, reason). Logged in raw_events with the reason.
     2. Type-level policy. Certain memory types are auto-public for the user:
       - Preference memories (e.g., "I prefer mornings") — public by default.
       - Identity facts (name, role, location) — public.
       - Goal memories with cross_agent: true flag — public.
       - Procedural, Episodic, and most Semantic memories — private by default.
     3. Entity-bridged. When an entity is flagged as cross-agent (e.g., a person who appears in both project and school contexts), memories mentioning that entity become discoverable
     via SHARED_WITH propagation — but only the existence and summary are visible to non-owner agents, not the full content. The owner agent must approve full disclosure on first
     cross-agent retrieval.

     Coherence (MESI-inspired)

     When multiple agents have the same memory loaded into their L1 working sets:

     ┌───────────────┬──────────────────────────────────────────────────────────────────┐
     │     State     │                             Meaning                              │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ M (Modified)  │ one agent has loaded it AND is about to write a refinement       │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ E (Exclusive) │ one agent has loaded it; no others                               │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ S (Shared)    │ multiple agents have it loaded; all read-only                    │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ I (Invalid)   │ memory was just superseded or contradicted; agents must re-fetch │
     └───────────────┴──────────────────────────────────────────────────────────────────┘

     The coherence_log collection tracks the state. On a write that modifies a Shared memory, all loaded copies transition to Invalid; the next time those agents touch the memory, they
     re-fetch from L2.

     For a single-user, single-process deployment this is mostly cosmetic — but the protocol is the same one we'd need under multi-worker, so we encode it now.

     Conflict resolution

     When extraction detects a new memory that contradicts an existing one:
     1. Both memories are kept. The new one gets a CONTRADICTS edge to the old.
     2. Confidence of both is recomputed: the more-recently-reinforced one's confidence rises slightly; the older one's drops.
     3. The next retrieval that surfaces either will see both and may trigger a reflective synthesis ("user previously said X, now says Y; treat Y as current unless asked").
     4. The heavy consolidator may eventually introduce a SUPERSEDES edge if the conflict resolves consistently over time.

     ---
     I. Concurrency & Performance

     Worker topology

                          ┌──────────────────────────────┐
                          │  FastAPI (asyncio event loop)│
                          │   - request handlers         │
                          │   - L1 blackboards (in-proc) │
                          │   - retrieval cascade        │
                          └──────┬───────────────────────┘
                                 │ submit
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
       ┌─────────┐         ┌─────────────┐    ┌──────────────────┐
       │ Extract │         │ Light       │    │ Heavy            │
       │ pool    │         │ consolidator│    │ consolidator     │
       │ (1 task │         │ (idle ≥60s) │    │ (cron 03:00 +    │
       │ per req)│         │             │    │  fallback ≥30min)│
       └─────────┘         └─────────────┘    └──────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                 ┌──────────────────────────────┐
                 │ Mongo · Qdrant · Neo4j · LLM │
                 └──────────────────────────────┘

     - Request handlers are async. Retrieval cascade fans out to Mongo/Qdrant/Neo4j with asyncio.gather.
     - Extract pool: one asyncio task per request for the LLM extraction tail. Cap concurrent in-flight extractions at 4 to avoid Ollama saturation.
     - Light consolidator: a single asyncio task on the same loop, woken by an idle detector (no requests for 60 s). Runs: edge re-weighting from co_activation_log, decay sweep, dedupe
     pass on memories created in the last hour, utility recompute. No LLM calls. Yields control on every iteration.
     - Heavy consolidator: APScheduler job at 03:00, plus an idle-fallback (>30 min idle). Runs in a separate process (consolidator.py) to avoid blocking the request event loop.
     Operations: cluster detection, abstraction induction (LLM), graph topology restructuring, episodic→semantic conversion, cold-tier migration.

     Locks and atomicity

     - Per-memory write lock: an asyncio.Lock keyed by memory_id, held during the triple-write (Mongo + Qdrant + Neo4j). Prevents concurrent reinforcement races.
     - Neo4j transactions: every write is one transaction; reads are auto-committed.
     - Activation updates (in-process): single-writer per blackboard. Since each blackboard is per-(user, agent) and we have one event loop, atomicity is automatic.

     Latency targets

     ┌────────────────────────────┬──────────────┬────────────────────────────────────────────────────┐
     │            Path            │    Target    │                        How                         │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L1 read                    │ < 100 µs     │ dict lookup                                        │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L2 vector + Mongo (warm)   │ < 50 ms      │ parallel gather                                    │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L3 PPR (10k node subgraph) │ < 150 ms     │ bounded depth + edge-weight cutoff                 │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Full cascade               │ < 300 ms p95 │ budget per stage                                   │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Backing-store append       │ < 5 ms       │ unindexed insert, fire-and-forget on critical path │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Async write tail           │ invisible    │ runs after response sent                           │
     └────────────────────────────┴──────────────┴────────────────────────────────────────────────────┘

     ---
     J. Customization for Your Use Case

     Agent registry (extends current app/agents/__init__.py)

     AGENTS = {
         # Health family (existing)
         "logger":       AgentSpec("logger",       "...", force_store=True,  default_visibility="private"),
         "nutritionist": AgentSpec("nutritionist", "...", force_store=False, default_visibility="private"),
         "trainer":      AgentSpec("trainer",      "...", force_store=False, default_visibility="private"),

         # Productivity family (new)
         "project":      AgentSpec("project",      "...", force_store=False, default_visibility="private"),
         "school":       AgentSpec("school",       "...", force_store=False, default_visibility="private"),
         "research":     AgentSpec("research",     "...", force_store=False, default_visibility="private"),
     }

     # Cross-agent sharing policy (new file: app/agents/sharing_policy.py)
     SHARING_POLICY = {
         # Auto-public memory types regardless of authoring agent
         "auto_public_types": {"Preference", "Identity"},
         # Cross-family entities (mentions trigger cross-family discoverability)
         "bridge_entities": ["calendar", "schedule", "deadline", "energy_level", "sleep"],
         # Family groupings (memories within a family see each other more readily)
         "families": {
             "health":       {"logger", "nutritionist", "trainer"},
             "productivity": {"project", "school", "research"},
         },
     }

     Worked example 1 — Project memory item

     User → project: "I'm migrating the API to FastAPI by June 12, blocked on auth refactor."

     Write flow:
     1. Append to raw_events (id EV_001).
     2. Extract → {intent: "task", memory_type: "Procedural", summary: "Migrate API to FastAPI", entities: [{name: "FastAPI", kind: "tool"}, {name: "auth refactor", kind: "topic"}],
     importance: 0.9, confidence: 0.8}.
     3. Resolve duplicates: vector search finds nothing similar. New memory M_001.
     4. Lineage: source_event_id=EV_001, evidence=[EV_001], version=1.
     5. Mongo memories_warm insert. Qdrant warm upsert. Neo4j: Memory{M_001} -[:AUTHORED_BY]-> Agent{project}, -[:MENTIONS]-> Entity{FastAPI}, -[:MENTIONS]-> Entity{auth refactor}.
     Visibility: private (project only).
     6. project blackboard: M_001 activation = 1.0; entity FastAPI weight = 0.9.
     7. Light consolidator queue: dedupe-sweep within last hour.

     Later — user → school: "Studying API design patterns for my distributed systems class."

     Retrieval flow:
     1. Seed: vector hits in school-visible memories (none yet); entity match Entity{API} overlaps with M_001's MENTIONS Entity{FastAPI} via Topic node Topic{API design}.
     2. Spread: PPR from Entity{API design} reaches M_001 — but stage 5 ACL filter drops it because M_001.visibility = private and M_001.agent_owner = project.
     3. school agent does not see project work. Isolation preserved.

     Cross-agent fetch (user explicitly asks): "What am I working on at work that overlaps with my distributed systems class?"

     The research agent (or whichever is asked) marks the query as cross-family. Stage 5 relaxes ACL to "summary visible, full content blurred." M_001's summary becomes available to the
      responder. Owner agent (project) is notified in the coherence_log.

     Worked example 2 — School memory item evolving into a Concept

     Over three weeks, user mentions to school:
     - "Reviewed Lamport's paper on Time, Clocks." → M_010
     - "Compared Lamport timestamps to vector clocks in HW3." → M_011
     - "Realized happens-before is a partial order, not total." → M_012
     - "TA confirmed: causal consistency uses partial order." → M_013

     Light consolidator (each idle window): notices co-activation between M_010..M_013 rises as the user keeps querying around them. CO_ACTIVATED edge weights climb.

     Heavy consolidator (next 03:00 cycle): cluster {M_010, M_011, M_012, M_013} exceeds the abstraction threshold. LLM call: "Generalize these four memories into a Concept." Output:

     Concept: "Causal ordering in distributed systems"
     Summary: "User has internalized that distributed-time mechanisms (Lamport timestamps,
     vector clocks) implement happens-before as a partial order, supporting causal but not
     total consistency. Reinforced by coursework and TA confirmation."
     Confidence: 0.85

     Concept node C_001 is created in Neo4j with ABSTRACTION_OF edges to all four source memories. Source memories are not deleted. Future retrieval that activates any of M_010..M_013
     also activates C_001 via the abstraction edge — and the synthesis stage prefers C_001 for general questions, drops back to specifics for detailed ones.

     Cross-agent benefit: if the project agent later asks "what do I know about distributed consistency?", it will reach C_001 (because Concepts authored by school agent default to
     type-public if memory_type=Concept — a setting in SHARING_POLICY).

     Selective private memory

     Health-family memories (logger/nutritionist/trainer) stay private to the health family by default — SHARING_POLICY.families ensures they don't bleed into productivity retrieval.
     The bridge_entities list (schedule, energy_level, sleep) is the only allowed crossover: a productivity agent planning study hours can ask about energy_level and reach the relevant
     logger memory through summary-only disclosure.

     Global persistent memory

     User-rooted Concept nodes form the persistent global memory. They're authored by no single agent (or by a synthetic consolidator agent), default to public, and are the canonical
     "what the user knows / believes / pursues over time." This is the layer most worth backing up and most worth surfacing to the user as a UI.

     ---
     K. Recommended Implementation Stack

     ┌─────────────────┬─────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐
     │     Concern     │                                      Tech                                       │                                    Reason                                    │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Web framework   │ FastAPI (async)                                                                 │ already chosen; native asyncio; fits cascade gather                          │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Doc store       │ MongoDB                                                                         │ already chosen; fine for warm/cold/raw_events                                │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Vector index    │ Qdrant                                                                          │ already chosen; payload filters fix current Python post-filter               │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Graph store     │ Neo4j 5.x + APOC                                                                │ locked decision; APOC for PageRank, similarity, batch ops                    │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ LLM             │ Ollama (local) — llama3.2:1b for extraction; consider larger model for heavy    │ local-first; matches current; abstraction calls happen overnight, latency    │
     │                 │ abstraction                                                                     │ irrelevant                                                                   │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Embeddings      │ all-MiniLM-L6-v2 (current) → consider bge-small-en-v1.5 for better recall       │ drop-in replacement; same dim or 384                                         │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Scheduler       │ APScheduler                                                                     │ tiny dep; cron + interval triggers in-process                                │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Activation      │ In-process Python (numpy + dict + heapq for top-k)                              │ locked decision                                                              │
     │ state           │                                                                                 │                                                                              │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Backing store   │ Mongo raw_events collection + optional JSONL export                             │ append-only, no new infra                                                    │
     └─────────────────┴─────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘

     No new languages. Pure Python. No new managed services. Everything Docker-local. Honors "local-first" and "vendor-agnostic" (Neo4j is open-source CE, Mongo CE, Qdrant CE, Ollama).
     Swap Neo4j for any property-graph DB by swapping the Cypher driver behind a GraphStore interface.

     Suggested package layout (informational, NOT the migration plan)

     app/
       cognition/                      ← new module
         blackboard.py                 ← per-agent L1 working memory
         activation.py                 ← decay, spreading-activation Python kernel
         cascade.py                    ← 6-stage retrieval cascade
         consolidator/
           light.py                    ← idle-triggered worker
           heavy.py                    ← scheduled worker (separate process)
           abstraction.py              ← LLM-driven concept induction
       graph/                          ← new module
         neo4j_client.py
         schema.py                     ← node labels + edge types as constants
         projections.py                ← memory→graph projection
         queries.py                    ← parametrized Cypher
       memory/
         summarizer.py                 ← evolved (lineage scaffolding)
         memory_updater.py             ← evolved (triple-write + ACL)
         vector.py                     ← evolved (Qdrant payload filters)
         backing.py                    ← raw_events accessors  (NEW)
       agents/
         __init__.py                   ← AgentSpec + AGENTS (extended)
         sharing_policy.py             ← SHARING_POLICY  (NEW)
     twin/
       orchestrator.py                 ← evolved (cascade-driven)
       prompt_builder.py               ← evolved (memory grouping by tier + lineage)

     ---
     L. Failure Modes & Safeguards

     ┌──────────────────────────────────────────┬───────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
     │                 Failure                  │        Symptom        │                                                  Safeguard                                                   │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Neo4j down at request time               │ Stage 3/4 fail        │ Cascade gracefully degrades to L1+L2 only. Log incident. Marker in response: "degraded_retrieval": true.     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Qdrant down                              │ Stage 1 vector seed   │ Fall back to entity-only seed + working-set seed.                                                            │
     │                                          │ empty                 │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Mongo down                               │ Doc fetches fail      │ 503 to client. No graceful path — Mongo is the truth store.                                                  │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Ollama down                              │ Abstraction induction │ Heavy consolidator skips abstraction; queues for next cycle. Light consolidator unaffected (no LLM).         │
     │                                          │  fails                │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Activation state lost on crash           │ Cold start            │ Reseed: replay last 1 h of raw_events + load top-utility 100 memories per agent into L1. Recovery time < 5   │
     │                                          │                       │ s.                                                                                                           │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Memory bloat (no forgetting)             │ L3 grows unbounded    │ Forgetting policy in §F. Manual purge tool: forget_below_utility(threshold).                                 │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Stale abstractions (concept no longer    │ Wrong retrievals      │ Heavy consolidator re-evaluates Concept→source coherence weekly; flags for refresh.                          │
     │ matches sources)                         │                       │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Conflict drift (contradictions           │ User confused by      │ Reflective synthesis at retrieval time presents both with timestamps; SUPERSEDES edge introduced when        │
     │ accumulate, no resolution)               │ mixed signals         │ conflict consistently resolves toward the newer side over 3+ retrievals.                                     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Over-sharing (private memory leaks)      │ ACL bug               │ ACL filter is a single chokepoint at Stage 5; covered by integration tests. Default-deny on missing          │
     │                                          │                       │ visibility field.                                                                                            │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Retrieval pollution (one agent's noise   │ Bad answers           │ Spreading activation is ACL-restricted at the Cypher level — non-visible nodes are not traversed at all, not │
     │ drowns another's signal)                 │                       │  just filtered post-hoc.                                                                                     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Hot-loop activation positive feedback    │ Same memory always    │ Decay + edge-weight cap (weight ≤ 1.0) + diversity-aware reranker in Stage 5 (penalize too-similar items).   │
     │                                          │ wins                  │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Heavy consolidator runs during user      │ Latency spike         │ Idle detector + APScheduler coalesce=True, max_instances=1; heavy worker yields if it sees a request flag.   │
     │ activity                                 │                       │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Backing store unbounded growth           │ Disk fill             │ JSONL export + Mongo TTL after 1 year of inactivity (Concepts and lineage already preserve what matters from │
     │                                          │                       │  old raw events).                                                                                            │
     └──────────────────────────────────────────┴───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

     ---
     M. Final Blueprint

                      ┌────────────────────────────────────────────────┐
                      │                  USER MESSAGE                  │
                      └───────────────────────┬────────────────────────┘
                                              ▼
             ┌────────────────────────────────────────────────────────┐
             │  FastAPI request handler (asyncio)                     │
             │  • append to raw_events  (backing store)               │
             │  • spawn extract+persist task  (async tail)            │
             │  • run RETRIEVAL CASCADE                               │
             └───────────────────────┬────────────────────────────────┘
                                     ▼
        ┌────────── 6-STAGE RETRIEVAL CASCADE (ACL-aware throughout) ───────────┐
        │ 1 Seed (parallel):  vector  +  entity NER  +  agent's working set    │
        │ 2 Working-set expand (L1 priming)                                    │
        │ 3 Spreading activation: PPR over Neo4j with edge weights             │
        │ 4 Causal/temporal traversal: CAUSES, PRECEDES, SUPERSEDES walks      │
        │ 5 Filter + synthesis: ACL, lineage dedupe, rank by act·conf·recency  │
        │ 6 Reflect (post-response): bump activation, log co-activations       │
        └───────────────────────┬──────────────────────────────────────────────┘
                                ▼
             ┌────────────────────────────────────────────────────────┐
             │  Prompt builder · Ollama · response → user             │
             └───────────────────────┬────────────────────────────────┘
                                     ▼
             ┌────────────────────────────────────────────────────────┐
             │  Async write tail:                                     │
             │   extract → dedupe → triple-write (Mongo+Qdrant+Neo4j) │
             │   bump writer's L1 activation                          │
             │   queue light-consolidator tasks                       │
             └────────────────────────────────────────────────────────┘

                ┌─────────────────────────────────┐
                │   BACKGROUND COGNITION CYCLES   │
                │                                 │
                │  Light  (asyncio, idle ≥60 s):  │
                │   • edge re-weighting           │
                │   • activation decay            │
                │   • dedupe sweep (last hour)    │
                │   • utility recompute           │
                │                                 │
                │  Heavy  (cron 03:00 + ≥30 min): │
                │   • cluster detection           │
                │   • abstraction induction (LLM) │
                │   • episodic → semantic         │
                │   • L2 ↔ L3 migration           │
                │   • concept hierarchy           │
                └─────────────────────────────────┘

     Memory at a glance:
     L1  per-agent working memory blackboard  (in-process, volatile, 100 items)
             ↕ on retrieval / write
     L2  collaborative cognitive workspace    (Mongo warm + Qdrant warm + concept board, 30d window)
             ↕ on age / activation / consolidation
     L3  semantic cortex                      (Neo4j graph + Mongo cold + Qdrant cold)
             ↕ replay / audit only
     backing  immutable event log              (Mongo raw_events, append-only)

     Key invariants:
     - Every memory has an agent_owner and explicit visibility. Default is private.
     - Every memory has a lineage (source event, evidence, derived-from, supersedes, contradicted-by).
     - Every retrieval is a cascade, not a single query.
     - Every co-activation is learned: edges in the graph reflect what actually got used together.
     - Every consolidation is traceable: Concepts know their source memories; Episodes know their member events.
     - Activation is volatile by design: the system "wakes up" each restart and reseeds attention from cortex.

     ---
     Verification (how to know it works once implemented)

     End-to-end checks the future implementation must pass:

     1. Isolation. Write a memory via project with default visibility. Query through school. Assert the memory does not appear in the cascade output.
     2. Selective sharing. Mark a memory visibility=shared, shared_with=[research]. Query through research. Assert it appears. Through school. Assert it does not.
     3. Spreading activation. Insert two memories that share an entity but have low cosine similarity. Query with text matching only one. Assert the other surfaces via the graph.
     4. Abstraction induction. Synthesize 4+ co-activated memories on a topic. Run the heavy consolidator. Assert a Concept node is created with ABSTRACTION_OF edges to all four. Assert
      subsequent retrieval surfaces the Concept for general queries.
     5. Forgetting. Set utility/age thresholds. Run consolidator. Assert low-utility old memories are marked forgotten and disappear from L2 retrieval, while their Neo4j nodes remain
     for lineage.
     6. Warm restart. Kill and restart the FastAPI process during an active session. Assert that within 5 s, top-utility memories are reseeded into the relevant agent blackboards.
     7. Coherence. Two agents load the same shared memory. One refines it. Assert the other's next access fetches the new version (state transitions S → I → S).
     8. Latency. p95 cascade latency < 300 ms on a corpus of 50k memories, measured with pytest-bench or equivalent.
     9. Provenance. Every memory in the L3 cortex traces back to at least one raw_events entry. No orphans.
     10. Conflict survival. Insert two contradictory memories. Assert both are stored, a CONTRADICTS edge exists, and retrieval surfaces both with timestamps.

     When all ten pass, the substrate is doing what this document specifies.

     "/plan open" to edit this plan in VS Code

❯ /plan come uo with a elaborate plan to build the system defined in architecture.md file, divide it into phases for easier implementation                                               
  ⎿  Current Plan                                                                                                                          
     /home/barkha/.claude/plans/you-are-designing-a-robust-nest.md                                                                                                                       
                                                                                                                 
     Hierarchical Semantic Cognition — Architecture Proposal                                                                                                                             
                                                                                                                                                                                         
     ▎ A semantic operating system for collaborative cognition, not a chatbot memory store.                                                                                              
                                                                                                                                                                                         
     ---                                                                                                                                                                                 
     Context                                                                                                                                                                             
                                                                                                                                                                                         
     The current AI-Twins system is a flat memory pool: every agent (logger, nutritionist, trainer) reads from and writes to a single Mongo + Qdrant store keyed only by user_id.        
     Retrieval is pure cosine top-k. Memories carry a source_agent field but it is never used for filtering. Relationships, importance, and confidence are stored but inert. The system  
     performs adequately as a retrieval engine but cannot support the cognitive properties the next generation of agents needs: per-agent isolation, selective sharing, evolving         
     abstractions, attention-driven retrieval, causal reasoning, and offline consolidation.                                                                                              
                                                                                                                                                                                         
     This document specifies a redesign that treats memory as a living adaptive system rather than static storage. It draws from CPU cache hierarchy (locality, eviction, coherence),    
     operating systems (scheduling, ACL, paging), hippocampus/cortex models (replay, consolidation, abstraction), and cognitive architectures (spreading activation, blackboards, working
      memory). The output is a four-tier cognitive substrate that supports two existing agent families (logger / nutritionist / trainer — health) and three new ones (project / school / 
     research — productivity), with isolation by default and explicit, policy-driven sharing.                                                                                            
                                                                                                                                                                                         
     Locked decisions (informing this design):                                                                                                                                           
     1. Both agent families coexist on the new substrate.                                                                                                                                
     2. Neo4j as the L3 graph engine, deployed as a Docker service.                                                                                                                      
     3. Plan file is the architecture spec; implementation roadmap is a separate later document.                                                                                         
     4. Activation state lives in-process only, rebuilt on restart from L2/L3 snapshots + recent backing-store events.                                                                   
     5. Consolidation runs hybrid: light idle-triggered (every ~60 s of inactivity) + heavy scheduled (03:00 daily, with fallback on >30 min idle).

     ---
     A. Problem Framing

     A flat shared pool fails as soon as the system has more than one cognitive concern at once. Concretely:

     1. No isolation. Anything written by logger can leak into research retrieval. A workout note can pollute a thesis-related query if its embedding happens to be close. There is no
     way to keep school context out of project context.
     2. No selective sharing. When cross-agent context is needed (e.g. "remind me what I told the project agent about my schedule when planning my study time"), there is no protocol —
     sharing is all-or-nothing.
     3. Vector cosine alone is brittle. It ranks by surface similarity. It cannot answer "what depends on this?", "what did I conclude after X?", "what contradicts Y?", or "what
     abstractions have emerged from these episodes?".
     4. Memories are anonymous chunks. No lineage, no evidence, no contradictions, no version history. The system cannot explain why it surfaced a memory, or whether the memory has been
      refined or superseded.
     5. Storage-shaped, not cognition-shaped. Memories sit passively until queried. There is no priming, no spreading activation, no working set, no offline consolidation, no
     abstraction induction. The system never "thinks" between requests.

     A pure storage hierarchy (L1/L2/L3 with eviction) fixes problem 1–3. It does not fix 4–5. To fix 4–5 the hierarchy must also be a cognitive hierarchy: each tier has not just
     different latency and capacity, but different cognitive role.

     ┌─────────┬───────────────────────────┬─────────────────────────────────────────────────┐
     │  Tier   │       Storage view        │                 Cognitive view                  │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L1      │ per-agent hot cache       │ working memory blackboard                       │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L2      │ shared warm cache         │ collaborative cognitive workspace               │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ L3      │ persistent semantic store │ semantic cortex (concepts, episodes, lineage)   │
     ├─────────┼───────────────────────────┼─────────────────────────────────────────────────┤
     │ Backing │ append-only event log     │ hippocampal raw record / autobiographical trace │
     └─────────┴───────────────────────────┴─────────────────────────────────────────────────┘

     The two views are complementary. Storage gives us latency budgets, eviction policies, locality. Cognition gives us what each tier is for and how the tiers should interact
     dynamically.

     ---
     B. Research-Backed Design Principles

     The design borrows from the following, with specific reuse called out:

     - Spreading activation networks (Collins & Loftus 1975; Anderson's ACT-R 1993): activation spreads from queried nodes to associatively connected nodes with decay. Reused as: the L3
      retrieval engine, implemented as Personalized PageRank over Neo4j with edge weights derived from co-activation history.
     - Hippocampus / neocortex consolidation (McClelland, McNaughton & O'Reilly 1995): hippocampus stores episodic specifics; cortex slowly extracts semantics through replay. Reused as:
      L2 holds episodic/recent; L3 holds semantic/consolidated; the heavy consolidation cycle is the "replay" that promotes episodes to semantics.
     - Working memory (Baddeley 1992): a small, attention-bound workspace with executive control, separate from long-term store. Reused as: L1 is per-agent working memory holding
     activation state, working set, active goals, contradictions, hypotheses.
     - Blackboard architectures (HEARSAY-II 1980): agents post and read partial results on a shared structure. Reused as: L2 acts as the collaborative blackboard for inter-agent
     coordination.
     - Personalized PageRank (Haveliwala 2002): teleport-biased PageRank seeded on query-relevant nodes. Reused as: the activation-spreading kernel in the retrieval cascade.
     - HippoRAG (Gutierrez et al. 2024): combines PPR over a knowledge graph with vector search to mimic hippocampal indexing. Reused as: the L2→L3 escalation pattern.
     - GraphRAG (Microsoft 2024): community detection + hierarchical summarization over a knowledge graph. Reused as: the abstraction-induction step in the heavy consolidation cycle.
     - MemGPT (Packer et al. 2023): virtual memory model with paging between fast and slow tiers. Reused as: the L1↔L2 paging discipline.
     - Generative Agents reflection (Park et al. 2023): agents periodically synthesize higher-order insights from raw observations. Reused as: the heavy consolidation cycle's reflective
      synthesis step.
     - CPU cache coherence (MESI): shared/exclusive/invalid states for cache lines. Reused as: the L2 coherence model when multiple agents read the same memory simultaneously and one
     rewrites it.

     Design principles distilled from the above:
     - Prefer structured memory objects over raw text blobs.
     - Prefer hybrid retrieval (vector + graph + temporal + structured) over pure embedding search.
     - Prefer explicit relations over implicit similarity alone.
     - Prefer tiered storage with cognitive roles over a single flat pool.
     - Prefer policy-aware sharing over unrestricted global access.
     - Prefer consolidation, abstraction, and forgetting over unbounded accumulation.
     - Prefer provenance, versioning, and lineage for every memory object.
     - Prefer activation-driven retrieval over query-driven retrieval.

     ---
     C. Memory Hierarchy Design

                        ╔════════════════════════════════════════════════╗
                        ║  L1 — Working Memory Blackboard (per agent)    ║
                        ║  In-process Python. ~100 items per agent.      ║
                        ║  activation map, working set, goals,           ║
                        ║  contradictions, hypotheses, scratchpad.       ║
                        ║  Latency: <100 µs. Volatile.                   ║
                        ╚────────────────────┬───────────────────────────╝
                                             │ miss / cold
                        ╔════════════════════▼═══════════════════════════╗
                        ║  L2 — Collaborative Cognitive Workspace        ║
                        ║  Qdrant warm + Mongo (last ~30 d) + ttl-cache  ║
                        ║  shared concepts, transient summaries, active  ║
                        ║  cross-agent goals, attention pool.            ║
                        ║  Latency: <50 ms. Coherent across agents.      ║
                        ╚────────────────────┬───────────────────────────╝
                                             │ insufficient → expand
                        ╔════════════════════▼═══════════════════════════╗
                        ║  L3 — Semantic Cortex                          ║
                        ║  Neo4j (graph) + Mongo cold + Qdrant cold      ║
                        ║  concepts, episodes, lineage, causal chains,   ║
                        ║  temporal narratives, induced abstractions.    ║
                        ║  Latency: <300 ms. Source of truth.            ║
                        ╚════════════════════┬═══════════════════════════╝
                                             │ replay / audit only
                        ╔════════════════════▼═══════════════════════════╗
                        ║  Backing Store — Immutable Event Log           ║
                        ║  Mongo `raw_events` (append-only) + JSONL      ║
                        ║  raw user messages, extraction outputs,        ║
                        ║  agent responses, full provenance.             ║
                        ║  Latency: irrelevant. Never deleted.           ║
                        ╚════════════════════════════════════════════════╝

     L1 — Working Memory Blackboard (per agent)

     Cognitive role. The agent's current thought space. Not "the last few user messages" — the active reasoning state.

     Contents (per agent):
     - activation_map: Dict[memory_id, float] — current activation level of memories the agent has touched recently.
     - working_set: OrderedDict[memory_id, Memory] — capacity ~50, items resident from the most recent cascade.
     - attention_weights: Dict[entity_name, float] — what the agent is "attending to" right now (e.g. {"thesis": 0.9, "Prof. Liu": 0.7}).
     - active_goals: List[Goal] — goals the agent is currently advancing.
     - open_hypotheses: List[Hypothesis] — tentative claims not yet confirmed.
     - contradictions: List[ContradictionEvent] — detected conflicts between current input and existing memory.
     - scratchpad: List[str] — short-term notes the agent generates during reasoning.
     - last_cascade: RetrievalTrace — the multi-stage retrieval result of the last user turn (kept for reflection).

     Granularity. One blackboard per (user_id, agent_name) pair. Lives in a single FastAPI process.

     Update policy. Every read promotes touched memories into the working set with bumped activation. Every write seeds activation. A 1 Hz background tick decays activation
     exponentially (a ← a · exp(-λ·dt)). Items below a threshold drop out of the working set.

     Why per-agent, not global at L1. Agents must stay isolated by default. A single L1 would erase the isolation guarantee. Each agent has its own attentional state.

     L2 — Collaborative Cognitive Workspace

     Cognitive role. A shared blackboard where agents post and read recent results. Not just cache — the place where collaboration happens.

     Contents.
     - Vector index (Qdrant warm collection mem_warm): embeddings of memories created in the last ~30 days. ACL-aware via payload filters.
     - Doc store (Mongo memories_warm): full memory documents for the same window. Read-through cache for L1.
     - Shared concept board (Mongo concept_board): active cross-agent concepts and transient summaries. Agents post here when they detect a concept worth surfacing to others.
     - Active goals registry (Mongo active_goals): unresolved cross-agent goals (e.g., user said "remind me about X", any agent may resolve it).
     - Attention pool (in-process, replicated lazily across workers if multi-worker): a small pub/sub channel where agents broadcast "I am paying attention to entity Y" — other agents
     can opt to spread their own activation toward Y.
     - Coherence ledger (Mongo coherence_log): MESI-style state per memory_id when agents have it loaded into their L1.

     Granularity. Per user_id, shared across all agents within that user. ACL applied at read time.

     Update policy. Memory enters L2 the moment it is written. Eviction to L3 happens on age (>30 d) AND low recent activation, OR on explicit consolidation. A memory in L3 may be
     promoted back to L2 if activation crosses a threshold (e.g. user starts mentioning an old project again).

     L3 — Semantic Cortex

     Cognitive role. Long-term semantic memory. Holds what is known and how it connects, not what was just said.

     Contents.
     - Neo4j graph (semantic_cortex database):
       - Node labels: Memory, Entity, Episode, Concept, Decision, Goal, Task, Topic, Agent, User.
       - Edge types: MENTIONS, DEPENDS_ON, DERIVED_FROM, SUPERSEDES, CONTRADICTS, ABSTRACTION_OF, PART_OF, AUTHORED_BY, SHARED_WITH, CO_ACTIVATED (weighted), CAUSES, PRECEDES
     (temporal), REFINES.
       - Weighted edges carry weight and last_reinforced_at.
     - Cold doc store (Mongo memories_cold): full memory docs older than 30 d.
     - Cold vector index (Qdrant mem_cold): embeddings for cold memories, kept for long-tail semantic recall.
     - Concept hierarchies (Neo4j Concept nodes with ABSTRACTION_OF edges to source memories): induced by the heavy consolidation cycle.
     - Temporal narratives (Episode nodes linked by PRECEDES edges): consolidated story arcs ("the spring-2026 thesis review cycle").

     Granularity. Per user_id, partitioned in Neo4j by :User {id: $user_id} root with all derived nodes attached.

     Update policy. Mostly written by the consolidation workers, not by the request path. Memory-node creation can happen synchronously on write (cheap), but edge induction
     (CO_ACTIVATED weights, CAUSES, ABSTRACTION_OF) is the consolidator's job.

     Backing Store — Immutable Event Log

     Cognitive role. Autobiographical trace. The ground-truth record of everything that happened. Never deleted. Used for replay during heavy consolidation, for audit, and for
     explainability.

     Contents.
     - raw_events collection in Mongo, append-only, with fields:
       - event_id (ULID, sortable)
       - user_id, agent_name, timestamp
       - event_type: user_message | agent_response | extraction_output | consolidation_event
       - payload: full message / response / extraction JSON
       - produced_memory_ids: list (for traceability)

     Update policy. Append-only, no updates, no deletes. Optional periodic export to JSONL on disk for offline replay.

     ---
     D. Retrieval Architecture (Multi-Stage Activation Cascade)

     Retrieval is no longer "embed query, top-k, return." It is a six-stage cascade in which memory is activated through progressive expansion, then synthesized.

     Stage 1: Seed activation         (cheap, parallel)
     Stage 2: Working-set expansion   (L1 only, instant)
     Stage 3: Spreading activation    (L3 graph, PPR)
     Stage 4: Causal/temporal traversal (L3 graph, directed walks)
     Stage 5: Filter + synthesis      (ACL, dedupe, rank)
     Stage 6: Reflective update       (writes back to L1 + co-activation log)

     Stage 1 — Seed Activation (parallel, ~20 ms)

     Three seed sources are computed in parallel:

     async def seed(query, agent, user_id):
         return await asyncio.gather(
             vector_seed(query, user_id, agent_visible_only=True, k=5),  # Qdrant warm
             entity_seed(query, user_id),                                  # NER → Neo4j entity match
             working_set_seed(agent, user_id),                             # L1 lookup
         )

     Each returns Dict[memory_id, seed_strength]. The three are merged by max.

     Stage 2 — Working-Set Expansion (~1 ms)

     def expand_working_set(seeds, agent_blackboard):
         expanded = dict(seeds)
         for mid, strength in seeds.items():
             # Pull in items already co-resident in this agent's working set
             for neighbor_id in agent_blackboard.recently_co_activated(mid):
                 expanded[neighbor_id] = max(expanded.get(neighbor_id, 0), 0.5 * strength)
         return expanded

     This implements priming: items already on the agent's mind get a head start.

     Stage 3 — Spreading Activation via Personalized PageRank (~50–150 ms)

     def spread_activation(seeds, user_id, agent, max_iter=8, alpha=0.15, eps=1e-3):
         # Personalized PageRank over Neo4j, restricted to agent-visible nodes.
         # alpha = restart probability (teleport back to seeds)
         activation = dict(seeds)
         for _ in range(max_iter):
             new_act = defaultdict(float)
             for node, val in activation.items():
                 if val < eps: continue
                 new_act[node] += alpha * val            # restart component
                 edges = neo4j_neighbors(node, user_id, agent_visibility=agent)
                 total_w = sum(e.weight for e in edges) or 1.0
                 for e in edges:
                     new_act[e.target] += (1 - alpha) * val * (e.weight / total_w)
             if converged(activation, new_act, eps): break
             activation = new_act
         return activation

     Edge weights come from CO_ACTIVATED strength (reinforced by consolidation), DEPENDS_ON (high, fixed), MENTIONS (medium), CONTRADICTS (negative — propagation actually suppresses the
      target), SUPERSEDES (zero — superseded memories are dampened).

     This is implemented in Cypher with APOC's apoc.algo.pageRankWithWeights or a Python-side Power-method over an extracted subgraph for tighter control.

     Stage 4 — Causal / Temporal Traversal (~30–100 ms)

     For the top-N activated nodes, traverse directed edges:
     - Backward along CAUSES, DERIVED_FROM to recover provenance / why.
     - Forward along PRECEDES to recover what happened next.
     - Outward along REFINES, SUPERSEDES to find the latest version.

     Cypher example:
     MATCH (seed:Memory {id: $mid})
     OPTIONAL MATCH path1 = (cause:Memory)-[:CAUSES*1..3]->(seed)
     OPTIONAL MATCH path2 = (seed)-[:PRECEDES*1..3]->(later:Memory)
     OPTIONAL MATCH (latest:Memory)-[:SUPERSEDES*0..]->(seed)
     RETURN seed, collect(DISTINCT cause), collect(DISTINCT later), latest

     Stage 5 — Filter + Synthesis (~10 ms)

     def synthesize(activated, agent, query):
         # ACL filter: only memories visible to this agent
         visible = [m for m in activated if acl_check(m, agent)]
         # Drop superseded
         visible = drop_superseded(visible)
         # Drop contradicted (unless query is asking about contradictions)
         visible = drop_contradicted(visible) if not query.asks_about_conflicts() else visible
         # Lineage dedupe: when an abstraction and its source memories both made it,
         # keep the abstraction unless the query needs specifics
         visible = lineage_dedupe(visible, prefer=abstraction_or_specifics(query))
         # Final rank
         ranked = sorted(
             visible,
             key=lambda m: (
                 m.activation
                 * m.confidence
                 * recency_decay(m.last_accessed_at)
             ),
             reverse=True,
         )
         return ranked[:top_n]

     Stage 6 — Reflective Update (~5 ms, async)

     After the agent generates its response, the cascade trace is fed back:

     async def reflect(cascade_trace, agent_blackboard):
         # Bump activation of every memory used
         for mid in cascade_trace.used_memory_ids:
             agent_blackboard.activate(mid, +0.3)
         # Log co-activations: every pair (a,b) used together gains weight
         pairs = combinations(cascade_trace.used_memory_ids, 2)
         co_activation_log.append({"pairs": pairs, "ts": now()})
         # If query revealed a contradiction, queue it for the consolidator
         if cascade_trace.contradictions_detected:
             consolidator_queue.put(ContradictionTask(cascade_trace.contradictions_detected))

     The co-activation log is what the light consolidation cycle uses to reinforce CO_ACTIVATED edges in Neo4j.

     End-to-end latency budget

     ┌─────────────────────────────┬───────────────────┬────────────┐
     │            Stage            │      Target       │ Worst case │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 1 Seed                      │ 20 ms             │ 80 ms      │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 2 Working set               │ 1 ms              │ 5 ms       │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 3 Spreading activation      │ 80 ms             │ 200 ms     │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 4 Causal traversal          │ 50 ms             │ 150 ms     │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 5 Synthesis                 │ 10 ms             │ 30 ms      │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ 6 Reflect (async, off-path) │ 0 (post-response) │ —          │
     ├─────────────────────────────┼───────────────────┼────────────┤
     │ Retrieval total             │ ~160 ms           │ ~465 ms    │
     └─────────────────────────────┴───────────────────┴────────────┘

     LLM generation time is unchanged from today (Ollama bound).

     ---
     E. Memory Write/Update Architecture

     Write pipeline (sync portion + async tail)

     async def write_memory(user_msg, user_id, agent):
         # 1. Backing store FIRST — immutable, sync, fast
         event_id = await raw_events.insert({
             "user_id": user_id,
             "agent_name": agent.name,
             "timestamp": now(),
             "event_type": "user_message",
             "payload": user_msg,
         })
         # 2. Off the request path: schedule extraction
         asyncio.create_task(_extract_and_persist(event_id, user_msg, user_id, agent))
         return event_id

     The async tail:

     async def _extract_and_persist(event_id, msg, user_id, agent):
         # 3. Extract: LLM JSON → rule-based → fast_payload (existing path, evolved)
         extraction = await extract(msg, agent)
         if not extraction and not agent.force_store: return
         if not extraction: extraction = fast_store_payload(msg)
         # 4. Resolve duplicates
         candidate = build_memory(extraction, user_id, agent, event_id)
         duplicate = await find_duplicate(candidate, user_id, sim=0.92)
         if duplicate:
             await reinforce(duplicate, candidate)  # bump evidence, increment confidence,
                                                     # add provenance link, do NOT create new
             return
         # 5. Mint memory_id, set lineage scaffold
         candidate.id = ulid()
         candidate.lineage = Lineage(
             source_event_id=event_id,
             derived_from=[],
             evidence=[event_id],
             version=1,
         )
         # 6. Triple-write: Mongo doc, Qdrant vector, Neo4j node + initial edges
         await asyncio.gather(
             mongo.memories_warm.insert(candidate.to_doc()),
             qdrant.upsert(collection="mem_warm", point=candidate.to_point()),
             neo4j_project(candidate),  # creates Memory node + AUTHORED_BY + MENTIONS edges
         )
         # 7. Update agent blackboard (seed activation in writer's L1)
         blackboard_for(user_id, agent).activate(candidate.id, +1.0)
         # 8. Queue consolidation candidates
         light_worker_queue.put(DedupeSweep(user_id, window="1h"))

     Memory schema (Mongo memories_warm / memories_cold)

     {
       "_id": "01HMZK...",                    // ULID, sortable by creation time
       "user_id": "user_abc",
       "version": 1,
       "tier": "warm",                         // warm | cold
       "intent": "task",                       // task|preference|fact|correction|goal|contextual_reference
       "memory_type": "Procedural",            // Semantic|Episodic|Procedural|Preference|Concept|Decision
       "summary": "…",
       "raw_text_event_id": "01HMZJ...",       // pointer into raw_events
       "entities": [{"name": "thesis", "kind": "topic"}, …],
       "embedding_id": "01HMZK...",            // matches Qdrant point id (== _id by convention)

       // Lineage / provenance
       "lineage": {
         "source_event_id": "01HMZJ...",
         "derived_from": [],                   // memory_ids this was derived from
         "supersedes":   [],                   // memory_ids this replaces
         "abstraction_of": [],                 // memory_ids this generalizes
         "evidence":      ["01HMZJ..."],       // event_ids supporting this memory
         "contradicted_by": [],                // memory_ids contradicting this
       },

       // Trust + value
       "confidence": 0.8,                      // updated on reinforcement
       "importance": 0.7,                      // from INTENT_DEFAULTS, may drift
       "novelty":    0.6,                      // 1 - max_sim_at_creation
       "utility":    null,                     // computed by light consolidator (co-activation centrality)

       // Activation (snapshotted for warm restart; live state in L1)
       "last_activation":   0.0,
       "last_accessed_at":  "2026-05-07T12:00:00Z",
       "access_count":      0,

       // Authorship + ACL
       "agent_owner": "project",
       "visibility":  "private",               // private | shared | public
       "shared_with": [],                      // list of agent names if visibility=shared

       // Timestamps
       "created_at": "...", "updated_at": "..."
     }

     Qdrant payload

     Mirrors a subset for filterable retrieval: user_id, agent_owner, visibility, shared_with, memory_type, intent, created_at (epoch ms). All filters native Qdrant payload indexes — no
      Python post-filter (fixes the current vector.py:70 post-filter bug).

     Neo4j projection (per memory)

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
     UNWIND $shared_with AS shared_agent
       MERGE (sa:Agent {name: shared_agent})
       MERGE (m)-[:SHARED_WITH]->(sa)

     CO_ACTIVATED, CAUSES, DEPENDS_ON, SUPERSEDES, CONTRADICTS, ABSTRACTION_OF edges are written by the consolidation workers, not at write time.

     Promotion / demotion

     ┌─────────────────┬──────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
     │   Transition    │                         Trigger                          │                                               Action                                               │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L2 → L1         │ retrieval cascade selects it                             │ load doc into agent blackboard; bump activation                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L1 → L1         │ re-used in subsequent turn                               │ bump activation; reset decay timer                                                                 │
     │ (refresh)       │                                                          │                                                                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L1 → (drop)     │ activation < ε                                           │ drop from working set; activation snapshot persisted to L2 doc                                     │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L2 → L3         │ age > 30 d AND activation < threshold for 7 d            │ move doc warm → cold collection; vector warm → cold; Neo4j node unchanged                          │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L3 → L2         │ retrieval pulls cold memory N times in 24 h              │ promote: copy doc back to warm, vector to mem_warm, mark tier: warm                                │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ L3 → Concept    │ heavy consolidator detects ≥3 similar memories with      │ induce Concept node; create ABSTRACTION_OF edges to source memories                                │
     │                 │ co-activation                                            │                                                                                                    │
     ├─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Any → Forgotten │ utility < 0.05 AND confidence < 0.3 AND age > 90 d       │ mark forgotten: true (do not delete; remove from L2 + Qdrant warm + cold; Neo4j node + edges       │
     │                 │                                                          │ retained for lineage)                                                                              │
     └─────────────────┴──────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

     ---
     F. Cache Replacement & Eviction Policies — Semantic Economics

     Eviction is not LRU. It is value-based, with multiple signals.

     Utility scoring

     def utility(memory):
         return (
             0.30 * recency_score(memory)            # exp(-(now - last_access)/τ_r)
           + 0.25 * frequency_score(memory)          # log(1 + access_count)
           + 0.20 * importance(memory)               # from INTENT_DEFAULTS, may drift
           + 0.15 * centrality(memory)               # PageRank in CO_ACTIVATED subgraph
           + 0.10 * lineage_value(memory)            # 1 if other memories derive from it, else 0
           - 0.20 * staleness_penalty(memory)        # contradicted, superseded, untouched-and-old
           - 0.30 * conflict_penalty(memory)         # contradictions outweigh evidence
         )

     utility is recomputed by the light consolidator hourly. It drives:
     - L2 → L3 demotion: utility < 0.4 AND age > 30 d.
     - L3 → forgotten: utility < 0.05 AND age > 90 d AND no recent retrieval.
     - L1 capacity eviction: when working set exceeds 50, drop lowest-utility item.

     Decay

     Activation decays exponentially in L1:

     λ = 1 / 600  # half-life ≈ 7 minutes
     def decay_tick(blackboard, dt):
         for mid in list(blackboard.activation_map):
             a = blackboard.activation_map[mid] * exp(-λ * dt)
             if a < 1e-3:
                 del blackboard.activation_map[mid]
                 blackboard.working_set.pop(mid, None)
             else:
                 blackboard.activation_map[mid] = a

     Edge decay in Neo4j (light consolidator, hourly):

     MATCH ()-[r:CO_ACTIVATED]->()
     WHERE r.last_reinforced_at < datetime() - duration('PT1H')
     SET r.weight = r.weight * 0.95
     WITH r WHERE r.weight < 0.05
     DELETE r

     Abstraction induction (heavy consolidator)

     def induce_abstractions(user_id):
         # 1. Find candidate clusters: memories with high pairwise CO_ACTIVATED weight
         #    that are NOT already abstracted.
         clusters = neo4j.query("""
             MATCH (m1:Memory)-[r:CO_ACTIVATED]-(m2:Memory)
             WHERE r.weight > 0.7 AND NOT (m1)<-[:ABSTRACTION_OF]-(:Concept)
             WITH m1, collect(DISTINCT m2) AS group
             WHERE size(group) >= 2
             RETURN m1, group
         """)
         for cluster in clusters:
             members = cluster.group + [cluster.m1]
             if avg_pairwise_similarity(members) < 0.75: continue   # incoherent cluster
             # 2. LLM call: induce a concept summary
             summary = await llm.abstract(members)
             confidence = lineage_confidence(members)
             # 3. Create Concept node + edges
             concept_id = await neo4j.create_concept(summary, confidence, source_ids=[m.id for m in members])
             # 4. Provenance
             log_event("consolidation", produced=concept_id, sources=[m.id for m in members])

     ---
     G. Relationship Modeling

     Node types

     ┌──────────┬────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
     │  Label   │            Purpose             │                            Key fields                            │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ User     │ partition root                 │ id                                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Agent    │ author / sharing target        │ name                                                             │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Memory   │ atomic memory object           │ id, type, intent, confidence, importance, visibility, created_at │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Entity   │ named thing referenced         │ name, kind (person/place/topic/artifact)                         │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Episode  │ bounded experience             │ id, time_window, narrative                                       │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Concept  │ induced abstraction            │ id, label, summary, induced_at, confidence                       │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Decision │ recorded choice                │ id, made_at, rationale                                           │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Goal     │ active objective               │ id, status, due_at                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Task     │ actionable item                │ id, status, due_at                                               │
     ├──────────┼────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
     │ Topic    │ broad theme (thesis, fitness…) │ name                                                             │
     └──────────┴────────────────────────────────┴──────────────────────────────────────────────────────────────────┘

     Edge types

     ┌────────────────┬──────────────────┬─────────────┬───────────────────────────────────────┐
     │      Type      │    Direction     │   Weight    │                Meaning                │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ OWNED_BY       │ Memory → User    │ —           │ partition                             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ AUTHORED_BY    │ Memory → Agent   │ —           │ who wrote it                          │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ SHARED_WITH    │ Memory → Agent   │ —           │ ACL grant                             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ MENTIONS       │ Memory → Entity  │ static      │ named-entity reference                │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ PART_OF        │ Memory → Episode │ —           │ episodic grouping                     │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ DEPENDS_ON     │ Memory → Memory  │ static high │ logical dependency                    │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ DERIVED_FROM   │ Memory → Memory  │ —           │ provenance                            │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ SUPERSEDES     │ Memory → Memory  │ —           │ replaces                              │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CONTRADICTS    │ Memory ↔ Memory  │ —           │ conflict marker                       │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ REFINES        │ Memory → Memory  │ —           │ non-replacing improvement             │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ ABSTRACTION_OF │ Concept → Memory │ —           │ concept generalizes memory            │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CAUSES         │ Memory → Memory  │ static      │ causal claim                          │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ PRECEDES       │ Memory → Memory  │ static      │ temporal order                        │
     ├────────────────┼──────────────────┼─────────────┼───────────────────────────────────────┤
     │ CO_ACTIVATED   │ Memory ↔ Memory  │ learned     │ reinforced by retrieval co-occurrence │
     └────────────────┴──────────────────┴─────────────┴───────────────────────────────────────┘

     CO_ACTIVATED is the only edge type the system learns automatically; everything else is asserted at write time, by the consolidator on explicit triggers, or by user feedback.

     Hyperedge / N-ary modeling

     Some relations are inherently n-ary: "in episode E, agent A made decision D supported by memories M1, M2, contradicted by M3." Represent these as first-class nodes:

     (:Decision {id, made_at})
        -[:IN_EPISODE]->(:Episode)
        -[:BY_AGENT]->(:Agent)
        -[:SUPPORTED_BY]->(:Memory)+
        -[:OPPOSED_BY]->(:Memory)*

     This is the "reified relationship" pattern — preserves the graph's queryability while modeling complex relations.

     ---
     H. Consistency & Access Control

     ACL model

     Every memory has:
     - agent_owner — the agent that wrote it.
     - visibility ∈ {private, shared, public}:
       - private (default): only agent_owner can read.
       - shared: read access granted to agents in shared_with list.
       - public: any agent for the same user_id may read.
     - shared_with: List[agent_name] — populated when visibility = shared.

     Sharing protocols

     Three ways a memory can become shared:

     1. Explicit share. An agent calls share_memory(memory_id, target_agents, reason). Logged in raw_events with the reason.
     2. Type-level policy. Certain memory types are auto-public for the user:
       - Preference memories (e.g., "I prefer mornings") — public by default.
       - Identity facts (name, role, location) — public.
       - Goal memories with cross_agent: true flag — public.
       - Procedural, Episodic, and most Semantic memories — private by default.
     3. Entity-bridged. When an entity is flagged as cross-agent (e.g., a person who appears in both project and school contexts), memories mentioning that entity become discoverable
     via SHARED_WITH propagation — but only the existence and summary are visible to non-owner agents, not the full content. The owner agent must approve full disclosure on first
     cross-agent retrieval.

     Coherence (MESI-inspired)

     When multiple agents have the same memory loaded into their L1 working sets:

     ┌───────────────┬──────────────────────────────────────────────────────────────────┐
     │     State     │                             Meaning                              │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ M (Modified)  │ one agent has loaded it AND is about to write a refinement       │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ E (Exclusive) │ one agent has loaded it; no others                               │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ S (Shared)    │ multiple agents have it loaded; all read-only                    │
     ├───────────────┼──────────────────────────────────────────────────────────────────┤
     │ I (Invalid)   │ memory was just superseded or contradicted; agents must re-fetch │
     └───────────────┴──────────────────────────────────────────────────────────────────┘

     The coherence_log collection tracks the state. On a write that modifies a Shared memory, all loaded copies transition to Invalid; the next time those agents touch the memory, they
     re-fetch from L2.

     For a single-user, single-process deployment this is mostly cosmetic — but the protocol is the same one we'd need under multi-worker, so we encode it now.

     Conflict resolution

     When extraction detects a new memory that contradicts an existing one:
     1. Both memories are kept. The new one gets a CONTRADICTS edge to the old.
     2. Confidence of both is recomputed: the more-recently-reinforced one's confidence rises slightly; the older one's drops.
     3. The next retrieval that surfaces either will see both and may trigger a reflective synthesis ("user previously said X, now says Y; treat Y as current unless asked").
     4. The heavy consolidator may eventually introduce a SUPERSEDES edge if the conflict resolves consistently over time.

     ---
     I. Concurrency & Performance

     Worker topology

                          ┌──────────────────────────────┐
                          │  FastAPI (asyncio event loop)│
                          │   - request handlers         │
                          │   - L1 blackboards (in-proc) │
                          │   - retrieval cascade        │
                          └──────┬───────────────────────┘
                                 │ submit
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
       ┌─────────┐         ┌─────────────┐    ┌──────────────────┐
       │ Extract │         │ Light       │    │ Heavy            │
       │ pool    │         │ consolidator│    │ consolidator     │
       │ (1 task │         │ (idle ≥60s) │    │ (cron 03:00 +    │
       │ per req)│         │             │    │  fallback ≥30min)│
       └─────────┘         └─────────────┘    └──────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                 ┌──────────────────────────────┐
                 │ Mongo · Qdrant · Neo4j · LLM │
                 └──────────────────────────────┘

     - Request handlers are async. Retrieval cascade fans out to Mongo/Qdrant/Neo4j with asyncio.gather.
     - Extract pool: one asyncio task per request for the LLM extraction tail. Cap concurrent in-flight extractions at 4 to avoid Ollama saturation.
     - Light consolidator: a single asyncio task on the same loop, woken by an idle detector (no requests for 60 s). Runs: edge re-weighting from co_activation_log, decay sweep, dedupe
     pass on memories created in the last hour, utility recompute. No LLM calls. Yields control on every iteration.
     - Heavy consolidator: APScheduler job at 03:00, plus an idle-fallback (>30 min idle). Runs in a separate process (consolidator.py) to avoid blocking the request event loop.
     Operations: cluster detection, abstraction induction (LLM), graph topology restructuring, episodic→semantic conversion, cold-tier migration.

     Locks and atomicity

     - Per-memory write lock: an asyncio.Lock keyed by memory_id, held during the triple-write (Mongo + Qdrant + Neo4j). Prevents concurrent reinforcement races.
     - Neo4j transactions: every write is one transaction; reads are auto-committed.
     - Activation updates (in-process): single-writer per blackboard. Since each blackboard is per-(user, agent) and we have one event loop, atomicity is automatic.

     Latency targets

     ┌────────────────────────────┬──────────────┬────────────────────────────────────────────────────┐
     │            Path            │    Target    │                        How                         │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L1 read                    │ < 100 µs     │ dict lookup                                        │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L2 vector + Mongo (warm)   │ < 50 ms      │ parallel gather                                    │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ L3 PPR (10k node subgraph) │ < 150 ms     │ bounded depth + edge-weight cutoff                 │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Full cascade               │ < 300 ms p95 │ budget per stage                                   │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Backing-store append       │ < 5 ms       │ unindexed insert, fire-and-forget on critical path │
     ├────────────────────────────┼──────────────┼────────────────────────────────────────────────────┤
     │ Async write tail           │ invisible    │ runs after response sent                           │
     └────────────────────────────┴──────────────┴────────────────────────────────────────────────────┘

     ---
     J. Customization for Your Use Case

     Agent registry (extends current app/agents/__init__.py)

     AGENTS = {
         # Health family (existing)
         "logger":       AgentSpec("logger",       "...", force_store=True,  default_visibility="private"),
         "nutritionist": AgentSpec("nutritionist", "...", force_store=False, default_visibility="private"),
         "trainer":      AgentSpec("trainer",      "...", force_store=False, default_visibility="private"),

         # Productivity family (new)
         "project":      AgentSpec("project",      "...", force_store=False, default_visibility="private"),
         "school":       AgentSpec("school",       "...", force_store=False, default_visibility="private"),
         "research":     AgentSpec("research",     "...", force_store=False, default_visibility="private"),
     }

     # Cross-agent sharing policy (new file: app/agents/sharing_policy.py)
     SHARING_POLICY = {
         # Auto-public memory types regardless of authoring agent
         "auto_public_types": {"Preference", "Identity"},
         # Cross-family entities (mentions trigger cross-family discoverability)
         "bridge_entities": ["calendar", "schedule", "deadline", "energy_level", "sleep"],
         # Family groupings (memories within a family see each other more readily)
         "families": {
             "health":       {"logger", "nutritionist", "trainer"},
             "productivity": {"project", "school", "research"},
         },
     }

     Worked example 1 — Project memory item

     User → project: "I'm migrating the API to FastAPI by June 12, blocked on auth refactor."

     Write flow:
     1. Append to raw_events (id EV_001).
     2. Extract → {intent: "task", memory_type: "Procedural", summary: "Migrate API to FastAPI", entities: [{name: "FastAPI", kind: "tool"}, {name: "auth refactor", kind: "topic"}],
     importance: 0.9, confidence: 0.8}.
     3. Resolve duplicates: vector search finds nothing similar. New memory M_001.
     4. Lineage: source_event_id=EV_001, evidence=[EV_001], version=1.
     5. Mongo memories_warm insert. Qdrant warm upsert. Neo4j: Memory{M_001} -[:AUTHORED_BY]-> Agent{project}, -[:MENTIONS]-> Entity{FastAPI}, -[:MENTIONS]-> Entity{auth refactor}.
     Visibility: private (project only).
     6. project blackboard: M_001 activation = 1.0; entity FastAPI weight = 0.9.
     7. Light consolidator queue: dedupe-sweep within last hour.

     Later — user → school: "Studying API design patterns for my distributed systems class."

     Retrieval flow:
     1. Seed: vector hits in school-visible memories (none yet); entity match Entity{API} overlaps with M_001's MENTIONS Entity{FastAPI} via Topic node Topic{API design}.
     2. Spread: PPR from Entity{API design} reaches M_001 — but stage 5 ACL filter drops it because M_001.visibility = private and M_001.agent_owner = project.
     3. school agent does not see project work. Isolation preserved.

     Cross-agent fetch (user explicitly asks): "What am I working on at work that overlaps with my distributed systems class?"

     The research agent (or whichever is asked) marks the query as cross-family. Stage 5 relaxes ACL to "summary visible, full content blurred." M_001's summary becomes available to the
      responder. Owner agent (project) is notified in the coherence_log.

     Worked example 2 — School memory item evolving into a Concept

     Over three weeks, user mentions to school:
     - "Reviewed Lamport's paper on Time, Clocks." → M_010
     - "Compared Lamport timestamps to vector clocks in HW3." → M_011
     - "Realized happens-before is a partial order, not total." → M_012
     - "TA confirmed: causal consistency uses partial order." → M_013

     Light consolidator (each idle window): notices co-activation between M_010..M_013 rises as the user keeps querying around them. CO_ACTIVATED edge weights climb.

     Heavy consolidator (next 03:00 cycle): cluster {M_010, M_011, M_012, M_013} exceeds the abstraction threshold. LLM call: "Generalize these four memories into a Concept." Output:

     Concept: "Causal ordering in distributed systems"
     Summary: "User has internalized that distributed-time mechanisms (Lamport timestamps,
     vector clocks) implement happens-before as a partial order, supporting causal but not
     total consistency. Reinforced by coursework and TA confirmation."
     Confidence: 0.85

     Concept node C_001 is created in Neo4j with ABSTRACTION_OF edges to all four source memories. Source memories are not deleted. Future retrieval that activates any of M_010..M_013
     also activates C_001 via the abstraction edge — and the synthesis stage prefers C_001 for general questions, drops back to specifics for detailed ones.

     Cross-agent benefit: if the project agent later asks "what do I know about distributed consistency?", it will reach C_001 (because Concepts authored by school agent default to
     type-public if memory_type=Concept — a setting in SHARING_POLICY).

     Selective private memory

     Health-family memories (logger/nutritionist/trainer) stay private to the health family by default — SHARING_POLICY.families ensures they don't bleed into productivity retrieval.
     The bridge_entities list (schedule, energy_level, sleep) is the only allowed crossover: a productivity agent planning study hours can ask about energy_level and reach the relevant
     logger memory through summary-only disclosure.

     Global persistent memory

     User-rooted Concept nodes form the persistent global memory. They're authored by no single agent (or by a synthetic consolidator agent), default to public, and are the canonical
     "what the user knows / believes / pursues over time." This is the layer most worth backing up and most worth surfacing to the user as a UI.

     ---
     K. Recommended Implementation Stack

     ┌─────────────────┬─────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐
     │     Concern     │                                      Tech                                       │                                    Reason                                    │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Web framework   │ FastAPI (async)                                                                 │ already chosen; native asyncio; fits cascade gather                          │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Doc store       │ MongoDB                                                                         │ already chosen; fine for warm/cold/raw_events                                │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Vector index    │ Qdrant                                                                          │ already chosen; payload filters fix current Python post-filter               │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Graph store     │ Neo4j 5.x + APOC                                                                │ locked decision; APOC for PageRank, similarity, batch ops                    │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ LLM             │ Ollama (local) — llama3.2:1b for extraction; consider larger model for heavy    │ local-first; matches current; abstraction calls happen overnight, latency    │
     │                 │ abstraction                                                                     │ irrelevant                                                                   │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Embeddings      │ all-MiniLM-L6-v2 (current) → consider bge-small-en-v1.5 for better recall       │ drop-in replacement; same dim or 384                                         │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Scheduler       │ APScheduler                                                                     │ tiny dep; cron + interval triggers in-process                                │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Activation      │ In-process Python (numpy + dict + heapq for top-k)                              │ locked decision                                                              │
     │ state           │                                                                                 │                                                                              │
     ├─────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
     │ Backing store   │ Mongo raw_events collection + optional JSONL export                             │ append-only, no new infra                                                    │
     └─────────────────┴─────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘

     No new languages. Pure Python. No new managed services. Everything Docker-local. Honors "local-first" and "vendor-agnostic" (Neo4j is open-source CE, Mongo CE, Qdrant CE, Ollama).
     Swap Neo4j for any property-graph DB by swapping the Cypher driver behind a GraphStore interface.

     Suggested package layout (informational, NOT the migration plan)

     app/
       cognition/                      ← new module
         blackboard.py                 ← per-agent L1 working memory
         activation.py                 ← decay, spreading-activation Python kernel
         cascade.py                    ← 6-stage retrieval cascade
         consolidator/
           light.py                    ← idle-triggered worker
           heavy.py                    ← scheduled worker (separate process)
           abstraction.py              ← LLM-driven concept induction
       graph/                          ← new module
         neo4j_client.py
         schema.py                     ← node labels + edge types as constants
         projections.py                ← memory→graph projection
         queries.py                    ← parametrized Cypher
       memory/
         summarizer.py                 ← evolved (lineage scaffolding)
         memory_updater.py             ← evolved (triple-write + ACL)
         vector.py                     ← evolved (Qdrant payload filters)
         backing.py                    ← raw_events accessors  (NEW)
       agents/
         __init__.py                   ← AgentSpec + AGENTS (extended)
         sharing_policy.py             ← SHARING_POLICY  (NEW)
     twin/
       orchestrator.py                 ← evolved (cascade-driven)
       prompt_builder.py               ← evolved (memory grouping by tier + lineage)

     ---
     L. Failure Modes & Safeguards

     ┌──────────────────────────────────────────┬───────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
     │                 Failure                  │        Symptom        │                                                  Safeguard                                                   │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Neo4j down at request time               │ Stage 3/4 fail        │ Cascade gracefully degrades to L1+L2 only. Log incident. Marker in response: "degraded_retrieval": true.     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Qdrant down                              │ Stage 1 vector seed   │ Fall back to entity-only seed + working-set seed.                                                            │
     │                                          │ empty                 │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Mongo down                               │ Doc fetches fail      │ 503 to client. No graceful path — Mongo is the truth store.                                                  │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Ollama down                              │ Abstraction induction │ Heavy consolidator skips abstraction; queues for next cycle. Light consolidator unaffected (no LLM).         │
     │                                          │  fails                │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Activation state lost on crash           │ Cold start            │ Reseed: replay last 1 h of raw_events + load top-utility 100 memories per agent into L1. Recovery time < 5   │
     │                                          │                       │ s.                                                                                                           │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Memory bloat (no forgetting)             │ L3 grows unbounded    │ Forgetting policy in §F. Manual purge tool: forget_below_utility(threshold).                                 │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Stale abstractions (concept no longer    │ Wrong retrievals      │ Heavy consolidator re-evaluates Concept→source coherence weekly; flags for refresh.                          │
     │ matches sources)                         │                       │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Conflict drift (contradictions           │ User confused by      │ Reflective synthesis at retrieval time presents both with timestamps; SUPERSEDES edge introduced when        │
     │ accumulate, no resolution)               │ mixed signals         │ conflict consistently resolves toward the newer side over 3+ retrievals.                                     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Over-sharing (private memory leaks)      │ ACL bug               │ ACL filter is a single chokepoint at Stage 5; covered by integration tests. Default-deny on missing          │
     │                                          │                       │ visibility field.                                                                                            │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Retrieval pollution (one agent's noise   │ Bad answers           │ Spreading activation is ACL-restricted at the Cypher level — non-visible nodes are not traversed at all, not │
     │ drowns another's signal)                 │                       │  just filtered post-hoc.                                                                                     │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Hot-loop activation positive feedback    │ Same memory always    │ Decay + edge-weight cap (weight ≤ 1.0) + diversity-aware reranker in Stage 5 (penalize too-similar items).   │
     │                                          │ wins                  │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Heavy consolidator runs during user      │ Latency spike         │ Idle detector + APScheduler coalesce=True, max_instances=1; heavy worker yields if it sees a request flag.   │
     │ activity                                 │                       │                                                                                                              │
     ├──────────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     │ Backing store unbounded growth           │ Disk fill             │ JSONL export + Mongo TTL after 1 year of inactivity (Concepts and lineage already preserve what matters from │
     │                                          │                       │  old raw events).                                                                                            │
     └──────────────────────────────────────────┴───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

     ---
     M. Final Blueprint

                      ┌────────────────────────────────────────────────┐
                      │                  USER MESSAGE                  │
                      └───────────────────────┬────────────────────────┘
                                              ▼
             ┌────────────────────────────────────────────────────────┐
             │  FastAPI request handler (asyncio)                     │
             │  • append to raw_events  (backing store)               │
             │  • spawn extract+persist task  (async tail)            │
             │  • run RETRIEVAL CASCADE                               │
             └───────────────────────┬────────────────────────────────┘
                                     ▼
        ┌────────── 6-STAGE RETRIEVAL CASCADE (ACL-aware throughout) ───────────┐
        │ 1 Seed (parallel):  vector  +  entity NER  +  agent's working set    │
        │ 2 Working-set expand (L1 priming)                                    │
        │ 3 Spreading activation: PPR over Neo4j with edge weights             │
        │ 4 Causal/temporal traversal: CAUSES, PRECEDES, SUPERSEDES walks      │
        │ 5 Filter + synthesis: ACL, lineage dedupe, rank by act·conf·recency  │
        │ 6 Reflect (post-response): bump activation, log co-activations       │
        └───────────────────────┬──────────────────────────────────────────────┘
                                ▼
             ┌────────────────────────────────────────────────────────┐
             │  Prompt builder · Ollama · response → user             │
             └───────────────────────┬────────────────────────────────┘
                                     ▼
             ┌────────────────────────────────────────────────────────┐
             │  Async write tail:                                     │
             │   extract → dedupe → triple-write (Mongo+Qdrant+Neo4j) │
             │   bump writer's L1 activation                          │
             │   queue light-consolidator tasks                       │
             └────────────────────────────────────────────────────────┘

                ┌─────────────────────────────────┐
                │   BACKGROUND COGNITION CYCLES   │
                │                                 │
                │  Light  (asyncio, idle ≥60 s):  │
                │   • edge re-weighting           │
                │   • activation decay            │
                │   • dedupe sweep (last hour)    │
                │   • utility recompute           │
                │                                 │
                │  Heavy  (cron 03:00 + ≥30 min): │
                │   • cluster detection           │
                │   • abstraction induction (LLM) │
                │   • episodic → semantic         │
                │   • L2 ↔ L3 migration           │
                │   • concept hierarchy           │
                └─────────────────────────────────┘

     Memory at a glance:
     L1  per-agent working memory blackboard  (in-process, volatile, 100 items)
             ↕ on retrieval / write
     L2  collaborative cognitive workspace    (Mongo warm + Qdrant warm + concept board, 30d window)
             ↕ on age / activation / consolidation
     L3  semantic cortex                      (Neo4j graph + Mongo cold + Qdrant cold)
             ↕ replay / audit only
     backing  immutable event log              (Mongo raw_events, append-only)

     Key invariants:
     - Every memory has an agent_owner and explicit visibility. Default is private.
     - Every memory has a lineage (source event, evidence, derived-from, supersedes, contradicted-by).
     - Every retrieval is a cascade, not a single query.
     - Every co-activation is learned: edges in the graph reflect what actually got used together.
     - Every consolidation is traceable: Concepts know their source memories; Episodes know their member events.
     - Activation is volatile by design: the system "wakes up" each restart and reseeds attention from cortex.

     ---
     Verification (how to know it works once implemented)

     End-to-end checks the future implementation must pass:

     1. Isolation. Write a memory via project with default visibility. Query through school. Assert the memory does not appear in the cascade output.
     2. Selective sharing. Mark a memory visibility=shared, shared_with=[research]. Query through research. Assert it appears. Through school. Assert it does not.
     3. Spreading activation. Insert two memories that share an entity but have low cosine similarity. Query with text matching only one. Assert the other surfaces via the graph.
     4. Abstraction induction. Synthesize 4+ co-activated memories on a topic. Run the heavy consolidator. Assert a Concept node is created with ABSTRACTION_OF edges to all four. Assert
      subsequent retrieval surfaces the Concept for general queries.
     5. Forgetting. Set utility/age thresholds. Run consolidator. Assert low-utility old memories are marked forgotten and disappear from L2 retrieval, while their Neo4j nodes remain
     for lineage.
     6. Warm restart. Kill and restart the FastAPI process during an active session. Assert that within 5 s, top-utility memories are reseeded into the relevant agent blackboards.
     7. Coherence. Two agents load the same shared memory. One refines it. Assert the other's next access fetches the new version (state transitions S → I → S).
     8. Latency. p95 cascade latency < 300 ms on a corpus of 50k memories, measured with pytest-bench or equivalent.
     9. Provenance. Every memory in the L3 cortex traces back to at least one raw_events entry. No orphans.
     10. Conflict survival. Insert two contradictory memories. Assert both are stored, a CONTRADICTS edge exists, and retrieval surfaces both with timestamps.

     When all ten pass, the substrate is doing what this document specifies.
