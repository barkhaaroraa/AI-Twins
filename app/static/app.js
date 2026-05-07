document.addEventListener("DOMContentLoaded", () => {

  // ------------------------------------------------------------------
  // DOM refs
  // ------------------------------------------------------------------
  const historyEl       = document.getElementById("history");
  const chatForm        = document.getElementById("chatForm");
  const userIdInput     = document.getElementById("userId");
  const messageInput    = document.getElementById("message");
  const traceEl         = document.getElementById("retrievalTrace");
  const updatesEl       = document.getElementById("memoryUpdates");
  const graphContainer  = document.getElementById("graphContainer");
  const inspectorEl     = document.getElementById("inspector");
  const timelineEl      = document.getElementById("timelineContainer");
  const sessionDetailEl = document.getElementById("sessionDetail");
  const consolidationEl = document.getElementById("consolidationView");
  const topicsEl        = document.getElementById("topicsView");
  const consolidateBtn  = document.getElementById("runConsolidation");
  const consolidationStatsEl = document.getElementById("consolidationStats");
  const statsInfoEl     = document.getElementById("statsInfo");

  let visNetwork = null;

  // ------------------------------------------------------------------
  // Tab switching
  // ------------------------------------------------------------------
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById("tab-" + tab.dataset.tab);
      if (target) target.classList.add("active");

      if (tab.dataset.tab === "graph") loadGraph();
      if (tab.dataset.tab === "timeline") loadTimeline();
      if (tab.dataset.tab === "consolidation") loadTopics();
      if (tab.dataset.tab === "agents") refreshSharedMemory();
      refreshStats();
    });
  });

  // ------------------------------------------------------------------
  // Type colors
  // ------------------------------------------------------------------
  const TYPE_COLORS = {
    Semantic: "#57d9ff",
    Episodic: "#9a86ff",
    Procedural: "#ff9f57",
    Preference: "#57ff9a",
    consolidated: "#ffd700",
  };
  const EDGE_STYLES = {
    related_to: { color: "rgba(255,255,255,0.25)", dashes: false },
    derived_from: { color: "#57d9ff", dashes: [5, 5] },
    contradicts: { color: "#ff6b6b", dashes: false },
    temporal_next: { color: "#9a86ff", dashes: [2, 4] },
  };

  function getUserId() { return userIdInput.value.trim() || "user1"; }

  // ------------------------------------------------------------------
  // Chat
  // ------------------------------------------------------------------
  function appendMessage(role, text) {
    const first = historyEl.querySelector(".placeholder");
    if (first) first.remove();
    const el = document.createElement("div");
    el.className = "message " + role;
    el.innerHTML = `<strong>${role === "user" ? "You" : "AI Twin"}</strong><div>${escapeHtml(text)}</div>`;
    historyEl.appendChild(el);
    historyEl.scrollTop = historyEl.scrollHeight;
  }

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const userId = getUserId();
    const message = messageInput.value.trim();
    if (!userId || !message) return;

    appendMessage("user", message);
    messageInput.value = "";
    messageInput.focus();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message }),
      });
      const data = await res.json();
      appendMessage("ai", data.response || "No response.");
      renderTrace(data.retrieval_trace);

      // Show previous background results if available, otherwise show "processing"
      if (data.memory_updates && data.memory_updates.status === "processing") {
        updatesEl.innerHTML = '<p class="placeholder">Memory processing in background...</p>';
        pollForUpdates(userId);
      } else {
        renderUpdates(data.memory_updates);
      }
      refreshStats();
    } catch (err) {
      appendMessage("ai", "Error: " + err.message);
    }
  });

  // ------------------------------------------------------------------
  // Poll for background memory updates
  // ------------------------------------------------------------------
  function pollForUpdates(userId, attempt = 0) {
    if (attempt > 15) {
      updatesEl.innerHTML = '<p class="placeholder">Background processing timed out.</p>';
      return;
    }
    setTimeout(async () => {
      try {
        const res = await fetch(`/api/memory-updates/${userId}`);
        const data = await res.json();
        if (data.ready) {
          renderUpdates(data.updates);
          refreshStats();
        } else {
          pollForUpdates(userId, attempt + 1);
        }
      } catch { pollForUpdates(userId, attempt + 1); }
    }, 1500);
  }

  // ------------------------------------------------------------------
  // Retrieval Trace
  // ------------------------------------------------------------------
  function renderTrace(trace) {
    if (!trace || !trace.results || trace.results.length === 0) {
      traceEl.innerHTML = '<p class="placeholder">No memories retrieved.</p>';
      return;
    }
    let html = "";
    const intent = trace.query_intent || {};
    if (intent.intent) {
      html += `<div style="margin-bottom:0.5rem"><span class="badge badge--intent">${intent.intent}</span>`;
      if (intent.temporal_hint) html += ` <span class="badge badge--intent">${intent.temporal_hint}</span>`;
      html += `</div>`;
    }

    for (const r of trace.results) {
      const bd = r.score_breakdown || {};
      html += `<div class="trace-item">`;
      html += `<span class="badge badge--${r.memory_type || 'Semantic'}">${r.memory_type || 'Semantic'}</span>`;
      html += `<span class="trace-item__score">${r.final_score?.toFixed(3) || '?'}</span>`;
      html += `<div class="trace-item__text">${escapeHtml(r.text || '')}</div>`;
      html += `<div class="score-bars">`;
      html += scoreBar("semantic", bd.semantic);
      html += scoreBar("graph", bd.graph_rank);
      html += scoreBar("recency", bd.recency);
      html += scoreBar("importance", bd.importance);
      html += `</div>`;
      html += `<div style="display:flex;gap:8px;margin-top:4px">`;
      html += `<span class="score-bar__label">Semantic</span>`;
      html += `<span class="score-bar__label">Graph</span>`;
      html += `<span class="score-bar__label">Recency</span>`;
      html += `<span class="score-bar__label">Importance</span>`;
      html += `</div>`;
      if (r.explanation) {
        html += `<div class="trace-item__explanation">${escapeHtml(r.explanation)}</div>`;
      }
      html += `</div>`;
    }
    traceEl.innerHTML = html;
  }

  function scoreBar(type, value) {
    const pct = Math.round((value || 0) * 100);
    return `<div class="score-bar score-bar--${type}"><div class="score-bar__fill" style="width:${pct}%"></div></div>`;
  }

  // ------------------------------------------------------------------
  // Memory Updates
  // ------------------------------------------------------------------
  function renderUpdates(updates) {
    if (!updates) { updatesEl.innerHTML = '<p class="placeholder">No updates.</p>'; return; }
    let html = "";

    if (updates.extracted) {
      const e = updates.extracted;
      html += `<div class="update-item">`;
      html += `<span class="badge badge--${e.memory_type || 'Semantic'}">${e.memory_type || ''}</span> `;
      html += `<span class="badge badge--intent">${e.intent || ''}</span><br>`;
      html += `${escapeHtml(e.summary || '')}`;
      if (e.entities && e.entities.length) {
        html += `<div style="margin-top:4px">${e.entities.map(en => `<span class="entity-pill">${escapeHtml(en)}</span>`).join(' ')}</div>`;
      }
      html += `</div>`;
    }

    if (updates.graph_edges_added) {
      html += `<div class="update-item">${updates.graph_edges_added} graph edges created</div>`;
    }
    if (updates.contradictions && updates.contradictions.length) {
      for (const c of updates.contradictions) {
        html += `<div class="update-item update-item--contradiction">Contradiction: ${escapeHtml(c.explanation || c.text || '')}</div>`;
      }
    }
    if (updates.consolidation_triggered) {
      html += `<div class="update-item">Consolidation triggered</div>`;
    }

    updatesEl.innerHTML = html || '<p class="placeholder">No updates.</p>';
  }

  // ------------------------------------------------------------------
  // Graph (vis.js)
  // ------------------------------------------------------------------
  async function loadGraph() {
    const userId = getUserId();
    try {
      const res = await fetch(`/api/graph/${userId}`);
      const data = await res.json();
      renderGraph(data);
    } catch (err) {
      graphContainer.innerHTML = `<p class="placeholder">Error loading graph: ${err.message}</p>`;
    }
  }

  function ensureLegend() {
    if (graphContainer.querySelector(".graph-legend")) return;
    const legend = document.createElement("div");
    legend.className = "graph-legend";
    const items = [
      ["Semantic", TYPE_COLORS.Semantic],
      ["Episodic", TYPE_COLORS.Episodic],
      ["Procedural", TYPE_COLORS.Procedural],
      ["Preference", TYPE_COLORS.Preference],
      ["Consolidated", TYPE_COLORS.consolidated],
    ];
    legend.innerHTML = items
      .map(([n, c]) => `<span><span class="dot" style="background:${c}"></span>${n}</span>`)
      .join("");
    graphContainer.appendChild(legend);
  }

  function renderGraph(data) {
    graphContainer.innerHTML = "";
    if (!data.nodes || data.nodes.length === 0) {
      graphContainer.innerHTML = '<p class="placeholder" style="padding:2rem">No memory nodes yet. Chat to build the graph.</p>';
      return;
    }
    ensureLegend();

    const nodes = data.nodes.map(n => {
      const baseColor = TYPE_COLORS[n.type] || "#57d9ff";
      const truncated = (n.label || "").length > 36
        ? (n.label || "").slice(0, 36) + "…"
        : (n.label || "(no summary)");
      return {
        id: n.id,
        label: truncated,
        shape: "dot",
        size: 14 + (n.importance || 0.5) * 22,
        color: {
          background: baseColor,
          border: "#ffffff",
          highlight: { background: "#fff", border: baseColor },
          hover: { background: "#fff", border: baseColor },
        },
        borderWidth: 2,
        font: {
          color: "#ffffff",
          size: 13,
          face: "system-ui, sans-serif",
          strokeWidth: 3,
          strokeColor: "#0b0d16",
          vadjust: -2,
        },
        title: `${n.type}\n${n.label}\nConfidence: ${(n.confidence ?? 0).toFixed(2)}\nImportance: ${(n.importance ?? 0).toFixed(2)}`,
        _raw: n,
      };
    });

    const edges = data.edges.map((e, i) => {
      const style = EDGE_STYLES[e.relation] || EDGE_STYLES.related_to;
      const baseColor = (style.color || "rgba(255,255,255,0.45)").replace(/0\.\d+\)/, "0.85)");
      return {
        id: i,
        from: e.from,
        to: e.to,
        color: { color: baseColor, highlight: "#ffffff", hover: "#ffffff", opacity: 0.85 },
        dashes: style.dashes || false,
        arrows: e.relation === "temporal_next"
          ? { to: { enabled: true, scaleFactor: 0.6 } }
          : undefined,
        title: `${e.relation} (${(e.weight ?? 0).toFixed(2)})`,
        width: 1.5 + (e.weight || 0.5) * 2,
        smooth: { enabled: true, type: "dynamic" },
      };
    });

    const visData = {
      nodes: new vis.DataSet(nodes),
      edges: new vis.DataSet(edges),
    };

    const options = {
      autoResize: false,
      width: "100%",
      height: "620px",
      nodes: { shadow: { enabled: true, color: "rgba(0,0,0,0.4)", size: 8 } },
      edges: { selectionWidth: 2 },
      physics: {
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -55,
          centralGravity: 0.012,
          springLength: 160,
          springConstant: 0.08,
          damping: 0.5,
          avoidOverlap: 0.6,
        },
        stabilization: { enabled: true, iterations: 250, fit: true },
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        navigationButtons: true,
        keyboard: true,
        zoomView: true,
        dragView: true,
      },
      layout: { improvedLayout: true },
    };

    if (visNetwork) visNetwork.destroy();
    visNetwork = new vis.Network(graphContainer, visData, options);

    visNetwork.once("stabilizationIterationsDone", () => {
      visNetwork.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
      // Freeze physics after stabilizing so the graph doesn't drift.
      visNetwork.setOptions({ physics: { enabled: false } });
    });

    visNetwork.on("click", (params) => {
      if (params.nodes.length > 0) {
        loadInspector(params.nodes[0]);
      }
    });
  }

  async function loadInspector(memoryId) {
    try {
      const res = await fetch(`/api/memory/${memoryId}/details`);
      const data = await res.json();
      renderInspector(data);
    } catch (err) {
      inspectorEl.innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
    }
  }

  function renderInspector(data) {
    const m = data.memory || {};
    const connections = data.connections || [];

    let html = "";

    html += field("Summary", escapeHtml(m.summary || ""));
    html += field("Type", `<span class="badge badge--${m.memory_type || 'Semantic'}">${m.memory_type || ''}</span>`);
    html += field("Intent", `<span class="badge badge--intent">${m.intent || m.type || ''}</span>`);
    html += field("Confidence", (m.confidence || 0).toFixed(2));
    html += field("Importance", (m.importance || 0).toFixed(2));
    html += field("Created", m.created_at || "");

    if (m.entities && m.entities.length) {
      html += `<div class="inspector__field"><div class="inspector__label">Entities</div>`;
      html += `<div class="inspector__entities">${m.entities.map(e => `<span class="entity-pill">${escapeHtml(e)}</span>`).join('')}</div></div>`;
    }

    if (m.relationships && m.relationships.length) {
      html += `<div class="inspector__field"><div class="inspector__label">Relationships</div>`;
      for (const r of m.relationships) {
        html += `<div style="font-size:0.8rem">${escapeHtml(r.subject || '')} <strong>${escapeHtml(r.predicate || '')}</strong> ${escapeHtml(r.object || '')}</div>`;
      }
      html += `</div>`;
    }

    if (connections.length) {
      html += `<div class="inspector__field"><div class="inspector__label">Connections (${connections.length})</div>`;
      for (const c of connections) {
        html += `<div class="connection-item">`;
        html += `<span>${escapeHtml((c.content || '').substring(0, 50))}</span>`;
        html += `<span class="connection-item__relation">${c.relation} (${c.direction})</span>`;
        html += `</div>`;
      }
      html += `</div>`;
    }

    inspectorEl.innerHTML = html || '<p class="placeholder">No data.</p>';
  }

  function field(label, value) {
    return `<div class="inspector__field"><div class="inspector__label">${label}</div><div class="inspector__value">${value}</div></div>`;
  }

  // ------------------------------------------------------------------
  // Timeline
  // ------------------------------------------------------------------
  let currentFilter = "all";

  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.dataset.filter;
      loadTimeline();
    });
  });

  async function loadTimeline() {
    const userId = getUserId();
    try {
      const res = await fetch(`/api/timeline/${userId}`);
      const data = await res.json();
      renderTimeline(data.sessions || []);
    } catch (err) {
      timelineEl.innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
    }
  }

  function renderTimeline(sessions) {
    if (!sessions.length) {
      timelineEl.innerHTML = '<p class="placeholder">No sessions yet.</p>';
      return;
    }
    let html = "";
    for (const s of sessions) {
      let memories = s.memories || [];
      if (currentFilter !== "all") {
        memories = memories.filter(m => (m.memory_type || m.type) === currentFilter);
      }

      html += `<div class="timeline-session" data-session="${s.session_id}">`;
      html += `<div class="timeline-session__header">${formatDate(s.start_time)} — ${memories.length} memories ${s.is_active ? '(active)' : ''}</div>`;
      for (const m of memories) {
        const mtype = m.memory_type || m.type || "Semantic";
        html += `<div class="timeline-memory">`;
        html += `<span class="timeline-dot timeline-dot--${mtype}"></span>`;
        html += `<span>${escapeHtml(m.summary || '')}</span>`;
        html += `</div>`;
      }
      html += `</div>`;
    }
    timelineEl.innerHTML = html;

    // Click session for detail
    timelineEl.querySelectorAll(".timeline-session").forEach(el => {
      el.addEventListener("click", () => {
        const sid = el.dataset.session;
        const session = sessions.find(s => s.session_id === sid);
        if (session) renderSessionDetail(session);
      });
    });
  }

  function renderSessionDetail(session) {
    let html = field("Session ID", session.session_id);
    html += field("Start", formatDate(session.start_time));
    html += field("Last Activity", formatDate(session.last_activity));
    html += field("Status", session.is_active ? "Active" : "Closed");
    html += field("Memory Count", session.memory_count || 0);

    if (session.memories && session.memories.length) {
      html += `<div class="inspector__field"><div class="inspector__label">Memories</div>`;
      for (const m of session.memories) {
        const mtype = m.memory_type || m.type || "Semantic";
        html += `<div style="font-size:0.82rem;padding:0.25rem 0"><span class="badge badge--${mtype}">${mtype}</span> ${escapeHtml(m.summary || '')}</div>`;
      }
      html += `</div>`;
    }
    sessionDetailEl.innerHTML = html;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
  }

  // ------------------------------------------------------------------
  // Consolidation
  // ------------------------------------------------------------------
  consolidateBtn.addEventListener("click", async () => {
    const userId = getUserId();
    consolidateBtn.disabled = true;
    consolidateBtn.textContent = "Running...";
    try {
      const res = await fetch(`/api/consolidate/${userId}`, { method: "POST" });
      const data = await res.json();
      renderConsolidationStats(data);
      loadTopics();
    } catch (err) {
      consolidationStatsEl.innerHTML = `Error: ${err.message}`;
    }
    consolidateBtn.disabled = false;
    consolidateBtn.textContent = "Run Consolidation";
  });

  function renderConsolidationStats(data) {
    consolidationStatsEl.innerHTML = `
      <strong>Clusters:</strong> ${data.clusters_formed || 0} |
      <strong>Duplicates:</strong> ${data.duplicates_removed || 0} |
      <strong>Summaries:</strong> ${data.summaries_created || 0}
    `;
  }

  async function loadTopics() {
    const userId = getUserId();
    try {
      const res = await fetch(`/api/topics/${userId}`);
      const data = await res.json();
      renderTopics(data.topics || []);
    } catch (err) {
      topicsEl.innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
    }
  }

  function renderTopics(topics) {
    if (!topics.length) {
      topicsEl.innerHTML = '<p class="placeholder">Not enough memories for topic modeling.</p>';
      return;
    }
    let html = "";
    for (const t of topics) {
      html += `<div class="topic-item">`;
      html += `<strong>Topic ${t.topic_id + 1}</strong> (${t.memory_ids?.length || 0} memories)`;
      html += `<div class="topic-keywords">${t.keywords.map(k => `<span class="keyword-pill">${escapeHtml(k)}</span>`).join('')}</div>`;
      html += `</div>`;
    }
    topicsEl.innerHTML = html;
  }

  // ------------------------------------------------------------------
  // Stats
  // ------------------------------------------------------------------
  async function refreshStats() {
    const userId = getUserId();
    try {
      const res = await fetch(`/api/stats/${userId}`);
      const data = await res.json();
      statsInfoEl.textContent = `Nodes: ${data.graph_nodes || 0} | Edges: ${data.graph_edges || 0} | Sessions: ${data.sessions || 0} | Memories: ${data.total_memories || 0}`;
    } catch { /* ignore */ }
  }

  // Initial stats load
  refreshStats();

  // ------------------------------------------------------------------
  // Agents tab — 3 agents, shared memory pool
  // ------------------------------------------------------------------
  const agentsUserIdEl = document.getElementById("agentsUserId");
  const agentsRefreshBtn = document.getElementById("agentsRefresh");
  const agentsResetBtn = document.getElementById("agentsResetMemory");
  const sharedMemListEl = document.getElementById("sharedMemoryList");
  const sharedMemCountEl = document.getElementById("sharedMemCount");

  function getAgentsUserId() {
    return (agentsUserIdEl?.value || "demo_health").trim() || "demo_health";
  }

  function appendAgentMessage(agent, role, text, opts = {}) {
    const wrap = document.getElementById(`chat-${agent}`);
    if (!wrap) return null;
    const placeholder = wrap.querySelector(".placeholder");
    if (placeholder) placeholder.remove();

    const el = document.createElement("div");
    el.className = `agent-msg agent-msg--${role}`;
    if (opts.loading) {
      el.classList.add("agent-msg--loading");
      el.textContent = "Thinking…";
    } else {
      el.textContent = text;
    }
    wrap.appendChild(el);
    wrap.scrollTop = wrap.scrollHeight;
    return el;
  }

  function appendAgentMeta(agent, html) {
    const wrap = document.getElementById(`chat-${agent}`);
    if (!wrap) return;
    const el = document.createElement("div");
    el.className = "agent-msg agent-msg--meta";
    el.innerHTML = html;
    wrap.appendChild(el);
    wrap.scrollTop = wrap.scrollHeight;
  }

  async function sendToAgent(agent, message) {
    const userId = getAgentsUserId();
    const form = document.querySelector(`.agent-card__form[data-agent="${agent}"]`);
    const btn = form?.querySelector("button");
    const ta = form?.querySelector("textarea");
    if (btn) btn.disabled = true;

    appendAgentMessage(agent, "user", message);
    const loadingEl = appendAgentMessage(agent, "ai", "", { loading: true });

    try {
      const res = await fetch(`/agent/${agent}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message }),
      });
      const data = await res.json();

      if (loadingEl) loadingEl.remove();
      appendAgentMessage(agent, "ai", data.response || "(no response)");

      const usedIds = (data.memory_used || [])
        .map(m => m.memory_id)
        .filter(Boolean);

      // Meta line: how many memories were pulled into context
      const usedCount = usedIds.length;
      const memSummary = usedCount
        ? `<span class="read-tag">read ${usedCount} shared memor${usedCount === 1 ? "y" : "ies"}</span> from the pool`
        : `no shared memories matched this query`;
      appendAgentMeta(agent, memSummary);

      // Update count badge for this agent
      pulseAgentBadge(agent, usedCount);

      // Refresh shared memory feed and highlight reads/writes
      await refreshSharedMemory({ readIds: usedIds, readBy: agent });
    } catch (err) {
      if (loadingEl) loadingEl.remove();
      appendAgentMessage(agent, "ai", "Error: " + err.message);
    } finally {
      if (btn) btn.disabled = false;
      if (ta) { ta.value = ""; ta.focus(); }
    }
  }

  function pulseAgentBadge(agent, count) {
    const badge = document.getElementById(`memCount-${agent}`);
    if (!badge) return;
    badge.textContent = count;
    badge.dataset.pulse = "0";
    // restart animation
    void badge.offsetWidth;
    badge.dataset.pulse = "1";
  }

  // Wire each agent form
  document.querySelectorAll(".agent-card__form").forEach(form => {
    const agent = form.dataset.agent;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const ta = form.querySelector("textarea");
      const msg = (ta?.value || "").trim();
      if (!msg) return;
      sendToAgent(agent, msg);
    });
  });

  let lastKnownIds = new Set();

  async function refreshSharedMemory(opts = {}) {
    const { readIds = [], readBy = null } = opts;
    const userId = getAgentsUserId();
    if (!sharedMemListEl) return;

    let memories = [];
    try {
      const res = await fetch(`/api/agent-memories/${userId}`);
      const data = await res.json();
      memories = data.memories || [];
    } catch (err) {
      sharedMemListEl.innerHTML = `<p class="placeholder">Error loading memory: ${err.message}</p>`;
      return;
    }

    if (!memories.length) {
      sharedMemListEl.innerHTML = '<p class="placeholder">No memories yet. Log something via the Logger to populate the shared pool.</p>';
      sharedMemCountEl.textContent = "0";
      lastKnownIds = new Set();
      return;
    }

    const newIds = new Set(memories.map(m => m.id));
    const justWritten = [...newIds].filter(id => !lastKnownIds.has(id));
    sharedMemCountEl.textContent = String(memories.length);

    sharedMemListEl.innerHTML = memories.map(m => {
      const src = m.source_agent || "null";
      const justNew = justWritten.includes(m.id) ? "mem-pill--just-written" : "";
      const justRead = readIds.includes(m.id) ? "mem-pill--just-read" : "";
      const readBadge = readIds.includes(m.id) && readBy
        ? `<span class="mem-pill__read-by">read by ${readBy}</span>`
        : "";
      const ents = (m.entities || []).slice(0, 5).map(
        e => `<span class="entity-pill">${escapeHtml(e)}</span>`
      ).join(" ");
      const srcLabel = m.source_agent
        ? m.source_agent
        : "system";
      return `
        <div class="mem-pill ${justNew} ${justRead}" data-source="${src}" data-id="${m.id}">
          ${readBadge}
          <div class="mem-pill__src">${escapeHtml(srcLabel)}<span class="mem-pill__type">· ${escapeHtml(m.memory_type || "Semantic")}</span></div>
          <div>${escapeHtml(m.summary || "")}</div>
          ${ents ? `<div class="mem-pill__entities">${ents}</div>` : ""}
        </div>
      `;
    }).join("");

    lastKnownIds = newIds;
  }

  agentsRefreshBtn?.addEventListener("click", () => refreshSharedMemory());
  agentsResetBtn?.addEventListener("click", () => {
    const fresh = `demo_fresh_${Math.floor(Math.random() * 9000) + 1000}`;
    agentsUserIdEl.value = fresh;
    ["logger","nutritionist","trainer"].forEach(a => {
      const wrap = document.getElementById(`chat-${a}`);
      if (wrap) wrap.innerHTML = `<p class="placeholder agent-card__hint">Empty memory — try the same question and compare.</p>`;
      const badge = document.getElementById(`memCount-${a}`);
      if (badge) badge.textContent = "0";
    });
    refreshSharedMemory();
  });
});
