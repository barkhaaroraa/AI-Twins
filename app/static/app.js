document.addEventListener("DOMContentLoaded", () => {

  const agentsUserIdEl   = document.getElementById("agentsUserId");
  const agentsRefreshBtn = document.getElementById("agentsRefresh");
  const agentsResetBtn   = document.getElementById("agentsResetMemory");
  const sharedMemListEl  = document.getElementById("sharedMemoryList");
  const sharedMemCountEl = document.getElementById("sharedMemCount");

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

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

      const usedCount = usedIds.length;
      const memSummary = usedCount
        ? `<span class="read-tag">read ${usedCount} shared memor${usedCount === 1 ? "y" : "ies"}</span> from the pool`
        : `no shared memories matched this query`;
      appendAgentMeta(agent, memSummary);

      pulseAgentBadge(agent, usedCount);

      // Memory write happens in a background thread — wait briefly so the
      // refresh actually reflects the new memory before highlighting it.
      setTimeout(() => refreshSharedMemory({ readIds: usedIds, readBy: agent }), 1500);
      // Initial refresh shows reads immediately.
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
    void badge.offsetWidth;
    badge.dataset.pulse = "1";
  }

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
      const srcLabel = m.source_agent ? m.source_agent : "system";
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
    ["logger", "nutritionist", "trainer"].forEach(a => {
      const wrap = document.getElementById(`chat-${a}`);
      if (wrap) wrap.innerHTML = `<p class="placeholder agent-card__hint">Empty memory — try the same question and compare.</p>`;
      const badge = document.getElementById(`memCount-${a}`);
      if (badge) badge.textContent = "0";
    });
    refreshSharedMemory();
  });

  refreshSharedMemory();
});
