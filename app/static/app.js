document.addEventListener("DOMContentLoaded", () => {

  const historyEl = document.getElementById("history");
  const memoryListEl = document.getElementById("memoryList");
  const chatForm = document.getElementById("chatForm");
  const userIdInput = document.getElementById("userId");
  const messageInput = document.getElementById("message");

  // Safety check
  if (!historyEl || !memoryListEl || !chatForm || !userIdInput || !messageInput) {
    console.error("Some DOM elements are missing.");
    return;
  }


  // -----------------------------
  // Append message to chat window
  // -----------------------------
  function appendMessage(role, text) {

    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;

    const label = document.createElement("strong");
    label.textContent = role === "user" ? "You" : "AI Twin";

    const content = document.createElement("div");
    content.textContent = text;

    messageEl.appendChild(label);
    messageEl.appendChild(content);

    historyEl.appendChild(messageEl);
    historyEl.scrollTop = historyEl.scrollHeight;
  }


  // -----------------------------
  // Update memory panel
  // -----------------------------
  function updateMemory(memories) {

    memoryListEl.innerHTML = "";

    if (!memories || memories.length === 0) {
      memoryListEl.innerHTML =
        `<p class="placeholder">No stored memory yet.</p>`;
      return;
    }

    memories.forEach(memory => {

      const item = document.createElement("div");
      item.className = "memory__item";

      const text = document.createElement("div");
      text.textContent = memory.text;

      const score = document.createElement("small");
      score.textContent = `similarity: ${memory.similarity_score}`;

      item.appendChild(text);
      item.appendChild(score);

      memoryListEl.appendChild(item);
    });
  }


  // -----------------------------
  // Send message to backend
  // -----------------------------
  async function sendMessage(userId, message) {

    try {

      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          user_id: userId,
          message: message
        })
      });

      if (!response.ok) {
        throw new Error(`Server error ${response.status}`);
      }

      const data = await response.json();

      appendMessage("ai", data.response);
      updateMemory(data.memory_used || []);

    } catch (error) {

      appendMessage("ai", `Error: ${error.message}`);

    }
  }


  // -----------------------------
  // Form submit handler
  // -----------------------------
  chatForm.addEventListener("submit", async (event) => {

    event.preventDefault();

    const userId = userIdInput.value.trim();
    const message = messageInput.value.trim();

    if (!userId || !message) return;

    appendMessage("user", message);

    messageInput.value = "";
    messageInput.focus();

    await sendMessage(userId, message);

  });

});