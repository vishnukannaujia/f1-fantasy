const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendButtonEl = document.getElementById("send-button");

function addMessage(role, text, mode) {
  const el = document.createElement("div");
  el.className = `message ${role}`;

  if (mode) {
    const badge = document.createElement("span");
    badge.className = `mode-badge ${mode}`;
    badge.textContent = mode === "team_builder" ? "Team Builder" : "Q&A";
    el.appendChild(badge);
    el.appendChild(document.createElement("br"));
  }

  el.appendChild(document.createTextNode(text));
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", message);
  inputEl.value = "";
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  const thinkingEl = addMessage(
    "assistant thinking",
    "Thinking... (team-building requests take longer -- they check live weather/news first)"
  );

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();

    thinkingEl.remove();
    if (!response.ok) {
      addMessage("error", data.error || "Something went wrong.");
    } else {
      addMessage("assistant", data.answer, data.mode);
    }
  } catch (err) {
    thinkingEl.remove();
    addMessage("error", "Could not reach the server: " + err.message);
  } finally {
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  }
});
