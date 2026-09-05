"""
Local chat web app for f1-fantasy: one chat box, routes each message to either
chains.py's plain Q&A chain or team_builder.py's LangGraph team-recommendation
flow, depending on what the message is asking for.

Stateless per message by design -- neither chains.py's RAG chain nor
team_builder.py's graph currently supports multi-turn conversational memory,
and adding that is out of scope here. The frontend keeps a visual conversation
history; each backend call is independent.

Run:
    python web/app.py
Then open http://localhost:5001 in a browser.
"""

import re
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from chains import build_rag_chain  # noqa: E402
from team_builder import run_team_builder  # noqa: E402

app = Flask(__name__)

# Team-building intent is detected by keyword rather than an extra LLM call --
# deliberately simple and deterministic for this MVP. A message matching any of
# these is routed to the (much slower: live web search + LLM) team_builder.py
# graph; everything else goes to the fast plain Q&A chain.
TEAM_BUILDING_KEYWORDS = re.compile(
    r"\b(build|recommend|captain|lineup|formation|roster|team\b.*\b(for|pick|build))\b",
    re.IGNORECASE,
)

_rag_chain = None


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain, _ = build_rag_chain(k=4)
    return _rag_chain


def is_team_building_request(message: str) -> bool:
    return bool(TEAM_BUILDING_KEYWORDS.search(message))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    try:
        if is_team_building_request(message):
            answer = run_team_builder(message)
            mode = "team_builder"
        else:
            answer = get_rag_chain().invoke(message)
            mode = "qa"
    except Exception as exc:  # noqa: BLE001 -- surface any backend error to the chat UI rather than a raw 500
        return jsonify({"error": str(exc)}), 500

    return jsonify({"answer": answer, "mode": mode})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
