"""
LangGraph team-optimization flow: given current price/standings/form/circuit data
plus live weather/incident info, propose a full F1 Fantasy roster (5 drivers + 2
constructors) within the $100M budget cap, with a Captain/DRS Boost pick and
reasoning grounded in real current data.

Why this needs its own graph instead of reusing chains.py's plain RAG chain:
budget-constrained team selection needs the ENTIRE price table simultaneously (not
a similarity-ranked top-k subset of chunks), and LLMs are unreliable at exact
arithmetic over many line items -- so validate_budget is deterministic Python, and
the graph retries propose_team with corrective feedback if the LLM's proposal goes
over budget or gets the roster shape wrong. See ARCHITECTURE.md for the diagram.

Live weather/news use Anthropic's own hosted web_search server tool (via the raw
anthropic SDK), NOT the WebSearch tool used to build this project -- that tool only
exists inside the Claude Code session that wrote this code, a standalone script
has no access to it. This is the one piece of real live data this script can fetch
on its own when you run it.

Usage:
    python src/ingest.py                          # once, or after prices update
    python src/team_builder.py "your question"     # e.g. "Build me a team for Monza"
"""

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List, Optional, TypedDict

import anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langfuse import observe
from langgraph.graph import END, StateGraph
from langsmith import traceable

from observability import langfuse_callbacks, langfuse_session, langsmith_session_metadata

from f1_data import parse_constructor_prices, parse_driver_prices

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
BUDGET_CAP = 100.0
MAX_RETRIES = 2

# Update these each race weekend, alongside the corpus refresh (see ARCHITECTURE.md).
RACE_NAME = "2026 Spanish Grand Prix"
RACE_LOCATION = "Madrid, Spain (Madring circuit)"
RACE_DATE = "September 13, 2026"


def extract_text(content) -> str:
    """Same normalization as chains.py: content is a str, or a list of blocks
    (e.g. thinking + text) depending on whether the model reasoned first."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def get_llm():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # This prompt is long and structurally demanding enough (budget-constrained
    # selection across 22 drivers + 11 constructors) that the model sometimes
    # spends its whole token budget on extended thinking before writing any
    # visible answer -- observed hitting stop_reason="max_tokens" with
    # thinking_tokens=2047/2048, producing a truncated, EMPTY response. Give it
    # enough headroom for thinking AND the structured answer.
    return ChatAnthropic(model=MODEL_NAME, max_tokens=8192)


class GraphState(TypedDict):
    question: str
    static_context: str
    live_conditions: str
    driver_prices: dict
    constructor_prices: dict
    proposal_text: str
    drivers: List[str]
    constructors: List[str]
    captain: Optional[str]
    reasoning: str
    total_cost: float
    validation_errors: List[str]
    retries: int
    final_team: str


def _read(name: str) -> str:
    return (DATA_DIR / name).read_text()


LEARNINGS_PATH = ROOT / "learnings" / "learnings.json"


def render_learnings(learnings: list) -> str:
    """Pure rendering/weighting logic, split out from load_learnings_text() so
    it's unit-testable without touching the filesystem -- given a list of
    learning dicts (the parsed contents of learnings.json), not a path.
    Assumes `learnings` is non-empty; the empty/missing-file cases are handled
    by the caller."""
    # Weight by evidence, not equally -- with only a handful of races scored so
    # far, treating every learning as an unconditional rule risks overfitting to
    # single-race noise. Recency gets a mild nudge (the competitive order shifts
    # over a season), never an automatic override of a more-confirmed pattern.
    lines = [
        "LESSONS FROM PAST RACES -- weight these by their evidence strength, NOT equally:",
        "- A lesson tagged 'preliminary (n=1)' is a hypothesis from a single race -- weigh it "
        "alongside your own reasoning about this race's specific facts, don't follow it blindly.",
        "- A lesson tagged 'confirmed (n>=2)' has held up across multiple independent races and "
        "should be weighted more heavily.",
        "- More recent lessons get slightly more weight than older ones (the season's competitive "
        "order shifts), but a single recent observation does NOT override an established, "
        "multiply-confirmed pattern outright.",
        "- Where a lesson is marked as refining an earlier one, treat it as narrowing/qualifying "
        "that earlier lesson to a specific condition, NOT replacing or contradicting it -- both "
        "still apply, in their respective circumstances.",
        "",
    ]
    # Oldest first: later lessons appearing later in the prompt gives them a
    # mild recency-favoring position, consistent with the "slightly more
    # weight" instruction above, without making it an override.
    for entry in sorted(learnings, key=lambda e: e["added"]):
        tag = f"{entry.get('confidence', 'preliminary')} (n={entry.get('race_count', 1)}), added {entry['added']}"
        if entry.get("refines"):
            tag += f", refines lesson #{entry['refines']}"
        lines.append(f"- [{tag}] {entry['lesson']}")
    return "\n".join(lines)


def load_learnings_text() -> str:
    """Learnings accumulated by eval/learnings_loop.py from real scored races
    -- this is the actual feedback loop: a new lesson appended to
    learnings.json is picked up here automatically, on the very next call,
    with no code change. Contrast with how the very first lesson (from the
    Zandvoort backtest) got into the system: hand-edited directly into this
    file's prompt string. That approach doesn't scale past one lesson."""
    if not LEARNINGS_PATH.exists():
        return ""
    learnings = json.loads(LEARNINGS_PATH.read_text())
    if not learnings:
        return ""
    return render_learnings(learnings)


def load_static_context(state: GraphState) -> GraphState:
    driver_prices = parse_driver_prices(DATA_DIR / "02_driver_prices.txt")
    constructor_prices = parse_constructor_prices(DATA_DIR / "03_constructor_prices.txt")

    driver_lines = [f"{name} ({info['team']}): ${info['price']}M" for name, info in driver_prices.items()]
    constructor_lines = [f"{team}: ${price}M" for team, price in constructor_prices.items()]

    context = "\n\n".join(
        part
        for part in [
            _read("01_fantasy_scoring_rules.txt"),
            "ALL DRIVER PRICES (use these exact names):\n" + "\n".join(driver_lines),
            "ALL CONSTRUCTOR PRICES (use these exact names):\n" + "\n".join(constructor_lines),
            _read("04_championship_standings.txt"),
            _read("05_recent_form_and_last_race.txt"),
            _read("07_madrid_circuit_notes.txt"),
            load_learnings_text(),
        ]
        if part
    )
    print(f"  [load_static_context] {len(driver_prices)} drivers, {len(constructor_prices)} constructors loaded")
    return {
        **state,
        "static_context": context,
        "driver_prices": driver_prices,
        "constructor_prices": constructor_prices,
    }


NO_LIVE_CONDITIONS_MESSAGE = (
    "No live weather/news data available for this run (the web_search call failed or returned "
    "nothing) -- reason about the race using only the static context above, and note in your "
    "REASONING that live conditions could not be checked this time."
)


@traceable(name="fetch_live_conditions (raw anthropic SDK, not auto-traced by LangChain)")
@observe(name="fetch_live_conditions")
def fetch_live_conditions(state: GraphState) -> GraphState:
    """Live web search is a genuinely flaky dependency (network errors, rate limits,
    timeouts) -- unlike the deterministic nodes elsewhere in this graph, a failure
    here should degrade the run (proceed on static context alone) rather than crash
    the whole team-building request. See ARCHITECTURE.md known gaps for why this
    didn't exist until now."""
    client = anthropic.Anthropic()
    query = (
        f"Search the web for two things about the {RACE_NAME} at {RACE_LOCATION} on {RACE_DATE}: "
        f"(1) the current weather forecast for race day, and (2) any breaking F1 news from the last "
        f"few days (practice session incidents, penalties, grid changes, mechanical issues) that "
        f"could affect this race weekend. Summarize both concisely."
    )
    try:
        resp = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": query}],
            timeout=30.0,
        )
    except anthropic.APIError as exc:
        print(f"  [fetch_live_conditions] web_search call failed ({exc.__class__.__name__}: {exc}) -- proceeding without live conditions")
        return {**state, "live_conditions": NO_LIVE_CONDITIONS_MESSAGE}

    live_text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    if not live_text.strip():
        print("  [fetch_live_conditions] web_search returned no usable text -- proceeding without live conditions")
        return {**state, "live_conditions": NO_LIVE_CONDITIONS_MESSAGE}

    print(f"  [fetch_live_conditions] retrieved {len(live_text)} chars of live weather/news context")
    return {**state, "live_conditions": live_text}


PROPOSAL_INSTRUCTIONS = """
You are an expert F1 Fantasy strategist. Using ONLY the context provided above \
(official scoring rules, current driver/constructor prices, standings, recent \
form, circuit notes, and live weather/news), propose an optimal F1 Fantasy \
roster for the upcoming race.

Requirements:
- Exactly 5 drivers and 2 constructors.
- Total cost must be $100M or less. This will be checked EXACTLY against the \
real price list, so use driver/constructor names EXACTLY as they appear in the \
ALL DRIVER PRICES / ALL CONSTRUCTOR PRICES lists above -- no extra text, no \
team name appended to a driver's name.
- Pick one of your 5 drivers as Captain (gets the DRS Boost -- doubled score).
- Pay close attention to any "LESSONS FROM PAST RACES" section in the context above --
those are specific, evidence-backed corrections derived from scoring this exact
reasoning process against real race results. They exist because this same process got
something wrong before; don't repeat it.

Reply in EXACTLY this format, with no extra commentary before or after:

DRIVERS:
1. <exact driver name>
2. <exact driver name>
3. <exact driver name>
4. <exact driver name>
5. <exact driver name>
CONSTRUCTORS:
1. <exact constructor name>
2. <exact constructor name>
CAPTAIN: <one of the 5 driver names above>
REASONING: <2-4 sentences explaining the picks, referencing current form, circuit \
fit, and value for money>
"""


def propose_team(state: GraphState) -> GraphState:
    llm = get_llm()
    retries = state.get("retries", 0)
    feedback = ""
    if state.get("validation_errors"):
        retries += 1
        feedback = (
            "\n\nYour PREVIOUS proposal was INVALID for these reasons:\n"
            + "\n".join(f"- {e}" for e in state["validation_errors"])
            + "\nFix these issues in your new proposal."
        )
    prompt = (
        f"{state['static_context']}\n\nLIVE CONDITIONS:\n{state['live_conditions']}\n\n"
        f"{PROPOSAL_INSTRUCTIONS}{feedback}\n\nQuestion: {state['question']}"
    )
    text = extract_text(llm.invoke(prompt).content)
    print(f"  [propose_team] attempt {retries + 1}")
    return {**state, "proposal_text": text, "retries": retries}


def _parse_proposal(text: str):
    section = None
    drivers, constructors, captain, reasoning_lines = [], [], None, []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("DRIVERS:"):
            section = "drivers"
            continue
        if upper.startswith("CONSTRUCTORS:"):
            section = "constructors"
            continue
        if upper.startswith("CAPTAIN:"):
            section = None
            captain = stripped.split(":", 1)[1].strip()
            continue
        if upper.startswith("REASONING:"):
            section = "reasoning"
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                reasoning_lines.append(rest)
            continue
        if section == "reasoning" and stripped:
            reasoning_lines.append(stripped)
            continue
        m = re.match(r"^\d+\.\s*(.+)$", stripped)
        if m and section == "drivers":
            drivers.append(m.group(1).strip())
        elif m and section == "constructors":
            constructors.append(m.group(1).strip())
    return drivers, constructors, captain, " ".join(reasoning_lines)


def _lookup(name: str, table: dict):
    if name in table:
        return name, table[name]
    for key in table:
        if key.lower() == name.lower():
            return key, table[key]
    return None, None


def validate_budget(state: GraphState) -> GraphState:
    drivers, constructors, captain, reasoning = _parse_proposal(state["proposal_text"])
    errors = []
    total = 0.0

    resolved_drivers = []
    for name in drivers:
        key, info = _lookup(name, state["driver_prices"])
        if key is None:
            errors.append(f"Driver '{name}' not found in the current price list -- use an exact name from ALL DRIVER PRICES.")
            continue
        resolved_drivers.append(key)
        total += info["price"]

    resolved_constructors = []
    for name in constructors:
        key, price = _lookup(name, state["constructor_prices"])
        if key is None:
            errors.append(f"Constructor '{name}' not found in the current price list -- use an exact name from ALL CONSTRUCTOR PRICES.")
            continue
        resolved_constructors.append(key)
        total += price

    if len(drivers) != 5:
        errors.append(f"Must pick exactly 5 drivers, got {len(drivers)}.")
    if len(constructors) != 2:
        errors.append(f"Must pick exactly 2 constructors, got {len(constructors)}.")
    if total > BUDGET_CAP:
        errors.append(f"Total cost ${total:.1f}M exceeds the ${BUDGET_CAP:.0f}M budget cap by ${total - BUDGET_CAP:.1f}M.")

    if captain:
        match = next((d for d in resolved_drivers if d.lower() == captain.lower()), None)
        if match:
            captain = match
        else:
            errors.append(f"Captain '{captain}' must be one of the 5 chosen drivers.")
    else:
        errors.append("No Captain specified.")

    print(
        f"  [validate_budget] drivers={len(resolved_drivers)} constructors={len(resolved_constructors)} "
        f"total=${total:.1f}M errors={len(errors)}"
    )

    return {
        **state,
        "drivers": resolved_drivers,
        "constructors": resolved_constructors,
        "captain": captain,
        "reasoning": reasoning,
        "total_cost": total,
        "validation_errors": errors,
    }


def route_after_validation(state: GraphState) -> str:
    if not state["validation_errors"]:
        return "finalize"
    if state["retries"] >= MAX_RETRIES:
        return "finalize"
    return "retry"


def finalize(state: GraphState) -> GraphState:
    header = ""
    if state["validation_errors"]:
        header = (
            f"WARNING: could not produce a fully valid team after {state['retries']} retries. "
            f"Remaining issues: {'; '.join(state['validation_errors'])}\n\n"
        )
    remaining = BUDGET_CAP - state["total_cost"]
    lines = [
        header + "RECOMMENDED TEAM:",
        "Drivers: " + ", ".join(state["drivers"]),
        "Constructors: " + ", ".join(state["constructors"]),
        f"Captain (DRS Boost): {state['captain']}",
        f"Total cost: ${state['total_cost']:.1f}M / ${BUDGET_CAP:.0f}M budget (${remaining:.1f}M remaining)",
        "",
        "Reasoning: " + state["reasoning"],
    ]
    final_team = "\n".join(lines)
    print("  [finalize] done")
    return {**state, "final_team": final_team}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("load_static_context", load_static_context)
    graph.add_node("fetch_live_conditions", fetch_live_conditions)
    graph.add_node("propose_team", propose_team)
    graph.add_node("validate_budget", validate_budget)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("load_static_context")
    graph.add_edge("load_static_context", "fetch_live_conditions")
    graph.add_edge("fetch_live_conditions", "propose_team")
    graph.add_edge("propose_team", "validate_budget")
    graph.add_conditional_edges(
        "validate_budget",
        route_after_validation,
        {"retry": "propose_team", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_team_builder_full(question: str) -> dict:
    """Returns the full graph result (drivers, constructors, captain, cost,
    reasoning, live_conditions, etc.), not just the formatted string -- callers
    that need structured data (e.g. eval/predictions_tracker.py, to persist a
    prediction record for later scoring against the real result) should use
    this instead of run_team_builder."""
    app = build_graph()
    # A shared session_id ties the LangSmith trace and the Langfuse trace for
    # THIS call together as the same logical run, even amid other concurrent
    # calls -- the two systems don't share an ID space on their own.
    # run_id additionally pins the exact LangSmith trace ID up front, so it's
    # known before the call finishes rather than searched for afterward.
    session_id = str(uuid.uuid4())
    run_id = uuid.uuid4()
    # LangGraph's compiled app.invoke() is already auto-traced by LangSmith
    # (it's built on LangChain's Runnable interface) whenever LANGSMITH_TRACING
    # is set -- no @traceable wrapper needed here, just a readable root name
    # instead of the generic default so traces are findable by question asked.
    # Langfuse needs the explicit callback attached, added here alongside it.
    config = {
        "run_name": f"team_builder: {question[:60]}",
        "run_id": run_id,
        "callbacks": langfuse_callbacks(),
        "metadata": langsmith_session_metadata(session_id),
    }
    print(f"  [tracing] session_id={session_id} langsmith_run_id={run_id}")
    with langfuse_session(session_id):
        return app.invoke({"question": question, "retries": 0, "validation_errors": []}, config=config)


def run_team_builder(question: str) -> str:
    return run_team_builder_full(question)["final_team"]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or f"Build me a team for the {RACE_NAME} at {RACE_LOCATION}."
    print(f"QUESTION: {q}\n")
    print(run_team_builder(q))
