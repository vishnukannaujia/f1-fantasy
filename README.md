# f1-fantasy

A RAG + LangGraph tool for the official F1 Fantasy game (fantasy.formula1.com): answers questions about
the current 2026 F1 season, and recommends a budget-constrained 5-driver + 2-constructor team grounded
in real current prices, standings, form, and live race-weekend conditions.

Grown out of a learning project on RAG/LangGraph design (see the sibling repo
[`rag-demo`](../rag-demo)) into something aimed at actually being useful for playing F1 Fantasy this
season. Full design rationale and diagram: **[ARCHITECTURE.md](ARCHITECTURE.md)**. Eval scripts, current
pass/fail numbers, and the concrete decisions each one drove: **[EVALS.md](EVALS.md)**.

## The topic

The corpus (`data/raw/*.txt`) covers the **2026 F1 season as of early September 2026**: official F1
Fantasy scoring rules, current driver/constructor prices, championship standings, the most recent race
result and current form, and circuit notes for the upcoming Italian Grand Prix at Monza. This is
deliberately current-season data a base LLM's training data won't reliably have -- the same principle
`rag-demo` demonstrated with recent AI news, applied here to something with a direct personal use case
(playing the actual fantasy game, not just a demo).

## Why LangChain, and why LangGraph

- **`chains.py`** -- a plain retrieve-then-generate LCEL chain, right for single-fact questions ("what's
  Hamilton's price?", "what does the DRS Boost do?"). Top-k similarity search over a local Chroma store
  is the correct tool here.
- **`team_builder.py`** -- a LangGraph `load_static_context -> fetch_live_conditions -> propose_team ->
  validate_budget` loop, because team-building is fundamentally different from a lookup: it needs the
  *entire* price table simultaneously (not a similarity-ranked top-k subset) to check a hard $100M
  budget constraint, and LLMs are unreliable at exact arithmetic over many line items. `validate_budget`
  is deterministic Python, and the graph retries `propose_team` with corrective feedback when the
  proposal is invalid -- a LangGraph retry loop enforcing a real constraint, not just a soft relevance
  judgment.

## Setup

```bash
cd ~/GitHub/f1-fantasy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Embeddings run locally via `sentence-transformers` (`BAAI/bge-large-en-v1.5` -- picked over smaller/more
generic models via a real head-to-head eval, see `ARCHITECTURE.md` § Embedding model reference); the
vector store is a local Chroma DB at `./chroma_db/`. Only the LLM calls (generation, and
`team_builder.py`'s live web search) need the Anthropic API key.

## Run it

```bash
python src/ingest.py                                          # build the vector store (once, or
                                                                # whenever data/raw/ changes)

python src/chains.py                                           # (see chains.py -- import and call
                                                                # build_rag_chain() for single questions)

python src/team_builder.py "Build me a team for the Italian Grand Prix at Monza."

python web/app.py   # local chat UI at localhost:5001 -- routes each message to chains.py (fast
                     # Q&A) or team_builder.py (slower, live-data team recommendation) by keyword
```

`team_builder.py`'s `fetch_live_conditions` node calls Anthropic's own hosted `web_search` server tool
to pull the actual current weather forecast and any breaking race-weekend news -- this is genuinely live
data fetched at run time, not baked into the corpus (see ARCHITECTURE.md for why).

**Tracing** (optional, off by default): set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` and/or
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env` to trace every call to LangSmith and/or Langfuse
side by side -- see `src/observability.py` and the `langsmith-trace` skill under `.claude/skills/`.

## Evaluating it

See **[EVALS.md](EVALS.md)** for the full table of what each script checks, current pass/fail numbers, and
the concrete decisions each one drove. Quick reference:

```bash
python eval/eval_chunking.py             # chunking: structural checks + recall@k (currently recall@4=100%)
python eval/eval_chunking_bakeoff.py     # compares 9 chunking strategies head-to-head
python eval/eval_team_builder_logic.py   # parsing + budget-math unit checks, no LLM calls (19/19 pass)
python eval/eval_generation.py           # does the final ANSWER state the right fact, not just retrieval (10/10 pass)
python eval/eval_quality_gate.py         # similarity-score threshold + deterministic "skip the LLM" gate (7/7 pass)
python eval/eval_prediction_backtest.py  # held-out: predict a past race, score vs. the real result
python eval/predictions_tracker.py summary   # track record across every race scored so far
```

`eval/eval_chunking.py` includes two "collision" cases (sprint vs. race DNF penalty, sprint vs. race
fastest-lap bonus) built specifically to stress-test whether near-duplicate facts get disambiguated
correctly. `eval/eval_prediction_backtest.py` and `predictions_tracker.py` together turn a one-off
backtest into an ongoing practice: predictions are saved to `predictions/*.json` before a race, then
scored against the real result afterward, building a real (not n=1) calibration dataset over the season.

## Project layout

```
data/raw/                 current F1 2026 season corpus (rules, prices, standings, form, circuit notes)
predictions/               saved pre-race predictions, scored against real results after each race
learnings/learnings.json   evidence-weighted lessons fed back into team_builder.py's prompt automatically
src/ingest.py              chunk (paragraph-level + row-level) + embed + persist to Chroma
src/f1_data.py             shared price-line parser (used by ingest.py and team_builder.py)
src/chains.py              plain LangChain RAG chain: single-fact Q&A, with a score-threshold quality gate
src/team_builder.py        LangGraph: budget-constrained team recommendation
src/observability.py       shared LangSmith/Langfuse tracing + session correlation helpers
web/app.py                 local Flask chat UI, routes to chains.py or team_builder.py
studio_graph.py, langgraph.json   LangGraph Studio config (python -m langgraph dev)
eval/eval_chunking.py      structural + retrieval-quality eval for the chunking layer
eval/eval_chunking_bakeoff.py     9-way chunking strategy comparison
eval/eval_team_builder_logic.py   unit checks for proposal parsing + budget validation
eval/eval_generation.py    generation-quality eval (answer correctness, not just retrieval)
eval/eval_quality_gate.py  score-threshold retriever + deterministic "skip the LLM" gate
eval/eval_prediction_backtest.py  held-out race-outcome backtest
eval/predictions_tracker.py       save/score predictions against real results, ongoing
eval/learnings_loop.py     proposes evidence-weighted lessons from a scored prediction
chroma_db/                 persisted vector store (gitignored, created by ingest.py)
ARCHITECTURE.md            diagram + design rationale + chunking/embedding reference + known gaps
EVALS.md                   every eval script + the decisions each one drove
```

## Known gaps

See **[ARCHITECTURE.md § Known gaps](ARCHITECTURE.md#known-gaps-honest-as-of-last-update)** for the
current honest list -- as of last update: no automated weekly data refresh, the standings file is
missing the full driver order beyond the top 2, the backtest-driven prompt fix hasn't been re-validated
against a second race, no hardening against indirect prompt injection via live web search content,
Langfuse is wired but never key-verified against a real trace, and the embedding-space visualization
artifact's hover tooltip doesn't work (confirmed via a real browser check, not yet root-caused).
