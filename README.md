# f1-fantasy

A RAG + LangGraph tool for the official F1 Fantasy game (fantasy.formula1.com): answers questions about
the current 2026 F1 season, and recommends a budget-constrained 5-driver + 2-constructor team grounded
in real current prices, standings, form, and live race-weekend conditions.

Grown out of a learning project on RAG/LangGraph design (see the sibling repo
[`rag-demo`](../rag-demo)) into something aimed at actually being useful for playing F1 Fantasy this
season. Full design rationale and diagram: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

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

Embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`); the vector store is a local
Chroma DB at `./chroma_db/`. Only the LLM calls (generation, and `team_builder.py`'s live web search)
need the Anthropic API key.

## Run it

```bash
python src/ingest.py                                          # build the vector store (once, or
                                                                # whenever data/raw/ changes)

python src/chains.py                                           # (see chains.py -- import and call
                                                                # build_rag_chain() for single questions)

python src/team_builder.py "Build me a team for the Italian Grand Prix at Monza."
```

`team_builder.py`'s `fetch_live_conditions` node calls Anthropic's own hosted `web_search` server tool
to pull the actual current weather forecast and any breaking race-weekend news -- this is genuinely live
data fetched at run time, not baked into the corpus (see ARCHITECTURE.md for why).

## Evaluating it

```bash
python eval/eval_chunking.py
```

Two layers: fast structural checks on the chunking functions themselves (no embeddings needed), and a
recall@k retrieval eval against 15 hand-labeled real questions -- including two "collision" cases
(sprint vs. race DNF penalty, sprint vs. race fastest-lap bonus) built specifically to stress-test
whether near-duplicate facts get disambiguated correctly. Currently: **recall@4 = 100%** (k=4 is the
production default in `chains.py`).

## Project layout

```
data/raw/               current F1 2026 season corpus (rules, prices, standings, form, circuit notes)
src/ingest.py            chunk (paragraph-level + row-level) + embed + persist to Chroma
src/f1_data.py           shared price-line parser (used by ingest.py and team_builder.py)
src/chains.py            plain LangChain RAG chain: single-fact Q&A
src/team_builder.py      LangGraph: budget-constrained team recommendation
eval/eval_chunking.py    structural + retrieval-quality eval for the chunking layer
chroma_db/               persisted vector store (gitignored, created by ingest.py)
ARCHITECTURE.md          diagram + design rationale + known gaps (living doc)
```

## Known gaps

See **[ARCHITECTURE.md § Known gaps](ARCHITECTURE.md#known-gaps-honest-as-of-last-update)** for the
current honest list -- as of last update: no automated weekly data refresh, the standings file is
missing the full driver order beyond the top 2, no generation-quality eval yet (only retrieval is
formally eval'd), and no hardening against indirect prompt injection via live web search content.
