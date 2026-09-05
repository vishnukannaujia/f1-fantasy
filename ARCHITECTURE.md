# f1-fantasy -- Architecture

Living doc. Update this alongside the code as the design evolves -- see Changelog at the bottom.

## Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#1C1F28','primaryTextColor':'#E8EAF0','primaryBorderColor':'#2A2E3A','lineColor':'#8B90A3','fontFamily':'ui-monospace, monospace','fontSize':'13px' }}}%%
flowchart TD
    subgraph SRC["Manual bootstrap (this session's WebSearch/WebFetch)"]
        WS1["Claude Code session tools<br/>prices - standings - results"]:::live
    end

    subgraph STATIC["Static Knowledge (data/raw/*.txt)"]
        RULES["01_fantasy_scoring_rules.txt"]:::static
        CIRCUIT["06_monza_circuit_notes.txt<br/>+ per-circuit profiles (future)"]:::static
        PRICES["02_driver_prices.txt - 03_constructor_prices.txt<br/>04_championship_standings.txt - 05_recent_form.txt"]:::structured
    end

    WS1 -->|"NOT YET BUILT: refresh_data.py"| PRICES
    RULES --> CHUNK["split_into_paragraphs<br/>dynamic cap (ingest.py)"]:::static
    CIRCUIT --> CHUNK
    CHUNK --> VDB[("Chroma vector store")]:::static
    PRICES --> ROWCHUNK["build_row_chunks<br/>one chunk per driver/constructor"]:::structured
    ROWCHUNK --> VDB
    PRICES --> PARSE["Parsed price tables<br/>(direct load, NOT embedded)"]:::structured

    subgraph EVAL["eval/eval_chunking.py"]
        STRUCT["Structural checks<br/>no embeddings"]:::neutral
        RECALL["recall@k vs. 15 labeled queries<br/>k=4 (production) = 100%"]:::neutral
    end
    VDB -.verified by.-> RECALL

    subgraph QA["General Q&A -- chains.py"]
        RETR["retriever.invoke(question)<br/>top-k=4 similarity"]:::static
        PLAINLLM["ChatAnthropic"]:::neutral
        VDB --> RETR --> PLAINLLM --> ANS["answer"]
    end

    subgraph TB["Team Builder -- team_builder.py (LangGraph)"]
        LOAD["load_static_context<br/>rules + circuit + parsed tables"]:::neutral
        LIVE["fetch_live_conditions<br/>Anthropic hosted web_search tool"]:::live
        PROP["propose_team (LLM, max_tokens=8192)"]:::neutral
        VALID{"validate_budget<br/>deterministic Python<br/>&lt;= $100M, 5 drivers + 2 constructors"}:::neutral
        FINAL["finalized team + reasoning"]

        PARSE --> LOAD
        LOAD --> LIVE --> PROP --> VALID
        VALID -->|over budget, retries left| PROP
        VALID -->|valid, or retries exhausted| FINAL
    end

    classDef live fill:#3D2B5C,stroke:#B983FF,color:#F3E8FF,stroke-width:2px;
    classDef static fill:#1B3B2C,stroke:#4ADE80,color:#DCFCE7,stroke-width:2px;
    classDef structured fill:#4A3B12,stroke:#FBBF24,color:#FEF3C7,stroke-width:2px;
    classDef neutral fill:#262A36,stroke:#9CA3AF,color:#E5E7EB,stroke-width:1.5px;
```

**Important distinction the diagram makes explicit**: the "Manual bootstrap" box (top) is *this Claude Code session's own* WebSearch/WebFetch tools, used once to write the initial `data/raw/*.txt` files -- those tools only exist inside this coding session and are not available to the standalone repo. `fetch_live_conditions` inside `team_builder.py`, by contrast, uses **Anthropic's own hosted `web_search` server tool** (via the raw `anthropic` SDK, `tools=[{"type": "web_search_20250305", ...}]`) -- that's the only live-data path that actually works when you run this repo's code yourself, independent of any Claude Code session.

## Legend

| Color | Meaning | Refresh cadence |
|---|---|---|
| 🟣 Purple | Live tool call (WebSearch/WebFetch), on-demand every query | Real-time, never cached |
| 🟢 Green | Static RAG corpus (vector store, top-k similarity search) | Rare (rules), or weekly (prices re-embedded) |
| 🟡 Amber | Structured data, loaded directly (not vector-searched) | Weekly, via `refresh_data.py` (**not yet built** -- see Known gaps) |
| ⚪ Slate | Orchestration / deterministic logic node | -- |

## Why prices/standings are NOT pure vector RAG

Team-building is a budget-constrained optimization across all 22 drivers + 11 constructors
simultaneously -- a top-k similarity search would only surface a handful of relevant chunks, not the
full price landscape needed to check a $100M budget. So these files are embedded into the vector store
*for the general Q&A chain* (single-fact lookups like "what's Hamilton's price?" work fine with top-k),
but also parsed directly into Python tables for the team-builder graph, which needs the whole table at
once. Same source files, two access patterns, chosen by what the query actually needs.

## Why live weather/incidents are tool calls, not corpus files

They're stale within hours -- baking them into the persisted vector store would either pollute it with
short-lived data or require re-ingesting before every single query. `fetch_live_conditions` calls
WebSearch on-demand instead, and its result is used transiently for that turn only.

## Why validate_budget is deterministic Python, not another LLM call

LLMs are unreliable at exact arithmetic over many line items. The retry loop
(`propose_team -> validate_budget -> propose_team`) mirrors the `grade -> retry` pattern from the
`rag-demo` sibling project, but here it enforces a hard numeric constraint instead of a relevance
judgment -- arguably a more justified use of the pattern.

## Why chunking is paragraph-level, not sentence-level or fixed-size

`RecursiveCharacterTextSplitter` at a fixed `chunk_size` was tried and rejected: it *merges* adjacent
short splits back together, so no single size worked across files whose natural paragraphs are
different lengths (short labeled rule sections vs. longer prose). Splitting further, to individual
sentences, was tested empirically and made things measurably worse: recall@4 on the eval set dropped
from 100% (paragraph-level) to 53% (sentence-level), because several sentences in this corpus depend on
a heading or neighboring bullet for their meaning (e.g. "Sprint DNF: -10..." only reads as
sprint-specific alongside its "SPRINT RACE POINTS" heading -- split apart, it becomes indistinguishable
from the generic race-DNF penalty). The rule that held up under testing: chunk at the boundary where a
fact stops being self-contained -- the paragraph for prose, the single line for the flat price tables
(where each line already has no shared context to lose).

## Chunking strategy reference (exact sizes, as of last ingest)

| Parameter | Value | Where it comes from |
|---|---|---|
| Prose chunking method | Split on `\n\n` (paragraph boundary), no merging | `split_into_paragraphs()` in `ingest.py` |
| Fallback splitter (rare) | `RecursiveCharacterTextSplitter`, only for a paragraph that individually exceeds the cap | `ingest.py` |
| Dynamic paragraph cap margin | 1.25x the longest real paragraph | `PARAGRAPH_MAX_LEN_MARGIN` |
| Dynamic paragraph cap floor | 600 chars minimum, regardless of corpus | `PARAGRAPH_MAX_LEN_FLOOR` |
| **Current computed cap** | **1162 chars** (930-char longest paragraph x 1.25) | `determine_paragraph_max_len()`, re-derived on every `ingest.py` run |
| Row-chunk files | `02_driver_prices.txt`, `03_constructor_prices.txt` -- one chunk per line, not paragraph-split at all | `ROW_CHUNK_FILES` in `ingest.py` |
| Prose chunks produced | 26 (from 4 files: `01`, `04`, `05`, `06`) | min=89, max=930, mean=349 chars |
| Row chunks produced | 33 (22 drivers + 11 constructors) | min=46, max=72 chars |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, local | `EMBEDDING_MODEL` in `ingest.py` |
| Vector store | Chroma, cosine similarity, persisted to `./chroma_db/` | `ingest.py` |
| Production retrieval `k` | 4 (used by `chains.py`'s default) | `PRODUCTION_K` in `eval_chunking.py` |

This cap is **derived, not hand-picked** -- rerun `python src/ingest.py` after adding new corpus files and
these numbers (especially the computed cap) will change automatically if a longer paragraph appears.
Don't hardcode these values elsewhere; read them from the source if you need them programmatically.

## Evals -- purpose, usage, and what each one actually checks

Five scripts under `eval/` and one prediction ledger under `predictions/`, each validating a different
layer of the system. None require pytest -- all are plain runnable scripts with `assert`/exit-code-based
pass/fail, consistent with the project's existing style.

| Script | What it checks | Needs LLM calls? | Current result |
|---|---|---|---|
| `eval/eval_chunking.py` | **Chunking layer.** Layer 1 (structural): every parsed driver/constructor maps to exactly one row chunk, every prose chunk respects the dynamic cap. Layer 2 (retrieval): recall@k against 15 hand-labeled real questions, including two deliberate "collision" cases (sprint vs. race DNF, sprint vs. race fastest lap) built to stress-test disambiguation. | No (embeddings only) | recall@1=67%, **recall@4=100%** (production k), recall@8=100% |
| `eval/eval_team_builder_logic.py` | **Parsing + budget math.** `_parse_proposal`, `_lookup`, `validate_budget`, `route_after_validation` against hand-built fake proposals (valid, over-budget, wrong roster size, unknown driver name, captain outside roster). This is pure Python logic -- a bug here would silently corrupt every team recommendation regardless of how good the LLM reasoning is. | No | 19/19 checks pass |
| `eval/eval_generation.py` | **Generation quality**, not just retrieval -- does the FINAL answer state the right fact (not just "was the right chunk retrieved," which `eval_chunking.py` already covers). Includes one deliberately out-of-corpus question (Anthropic's financials) to check the model declines rather than hallucinates; this specific check needed an LLM judge after three straight heuristic/regex attempts produced false negatives from natural phrasing variance -- see the comment in the file for why. | Yes | 10/10 pass |
| `eval/eval_prediction_backtest.py` | **Held-out forecasting accuracy.** Predicts a race that already happened (Dutch GP/Zandvoort) using ONLY the context that would have been available before it (fresh research into pre-race standings/form/circuit notes), scores the prediction against the real known result. This is what actually validates whether `team_builder.py`'s reasoning process can be trusted for a real, unresolved upcoming race. | Yes | recall top6=5/6, podium=2/3, winner wrong (see finding below) |
| `eval/predictions_tracker.py` | **Ongoing live validation**, turning the one-off backtest above into a repeatable practice. `score` command takes a real post-race result and scores every saved formation in a `predictions/*.json` file against it (driver overlap, captain's actual finish); `summary` prints the track record across every race scored so far. This is how a single n=1 backtest becomes a real, growing calibration dataset over the rest of the season. | No | Italian GP prediction saved 2026-09-05, awaiting the real result to score |

**A concrete finding from the backtest, not just a pass/fail number**: the Zandvoort backtest correctly
identified 5 of the real top-6 finishers and 2 of 3 podium finishers, but predicted the wrong winner --
its own stated reasoning explicitly discounted a driver's just-happened win as "a one-off rather than
sustained pace," and that driver went on to win again. This was fed back into `team_builder.py`'s
`PROPOSAL_INSTRUCTIONS` as an explicit instruction to weight immediate hot-streak form at least as
heavily as season-long standings -- a real, evidence-backed prompt change, not a hypothetical one. That
fix has **not yet been re-validated** against a second held-out race -- see Known gaps.

## Known gaps (honest as of last update)

- **`refresh_data.py` does not exist yet.** `data/raw/*.txt` was populated once via this session's own
  WebSearch/WebFetch tools; there is no automated weekly refresh. Updating prices/standings/form before
  a future race weekend currently means manually re-running that research and rewriting the files.
- **`04_championship_standings.txt` is incomplete** -- only the top 2 drivers (Antonelli, Russell) were
  confirmed; the rest of the driver order was not found during research and is explicitly flagged as
  missing in the file itself.
- **The Zandvoort backtest's prompt fix has not been re-validated.** Weighting recent hot-streak form
  was added to `PROPOSAL_INSTRUCTIONS` based on ONE failure case. `eval/predictions_tracker.py` exists
  now specifically to accumulate more real data points as the season continues -- the Italian GP
  prediction is saved and awaiting the real result; a second backtest (e.g. the Hungarian GP, for which
  pre-race context was already gathered) would be the fastest way to check the fix generalizes rather
  than just patches the one case we happened to see fail.
- **No hardening against indirect prompt injection via live web search.** `fetch_live_conditions`
  concatenates live web content directly into the `propose_team` prompt with no sanitization. The
  sibling project `rag-demo` has a dedicated red-team harness for exactly this class of risk in
  retrieved content; this project has not been threat-modeled at all yet. Lower stakes here (personal
  tool, not production), but a real gap, not an oversight to hide.
- **No error handling** if `fetch_live_conditions`'s web search call fails (network error, rate limit,
  empty result) -- an exception there currently crashes the whole graph rather than degrading gracefully.
- **No web UI yet** -- everything so far is CLI/script-driven (`python src/team_builder.py "..."`,
  `python eval/*.py`). A chatbot web front-end is planned but not yet built.

## Changelog

- **2026-09-05** -- Initial architecture: static RAG (rules + circuit profiles) vs. structured
  direct-load (prices/standings/form) vs. live tool calls (weather/incidents), wired into a
  `load_static_context -> fetch_live_conditions -> propose_team -> validate_budget` LangGraph loop for
  the team-builder, plus a plain `chains.py` Q&A path reusing the same vector store.
- **2026-09-05** -- Reworked prose chunking from fixed-size `RecursiveCharacterTextSplitter` to
  paragraph-boundary splitting with a corpus-derived dynamic cap (`determine_paragraph_max_len`), added
  row-level chunking for the two price files, and built `eval/eval_chunking.py` (structural checks +
  recall@k against 15 labeled queries) -- verified recall@4 = 100%, and empirically confirmed
  sentence-level splitting performs worse (53%) rather than assuming it.
- **2026-09-05** -- First real end-to-end run of `team_builder.py` surfaced a genuine bug: the
  `propose_team` LLM call was hitting `stop_reason="max_tokens"` with its entire token budget consumed
  by extended thinking, producing a completely empty response (0 drivers/constructors parsed on every
  retry). Fixed by raising `max_tokens` from 2048 to 8192. Confirmed working after the fix: produced a
  valid, under-budget (\$99.7M/\$100M) team for the Italian GP, correctly incorporating a real live detail
  (a driver's grid penalty) surfaced by `fetch_live_conditions` into its reasoning.
- **2026-09-05** -- Built the remaining eval layers: `eval_team_builder_logic.py` (19/19 unit checks on
  parsing/budget math), `eval_generation.py` (10/10, needed an LLM judge for one check after three
  straight substring/regex heuristics produced false negatives), and `eval_prediction_backtest.py` (a
  held-out backtest against the real Zandvoort result -- 5/6 top-6 overlap, wrong winner, which directly
  motivated a hot-streak-weighting fix to `PROPOSAL_INSTRUCTIONS`). Added `predictions_tracker.py` and
  `predictions/` to turn that one-off backtest into an ongoing, repeatable practice across future races.
  Generated 4 strategically distinct team formations for the Italian GP (Safe, Value, Winnability-
  optimized, Aggressive) and saved them to `predictions/2026-09-06_italian_gp.json`, pending scoring once
  the race happens.
