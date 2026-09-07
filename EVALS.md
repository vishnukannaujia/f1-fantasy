# f1-fantasy — Evals

Living doc, split out from `ARCHITECTURE.md` because it grew past a summary table. This page covers
every eval script under `eval/`, the two related tools that aren't strictly pass/fail evals, and the
concrete design decisions each eval actually drove — not just numbers, but what changed because of them.

**The throughline across all of this**: test empirically, don't assume. Several conclusions in this repo
turned out to be the opposite of the "obviously correct" guess (sentence-level chunking looked reasonable
and lost badly; a fancier LLM-proposed chunking scheme underperformed the simple paragraph-boundary
approach; a cross-encoder reranker sounded like a clean upgrade and turned out to be a lateral trade, not
a win). None of that would have surfaced without actually running the eval and looking at the number.

## Eval scripts (pass/fail, runnable directly)

| Script | What it checks | Needs LLM calls? | Current result |
|---|---|---|---|
| `eval/eval_chunking.py` | **Chunking layer.** Layer 1 (structural, no embeddings): every parsed driver/constructor maps to exactly one row chunk, every prose chunk respects the dynamic cap. Layer 2 (retrieval): recall@k against 15 hand-labeled real questions, including two deliberate "collision" cases (sprint vs. race DNF, sprint vs. race fastest lap) built to stress-test disambiguation. | No (embeddings only) | recall@1=93%, **recall@4=100%** (production k), recall@8=100% |
| `eval/eval_chunking_bakeoff.py` | **Chunking strategy comparison.** Tests 9 chunking strategies (fixed-size at 3 sizes, token-based, sentence-based, structure-aware/current, semantic embedding-breakpoint, sliding window, agentic/LLM-proposed) against the exact same 15-query set, holding everything else constant. | Yes (agentic strategy only) | Current approach ties for best on recall@4 (100%) and wins on every secondary axis (cost, interpretability, no manual tuning) |
| `eval/eval_team_builder_logic.py` | **Parsing + budget math + learnings-loop weighting.** `_parse_proposal`, `_lookup`, `validate_budget`, `route_after_validation` against hand-built fake proposals (valid, over-budget, wrong roster size, unknown driver name, captain outside roster). Also `render_learnings` — the evidence-weighting logic that was refactored out of `load_learnings_text()` specifically to make it testable without touching the filesystem: confidence/evidence-count tagging, defaults when fields are missing, `refines` tags appearing only where set, oldest-added-first ordering (the mechanism behind the recency nudge), and that the anti-overfitting instructions themselves are actually present in the rendered text, not just documented as intent in a comment. Pure Python logic — a bug here would silently corrupt every team recommendation, or silently break the one thing standing between "learn from mistakes" and "overfit to the last race," regardless of how good the LLM reasoning is. | No | 30/30 checks pass |
| `eval/eval_generation.py` | **Generation quality**, not just retrieval — does the FINAL answer state the right fact (`eval_chunking.py` only checks whether the right chunk was retrieved, not what the model did with it). Includes one deliberately out-of-corpus question (Anthropic's financials) to check the model declines rather than hallucinates; needed an LLM judge after three straight heuristic/regex attempts produced false negatives from natural phrasing variance. | Yes | 10/10 pass |
| `eval/eval_quality_gate.py` | **The similarity-score-threshold retriever and its deterministic "skip the LLM" short-circuit** in `chains.py`. Verifies clearly off-topic queries retrieve zero docs and the chain returns `NOT_FOUND_MESSAGE` verbatim (no LLM call needed to check this — the gate itself prevents one), and that legitimate + topically-adjacent-but-factually-absent queries still retrieve normally. | No (even for the off-topic cases — that's the point of the gate) | 7/7 pass |
| `eval/eval_prediction_backtest.py` | **Held-out forecasting accuracy.** Predicts a race that already happened (Dutch GP/Zandvoort) using ONLY the context that would have been available before it, scores the prediction against the real known result. What actually validates whether `team_builder.py`'s reasoning can be trusted for a real, unresolved upcoming race. | Yes | recall top6=5/6, podium=2/3, winner wrong (see the concrete finding below) |

## Related tooling (not pass/fail evals, but part of the same practice)

| Script | Purpose |
|---|---|
| `eval/predictions_tracker.py` | Turns the one-off backtest above into an ongoing practice: `score` takes a real post-race result and scores every saved formation in a `predictions/*.json` file against it; `summary` prints the track record across every race scored so far. |
| `eval/learnings_loop.py` | Extracts evidence-weighted lessons from a scored prediction file and proposes (doesn't auto-commit) additions to `learnings/learnings.json`, which `team_builder.py` reads on every future call. Classifies each proposal as NEW / REFINES an existing lesson / CONFIRMS one, so a new race's evidence narrows an existing lesson instead of silently overriding it. |

## Decisions actually made via these evals (not picked on theory)

**Chunking strategy** — paragraph-boundary splitting with a corpus-derived dynamic cap, not any fixed
`chunk_size`. `RecursiveCharacterTextSplitter` at a fixed size always *merges* adjacent short splits back
together, so no single size worked across files with naturally different paragraph lengths. Sentence-level
splitting was tested and dropped recall@4 to 53% — several sentences in this corpus depend on a heading or
neighboring bullet for their meaning, and splitting them apart loses that. See `eval_chunking_bakeoff.py`.

**Embedding model** — `BAAI/bge-large-en-v1.5`, chosen via a real head-to-head, not by reputation:

| Model | Dims | Type | recall@1 |
|---|---|---|---|
| `all-MiniLM-L6-v2` (original) | 384 | Symmetric similarity | 60% |
| `all-mpnet-base-v2` | 768 | Symmetric similarity | 80% |
| **`bge-large-en-v1.5` (current)** | 1024 | **Retrieval-tuned, asymmetric** | **93%** |

recall@4/8 were already at 100% throughout, so recall@1 was the only axis that differentiated these three.
The jump to BGE isn't explained by size alone — it's trained for asymmetric query-to-passage retrieval,
which is structurally what `chains.py` does (a short question matched against a longer chunk), unlike
MPNet's general sentence-similarity training.

**Distance metric** — cosine, declared explicitly (`collection_metadata={"hnsw:space": "cosine"}` in
`ingest.py`), not left as Chroma's undeclared default (squared L2). Tested all three options Chroma
supports (cosine, l2, ip) head-to-head: **identical** results across all three (93%/100%/100%), exactly as
the math predicts for unit-normalized vectors (`‖a-b‖² = 2(1-cos_sim)` when `‖a‖=‖b‖=1`, verified our BGE
output actually is unit-normalized — sampled vector norms were all exactly 1.0). Declaring cosine explicitly
isn't about improving today's numbers, which are already tied — it's about not silently depending on the
current embedding model's default normalization behavior, which a future model swap might not share.

**Similarity score threshold** — `score_threshold=0.60` (relevance score, i.e. `1 - cosine_distance`) on
`chains.py`'s retriever, picked from real measured score distributions, not a guess:
- Legitimate in-corpus queries (all 15 eval questions): distance 0.089–0.363 worst case.
- Clearly off-topic queries ("What is Formula E?", "How do I bake a chocolate cake?"): distance 0.44–0.665.
- **Important limit found in the process**: a topically-adjacent-but-factually-absent question ("What was
  the 2025 Italian GP result?" — this corpus only covers 2026) scored 0.24, well inside the "legitimate"
  range, because it's topically close even though the specific fact isn't present. A distance threshold
  is a coarse pre-filter for clearly unrelated questions, not a fact-checker — that job still belongs to
  the LLM's own judgment via the system prompt, and both layers are needed together, not one or the other.

**Programmatic quality gate** — when nothing clears the score threshold, `chains.py` returns a
deterministic `NOT_FOUND_MESSAGE` and skips the LLM call entirely, rather than sending an empty-context
prompt and hoping the model notices. Cheaper, faster, and doesn't depend on the model reliably catching an
edge case unprompted.

**Cross-encoder reranking — tested, and explicitly NOT added.** Retrieved top-10 via the bi-encoder,
reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, compared recall@1 before/after on the same 15
queries: reranking fixed one case but broke a different one that was previously correct — net recall@1
identical (14/15 either way). Inspected the regression directly: the reranker scored a sprint-specific
chunk *higher* than the correct answer for a query that explicitly said "not a sprint," over-weighting
lexical overlap with racing terminology without handling the negation correctly. A general-purpose
web-search-trained reranker isn't a clean win on this narrow, jargon-dense, negation-sensitive corpus —
concrete evidence, not a guess, for why this wasn't wired into production. This is exactly as legitimate
an outcome of testing as a positive result would have been.

## A concrete finding from the backtest, not just a pass/fail number

The Zandvoort backtest correctly identified 5 of the real top-6 finishers and 2 of 3 podium finishers, but
predicted the wrong winner — its own stated reasoning explicitly discounted a driver's just-happened win as
"a one-off rather than sustained pace," and that driver went on to win again. This was fed back into
`learnings/learnings.json` (see `ARCHITECTURE.md` for the full learnings-loop design) as an explicit
instruction to weight immediate hot-streak form at least as heavily as season-long standings — a real,
evidence-backed correction, not a hypothetical one. A second finding from the real, scored Italian GP result
then *refined* (not overrode) that lesson further: a confirmed large grid penalty for a driver on a
structurally faster car can outweigh pure momentum, which is why `learnings.json` tracks `refines`
relationships between lessons instead of flattening everything into one undifferentiated rule list.
