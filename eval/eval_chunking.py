"""
Eval for the chunking layer in src/ingest.py.

Two layers, cheapest first:
  1. Structural checks -- no embeddings, no LLM calls, runs in milliseconds.
     Verifies the chunking functions' own guarantees: every parsed driver/
     constructor appears in exactly one row chunk, and every prose chunk
     respects the dynamically-derived paragraph cap (validate_chunks would
     already raise on this at ingest time -- this re-runs it standalone so a
     chunking regression is caught by `eval`, not just discovered at ingest).
  2. Retrieval eval -- recall@k against a hand-labeled set of real questions,
     run against the live persisted vector store. This is the layer that
     actually answers "does the chunking strategy serve retrieval well?" --
     structural correctness alone doesn't guarantee a query finds the right
     chunk (see the sprint-DNF-vs-race-DNF collision case below).

Run any time chunking logic or the corpus changes:
    python eval/eval_chunking.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from f1_data import parse_constructor_prices, parse_driver_prices  # noqa: E402
from ingest import (  # noqa: E402
    DATA_DIR,
    build_row_chunks,
    determine_paragraph_max_len,
    get_vector_store,
    load_prose_documents,
    split_into_paragraphs,
    validate_chunks,
)

# ---------- Layer 1: structural checks (no embeddings, no LLM) ----------


def check_row_chunk_coverage():
    drivers = parse_driver_prices(DATA_DIR / "02_driver_prices.txt")
    constructors = parse_constructor_prices(DATA_DIR / "03_constructor_prices.txt")
    row_chunks = build_row_chunks(DATA_DIR)

    driver_chunks = [c for c in row_chunks if "driver price" in c.page_content]
    constructor_chunks = [c for c in row_chunks if "constructor price" in c.page_content]

    assert len(driver_chunks) == len(drivers), (
        f"expected {len(drivers)} driver row chunks, got {len(driver_chunks)}"
    )
    assert len(constructor_chunks) == len(constructors), (
        f"expected {len(constructors)} constructor row chunks, got {len(constructor_chunks)}"
    )
    for name in drivers:
        assert any(name in c.page_content for c in driver_chunks), (
            f"driver {name!r} is missing from the row chunks"
        )
    for team in constructors:
        assert any(team in c.page_content for c in constructor_chunks), (
            f"constructor {team!r} is missing from the row chunks"
        )
    return f"{len(drivers)} drivers + {len(constructors)} constructors each map to exactly one row chunk"


def check_paragraph_cap_respected():
    prose_docs = load_prose_documents(DATA_DIR)
    max_len = determine_paragraph_max_len(prose_docs)
    chunks = []
    for doc in prose_docs:
        chunks.extend(split_into_paragraphs(doc, max_len=max_len))
    validate_chunks(chunks, max_len, label="prose chunks")  # raises SystemExit on any violation
    return f"all {len(chunks)} prose chunks respect the dynamically-derived {max_len}-char cap"


STRUCTURAL_CHECKS = [check_row_chunk_coverage, check_paragraph_cap_respected]


# ---------- Layer 2: retrieval eval (recall@k against real queries) ----------

# (query, required substrings -- ALL must appear (case-insensitive) somewhere in
#  the top-k chunks for a hit, note)
EVAL_CASES = [
    ("What is George Russell's price?", ["Russell", "27.6"], "driver price lookup"),
    ("How much does Valtteri Bottas cost?", ["Bottas", "3.0"], "driver price lookup, cheap end"),
    ("What's Fernando Alonso's fantasy price?", ["Alonso", "6.8"], "driver price lookup"),
    ("What does Cadillac cost in fantasy?", ["Cadillac", "3.0"], "constructor price lookup"),
    ("How expensive is the Mercedes constructor?", ["Mercedes", "32.9"], "constructor price lookup"),
    ("What is the sprint DNF penalty?", ["Sprint DNF", "-10"], "collision: sprint vs. race DNF"),
    ("What is the DNF penalty in a normal Grand Prix, not a sprint?", ["-20 points"], "collision: race vs. sprint DNF"),
    ("What does the Wildcard chip do?", ["Wildcard", "unlimited free transfers"], "rules lookup"),
    ("How many free transfers are allowed per race weekend?", ["Two free transfers"], "rules lookup"),
    ("What is the qualifying points system?", ["P1", "P10"], "rules lookup"),
    ("What's the fastest lap bonus in a Grand Prix?", ["Fastest lap", "+10"], "collision: race vs. sprint fastest lap"),
    ("Who leads the drivers championship?", ["Antonelli", "242"], "standings lookup"),
    ("Who won the most recent race?", ["Norris", "Zandvoort"], "recent form lookup"),
    ("Is Monza a low downforce or high downforce circuit?", ["downforce"], "circuit notes lookup"),
    ("Which team is the favorite at Monza?", ["Mercedes"], "circuit notes lookup"),
]

K_VALUES = (1, 4, 8)
PRODUCTION_K = 4  # matches chains.py's default k


def run_retrieval_eval(vector_store, k_values=K_VALUES):
    retrievers = {k: vector_store.as_retriever(search_kwargs={"k": k}) for k in k_values}
    rows, hits_by_k = [], {k: [] for k in k_values}

    for query, required, note in EVAL_CASES:
        row = {"query": query, "note": note}
        for k in k_values:
            docs = retrievers[k].invoke(query)
            combined = "\n".join(d.page_content for d in docs).lower()
            hit = all(req.lower() in combined for req in required)
            row[f"k={k}"] = hit
            hits_by_k[k].append(hit)
        rows.append(row)
    return rows, hits_by_k


def print_report(rows, hits_by_k, k_values=K_VALUES):
    col = max(len(r["note"]) for r in rows) + 2
    header = f"{'case':{col}}" + "".join(f"k={k:<6}" for k in k_values)
    print(header)
    print("-" * len(header))
    for row in rows:
        marks = "".join(f"{'PASS':<8}" if row[f'k={k}'] else f"{'FAIL':<8}" for k in k_values)
        print(f"{row['note']:{col}}{marks}")
    print()
    for k in k_values:
        passed = sum(hits_by_k[k])
        total = len(hits_by_k[k])
        print(f"recall@{k}: {passed}/{total} ({passed / total:.0%})")


def main():
    print("=== Layer 1: structural checks (no embeddings, no LLM) ===")
    for check in STRUCTURAL_CHECKS:
        print(f"  PASS -- {check()}")
    print()

    print("=== Layer 2: retrieval eval (recall@k against the live vector store) ===")
    print("(run `python src/ingest.py` first if the store doesn't reflect the current corpus)\n")
    vector_store = get_vector_store()
    rows, hits_by_k = run_retrieval_eval(vector_store)
    print_report(rows, hits_by_k)

    k4_recall = sum(hits_by_k[PRODUCTION_K]) / len(hits_by_k[PRODUCTION_K])
    print()
    if k4_recall < 1.0:
        print(f"FAILED: recall@{PRODUCTION_K} is {k4_recall:.0%} -- below the 100% bar expected for this small, curated corpus.")
        sys.exit(1)
    print(f"All checks passed. recall@{PRODUCTION_K} = {k4_recall:.0%}.")


if __name__ == "__main__":
    main()
