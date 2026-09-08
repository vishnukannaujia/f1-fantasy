"""
Empirical test of cross-encoder reranking: does adding a second-stage reranker
on top of the current bi-encoder retrieval actually improve recall on THIS
corpus? Persists what was originally a one-off comparison (see EVALS.md's
"Cross-encoder reranking -- tested, and explicitly NOT added" decision) as a
standing, re-runnable script, so the rejection can be re-verified any time the
corpus changes materially -- a rejected-with-evidence result deserves the same
re-verifiability as an adopted one (see the rag-eval-bakeoff skill).

Protocol: retrieve top-10 via the current production bi-encoder
(BAAI/bge-large-en-v1.5), rerank with a general-purpose cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2), compare recall@1 before/after on the
same 15-query eval set from eval_chunking.py. If reranking fixes one case but
breaks a different previously-correct one (net-zero or worse), that's strong
evidence to keep rejecting it for this corpus -- print the specific broken
case, don't just report the aggregate number.

Requires the `sentence-transformers` CrossEncoder class (already a
transitive dependency via sentence-transformers, used for reranking here
rather than embedding).

Run:
    python eval/eval_reranking.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from sentence_transformers import CrossEncoder

from ingest import get_vector_store  # noqa: E402
from eval_chunking import EVAL_CASES  # noqa: E402

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_TOP_N = 10  # bi-encoder candidate pool handed to the reranker
PRODUCTION_K = 1  # recall@1 is the axis reranking would actually move -- recall@4/8
# are already saturated at 100% for the current setup, so a net change there isn't
# possible to observe; see eval_embedding_bakeoff.py's same note.


def hit(required, text) -> bool:
    text_lower = text.lower()
    checker = any if len(required) > 1 else all
    return checker(r.lower() in text_lower for r in required)


def main():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_N})
    reranker = CrossEncoder(RERANKER_MODEL)

    print(f"=== Reranking eval: bi-encoder top-{RETRIEVE_TOP_N} + {RERANKER_MODEL} vs. bi-encoder alone ===\n")
    print("(run `python src/ingest.py` first if the store doesn't reflect the current corpus)\n")

    before_hits, after_hits = 0, 0
    changed_cases = []

    for query, required, note in EVAL_CASES:
        candidates = retriever.invoke(query)
        before_top1 = candidates[0].page_content if candidates else ""
        before_ok = hit(required, before_top1)
        before_hits += before_ok

        pairs = [(query, c.page_content) for c in candidates]
        scores = reranker.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, candidates), key=lambda p: -p[0])]
        after_top1 = reranked[0].page_content if reranked else ""
        after_ok = hit(required, after_top1)
        after_hits += after_ok

        status = "same" if before_ok == after_ok else ("FIXED" if after_ok else "BROKE")
        if status != "same":
            changed_cases.append((note, query, status, before_top1[:150], after_top1[:150]))
        print(f"[{status:<5}] {note}  (before={'PASS' if before_ok else 'FAIL'}, after={'PASS' if after_ok else 'FAIL'})")

    total = len(EVAL_CASES)
    print(f"\nrecall@1 before reranking: {before_hits}/{total} ({before_hits/total:.0%})")
    print(f"recall@1 after  reranking: {after_hits}/{total} ({after_hits/total:.0%})")

    if changed_cases:
        print("\nCases where reranking changed the outcome (inspect these before deciding anything):")
        for note, query, status, before_text, after_text in changed_cases:
            print(f"\n  [{status}] {note}")
            print(f"    query: {query!r}")
            print(f"    top-1 BEFORE rerank: {before_text!r}")
            print(f"    top-1 AFTER  rerank: {after_text!r}")
    else:
        print("\nNo case changed outcome either direction.")

    net_change = after_hits - before_hits
    print(f"\nNet recall@1 change: {'+' if net_change >= 0 else ''}{net_change} case(s)")
    if net_change <= 0:
        print(
            "Net-zero-or-negative result: consistent with the existing EVALS.md finding that a "
            "general-purpose reranker is not a clean win on this narrow, jargon-dense corpus. "
            "Inspect the changed cases above for the specific mechanism before considering reranking again."
        )
    else:
        print(
            "Net-positive result -- this CONTRADICTS the previously-documented rejection. Before adopting "
            "reranking, root-cause why this differs from the original test (corpus change? different "
            "reranker version? different candidate pool size?) and update EVALS.md either way."
        )


if __name__ == "__main__":
    main()
