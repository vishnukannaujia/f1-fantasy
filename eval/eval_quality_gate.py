"""
Eval for chains.py's quality gate: the similarity-score-threshold retriever
and the deterministic "skip the LLM entirely" short-circuit when nothing
scores above threshold (see DEFAULT_SCORE_THRESHOLD in chains.py for how that
number was picked).

Two layers:
  1. Retriever-level (no LLM call, fast) -- clearly off-topic queries should
     retrieve zero docs; legitimate and borderline-but-topically-adjacent
     queries should still retrieve some.
  2. Chain-level (still no LLM call for the off-topic cases specifically,
     since the gate skips it) -- confirms the full wiring: an off-topic
     question gets NOTHING_FOUND_MESSAGE verbatim, deterministically, not
     just "docs came back empty at the retriever".

Run:
    python eval/eval_quality_gate.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from chains import NOT_FOUND_MESSAGE, build_rag_chain  # noqa: E402
from eval_history import record_run  # noqa: E402

# (query, expect_docs -- True if this should retrieve something, False if the
#  gate should fire, note)
CASES = [
    ("What is Lando Norris fantasy price?", True, "legitimate: driver price"),
    ("What is the sprint DNF penalty?", True, "legitimate: rules"),
    ("Is Monza a low downforce circuit?", True, "legitimate: circuit notes"),
    ("What is Formula E?", False, "off-topic: unrelated racing series"),
    ("What is the capital of France?", False, "off-topic: unrelated entirely"),
    ("How do I bake a chocolate cake?", False, "off-topic: unrelated entirely"),
    ("What was the 2025 Italian Grand Prix result?", True, "borderline: topically adjacent, factually absent -- should still reach the LLM, which is expected to notice the gap itself (see eval_generation.py for that check)"),
]


def main():
    print("=== Quality gate eval (chains.py) ===\n")
    chain, retriever = build_rag_chain(k=4)

    passed = 0
    for query, expect_docs, note in CASES:
        docs = retriever.invoke(query)
        got_docs = len(docs) > 0
        retriever_ok = got_docs == expect_docs

        # Chain-level check only meaningful (and only free) for the off-topic
        # cases -- the gate means no LLM call happens, so we can assert the
        # exact deterministic message. For expect_docs=True cases we don't
        # invoke the chain here (that's what eval_generation.py already
        # covers, at real LLM cost) -- this eval stays fast and free.
        if not expect_docs:
            answer = chain.invoke(query)
            chain_ok = answer == NOT_FOUND_MESSAGE
        else:
            chain_ok = True  # not checked here, by design

        ok = retriever_ok and chain_ok
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {note}")
        print(f"  Q: {query}")
        print(f"  retrieved {len(docs)} docs (expected {'>0' if expect_docs else '0'})")
        if not expect_docs:
            print(f"  chain returned NOT_FOUND_MESSAGE verbatim: {chain_ok}")
        print()

    record_run("eval_quality_gate", passed, len(CASES))
    print(f"{passed}/{len(CASES)} passed")
    if passed < len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    main()
