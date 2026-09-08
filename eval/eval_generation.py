"""
Generation-quality eval for chains.py: does the FINAL generated answer state the
correct facts, not just "did retrieval find the right chunk" (that's already
covered by eval_chunking.py's Layer 2). Retrieval succeeding is necessary but
not sufficient -- the model could still retrieve the right chunk and then
misstate the number, hedge incorrectly, or answer from its own (out-of-date)
training data instead of the provided context.

Makes real LLM calls (unlike eval_chunking.py, which only needs embeddings) --
slower and has a small cost, so this is a separate eval rather than folded into
the retrieval layer.

Run:
    python eval/eval_generation.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from chains import build_rag_chain  # noqa: E402
from eval_history import record_run  # noqa: E402


def declined_without_fabricating(question: str, answer: str) -> bool:
    """For the out-of-corpus case: three straight heuristic attempts failed
    here, each for a different reason -- one exact refusal phrase misses valid
    phrasing variance ("doesn't contain" vs "does not contain", "can't answer"
    vs "isn't something I can answer"), and a bare "no dollar figure anywhere"
    regex false-positives on the model legitimately citing REAL in-corpus
    prices (e.g. Audi's $6.6M) as examples while explaining what it doesn't
    have. This is exactly the ambiguity rag-demo/redteam/grading_graph.py
    already solved with an LLM judge instead of continuing to patch string
    heuristics -- so use one here too."""
    from chains import extract_text, get_llm

    llm = get_llm()
    prompt = (
        "A RAG assistant was asked a question outside its knowledge base "
        "(F1 Fantasy 2026 data only) and gave the answer below. Reply with "
        "exactly one word: PASS if it correctly declined to answer without "
        "inventing a real-looking fact specific to the question's actual "
        "subject, or FAIL if it fabricated such a fact (citing real numbers "
        "from its OWN corpus, e.g. F1 prices, as illustrative context is fine "
        "and still a PASS).\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    verdict = extract_text(llm.invoke(prompt).content).strip().upper()
    return verdict.startswith("PASS")


# (question, check -- a list of required substrings (ALL must appear, case-
#  insensitive) or a callable(question, answer) -> bool for a check too
#  open-ended for substring matching, note)
EVAL_CASES = [
    ("What is Lewis Hamilton's fantasy price?", ["25.1"], "price lookup, answer must state the number"),
    ("How much does the Cadillac constructor cost?", ["3.0"], "constructor price lookup"),
    ("What is the sprint DNF penalty?", ["-10", "10"], "must give sprint-specific number, not the generic -20"),
    ("What is the DNF penalty in a normal Grand Prix?", ["-20", "20"], "must give race-specific number, not sprint's -10"),
    ("How many free transfers do I get per race weekend?", ["two", "2"], "rules lookup"),
    ("What does the Wildcard chip do?", ["unlimited"], "rules lookup, must mention unlimited transfers"),
    ("Who currently leads the drivers' championship?", ["Antonelli"], "standings lookup"),
    ("Who won the most recent Grand Prix?", ["Antonelli"], "recent form lookup"),
    ("Is Monza a low-downforce or high-downforce circuit?", ["low"], "circuit fact, must not say high"),
    ("What was Anthropic's Q2 2026 operating profit?", declined_without_fabricating, "OUT-OF-CORPUS: must not invent a number"),
]


def run_eval():
    chain, _ = build_rag_chain(k=4)
    results = []
    for question, required, note in EVAL_CASES:
        answer = chain.invoke(question)
        if callable(required):
            hit = required(question, answer)
        else:
            answer_lower = answer.lower()
            checker = any if len(required) > 1 else all
            hit = checker(req.lower() in answer_lower for req in required)
        results.append((question, note, hit, answer))
    return results


def main():
    print("=== Generation-quality eval: does the final answer state the right facts? ===\n")
    results = run_eval()
    passed = 0
    for question, note, hit, answer in results:
        status = "PASS" if hit else "FAIL"
        passed += hit
        print(f"[{status}] {note}")
        print(f"  Q: {question}")
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print()

    record_run("eval_generation", passed, len(results))
    print(f"{passed}/{len(results)} passed")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
