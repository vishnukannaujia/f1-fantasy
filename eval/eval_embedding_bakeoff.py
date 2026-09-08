"""
Empirical bake-off: compare embedding models head-to-head on THIS corpus,
using the SAME 15-query eval set from eval_chunking.py and the SAME (current,
production) chunking strategy held constant -- only the embedding model
varies between runs. This persists what was originally a one-off comparison
(see EVALS.md's "Embedding model" decision) as a standing, re-runnable script,
so the conclusion (BAAI/bge-large-en-v1.5 wins) can be re-verified any time
the corpus changes materially, the same way eval_chunking_bakeoff.py already
lets the chunking-strategy conclusion be re-verified.

Three real candidates across a meaningful axis of variation (see the
rag-eval-bakeoff skill's references/embedding-and-transformers.md for why
these three specifically): a small general-purpose model, a larger
general-purpose model, and a retrieval-tuned (asymmetric query/passage) model
-- the current production choice.

Run:
    python eval/eval_embedding_bakeoff.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingest import (  # noqa: E402
    DATA_DIR,
    EMBEDDING_MODEL,
    build_row_chunks,
    determine_paragraph_max_len,
    load_prose_documents,
    split_into_paragraphs,
)
from eval_chunking import EVAL_CASES, K_VALUES  # noqa: E402

CANDIDATE_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "General-purpose, small (384-dim)"),
    ("sentence-transformers/all-mpnet-base-v2", "General-purpose, larger (768-dim)"),
    (EMBEDDING_MODEL, "Retrieval-tuned, asymmetric (1024-dim) -- CURRENT PRODUCTION"),
]


def build_current_strategy_chunks():
    """Hold chunking constant at the current production strategy (structure-
    aware/paragraph + row-level for tabular files) -- this bakeoff varies only
    the embedding model, not the chunking strategy (that's
    eval_chunking_bakeoff.py's job)."""
    prose_docs = load_prose_documents(DATA_DIR)
    max_len = determine_paragraph_max_len(prose_docs)
    prose_chunks = []
    for doc in prose_docs:
        prose_chunks.extend(split_into_paragraphs(doc, max_len=max_len))
    row_chunks = build_row_chunks(DATA_DIR)
    return prose_chunks + row_chunks


def run_eval_for_model(model_name, all_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, collection_name=f"embed_bakeoff_{abs(hash(model_name))}"
    )
    retrievers = {k: vs.as_retriever(search_kwargs={"k": k}) for k in K_VALUES}
    hits_by_k = {k: 0 for k in K_VALUES}
    for query, required, _note in EVAL_CASES:
        for k in K_VALUES:
            docs = retrievers[k].invoke(query)
            combined = "\n".join(d.page_content for d in docs).lower()
            checker = any if len(required) > 1 else all
            if checker(r.lower() in combined for r in required):
                hits_by_k[k] += 1
    return {k: hits_by_k[k] / len(EVAL_CASES) for k in K_VALUES}


def main():
    all_chunks = build_current_strategy_chunks()
    print(f"Chunking held constant: {len(all_chunks)} chunks (current production strategy)\n")

    results = []
    for model_name, description in CANDIDATE_MODELS:
        print(f"Running: {model_name} ({description}) ...")
        t0 = time.time()
        recall = run_eval_for_model(model_name, all_chunks)
        elapsed = time.time() - t0
        results.append({"model": model_name, "description": description, "elapsed": elapsed, **{f"recall@{k}": recall[k] for k in K_VALUES}})

    print("\n" + "=" * 130)
    header = f"{'Model':<42}{'Description':<58}{'embed(s)':<10}"
    for k in K_VALUES:
        header += f"{'r@'+str(k):<8}"
    print(header)
    print("-" * 130)
    for r in sorted(results, key=lambda r: -r["recall@1"]):
        row = f"{r['model']:<42}{r['description']:<58}{r['elapsed']:<10.1f}"
        for k in K_VALUES:
            row += f"{r[f'recall@{k}']*100:<7.0f}%"
        print(row)

    print(
        "\nNote: recall@4/8 typically saturate near 100% for all candidates on a corpus this small -- "
        "recall@1 is where these models actually differentiate (see the rag-eval-bakeoff skill: "
        "chunking-strategy comparisons shouldn't over-index on recall@1, but embedding-model comparisons "
        "should, since that's the axis where semantic quality differences actually show up)."
    )


if __name__ == "__main__":
    main()
