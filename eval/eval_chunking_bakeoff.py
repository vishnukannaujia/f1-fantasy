"""
Empirical bake-off: test every chunking strategy from the general-knowledge
survey against THIS corpus, using the SAME 15-query eval set from
eval_chunking.py, holding everything else constant (embedding model, row-level
chunking for the price files, retrieval k values). Only the PROSE-file
chunking strategy varies between runs -- that's the one open design choice
this comparison is meant to settle empirically instead of by assertion.

Row/record-level chunking isn't included as a prose variant: it's the correct,
already-adopted strategy for the two flat price files specifically (each line
is independently meaningful), not a general-purpose prose strategy, and it's
held constant (applied identically) in every run below.

Run:
    python eval/eval_chunking_bakeoff.py
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import (  # noqa: E402
    DATA_DIR,
    build_row_chunks,
    determine_paragraph_max_len,
    load_prose_documents,
    split_into_paragraphs,
)
from eval_chunking import EVAL_CASES, K_VALUES  # noqa: E402

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _sentences(text: str) -> list:
    sentences = []
    for line in text.replace("\n\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for sent in re.split(r"(?<=[.:])\s+", line):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


# ---------- Strategy implementations (each: list[Document] -> list[Document]) ----------


def strategy_fixed(prose_docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(prose_docs)


def strategy_token_based(prose_docs, chunk_size_tokens=150, overlap_tokens=20):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens, chunk_overlap=overlap_tokens, encoding_name="cl100k_base"
    )
    return splitter.split_documents(prose_docs)


def strategy_sentence(prose_docs):
    chunks = []
    for doc in prose_docs:
        for sent in _sentences(doc.page_content):
            chunks.append(Document(page_content=sent, metadata=doc.metadata))
    return chunks


def strategy_structure_aware_current(prose_docs):
    """The production approach: split on the file's own paragraph structure,
    dynamic corpus-derived cap, no merging of adjacent short paragraphs."""
    max_len = determine_paragraph_max_len(prose_docs)
    chunks = []
    for doc in prose_docs:
        chunks.extend(split_into_paragraphs(doc, max_len=max_len))
    return chunks


def strategy_semantic(prose_docs, breakpoint_percentile=85):
    """Embed each sentence, cut where consecutive-sentence cosine distance
    spikes above the given percentile (the standard 'semantic chunking'
    breakpoint method)."""
    import numpy as np

    chunks = []
    for doc in prose_docs:
        sentences = _sentences(doc.page_content)
        if len(sentences) < 3:
            chunks.append(Document(page_content=doc.page_content.strip(), metadata=doc.metadata))
            continue
        vectors = np.array(_embeddings.embed_documents(sentences))
        norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        sims = np.sum(norms[:-1] * norms[1:], axis=1)
        distances = 1 - sims
        threshold = np.percentile(distances, breakpoint_percentile)
        breakpoints = set(i + 1 for i, d in enumerate(distances) if d > threshold)

        start = 0
        for i in range(1, len(sentences) + 1):
            if i in breakpoints or i == len(sentences):
                chunks.append(Document(page_content=" ".join(sentences[start:i]), metadata=doc.metadata))
                start = i
    return chunks


def strategy_sliding_window(prose_docs, chunk_size=400, overlap=200):
    """Heavy overlap (50%) fixed-size window -- a variant of strategy_fixed
    but distinct enough (context duplicated across many chunks) to test
    separately."""
    return strategy_fixed(prose_docs, chunk_size=chunk_size, overlap=overlap)


def strategy_agentic(prose_docs):
    """An LLM proposes chunk boundaries directly, per document."""
    from chains import extract_text, get_llm

    llm = get_llm()
    chunks = []
    for doc in prose_docs:
        prompt = (
            "Split the following document into semantically coherent chunks -- each chunk should "
            "represent one complete idea or fact that would make sense retrieved on its own. "
            "Reply with ONLY the chunks, each separated by a line containing exactly '---', no other "
            "commentary.\n\nDocument:\n" + doc.page_content
        )
        text = extract_text(llm.invoke(prompt).content)
        for part in text.split("---"):
            part = part.strip()
            if part:
                chunks.append(Document(page_content=part, metadata=doc.metadata))
    return chunks


# ---------- Eval harness (reused pattern from eval_chunking.py) ----------


def run_eval_for_chunks(prose_chunks, row_chunks):
    all_chunks = prose_chunks + row_chunks
    vs = Chroma.from_documents(documents=all_chunks, embedding=_embeddings, collection_name=f"bakeoff_{id(prose_chunks)}")
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
    prose_docs = load_prose_documents(DATA_DIR)
    row_chunks = build_row_chunks(DATA_DIR)  # constant across every variant

    strategies = {
        "Fixed-size 400/60": lambda: strategy_fixed(prose_docs, 400, 60),
        "Fixed-size 800/120 (original default)": lambda: strategy_fixed(prose_docs, 800, 120),
        "Fixed-size 1200/150": lambda: strategy_fixed(prose_docs, 1200, 150),
        "Token-based (150/20 tokens)": lambda: strategy_token_based(prose_docs),
        "Sentence-based": lambda: strategy_sentence(prose_docs),
        "Structure-aware / paragraph (CURRENT)": lambda: strategy_structure_aware_current(prose_docs),
        "Semantic (embedding breakpoint)": lambda: strategy_semantic(prose_docs),
        "Sliding window 400/200 (50% overlap)": lambda: strategy_sliding_window(prose_docs),
        "Agentic (LLM-proposed)": lambda: strategy_agentic(prose_docs),
    }

    results = []
    for name, build_fn in strategies.items():
        print(f"Running: {name} ...")
        t0 = time.time()
        chunks = build_fn()
        build_seconds = time.time() - t0
        lengths = [len(c.page_content) for c in chunks]
        recall = run_eval_for_chunks(chunks, row_chunks)
        results.append({
            "name": name,
            "count": len(chunks),
            "min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths),
            "build_s": build_seconds,
            **{f"recall@{k}": recall[k] for k in K_VALUES},
        })

    print("\n" + "=" * 130)
    header = f"{'Strategy':<40}{'#chunks':<9}{'min':<6}{'max':<6}{'mean':<7}{'build(s)':<10}"
    for k in K_VALUES:
        header += f"{'r@'+str(k):<8}"
    print(header)
    print("-" * 130)
    for r in sorted(results, key=lambda r: -r[f"recall@{K_VALUES[-2]}"]):
        row = f"{r['name']:<40}{r['count']:<9}{r['min']:<6}{r['max']:<6}{r['mean']:<7.0f}{r['build_s']:<10.1f}"
        for k in K_VALUES:
            row += f"{r[f'recall@{k}']*100:<7.0f}%"
        print(row)


if __name__ == "__main__":
    main()
