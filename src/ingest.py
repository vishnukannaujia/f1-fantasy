"""
Build the vector store for the F1 Fantasy RAG assistant.

Loads the plain-text source documents in data/raw/ (current F1 2026 season data:
fantasy scoring rules, driver/constructor prices, standings, recent form, and
circuit notes for the upcoming race), chunks them, embeds them with a local
sentence-transformer model, and persists them into a Chroma vector store.

Two chunking strategies, chosen by content shape (see ARCHITECTURE.md):
  - Prose files (rules, form notes, circuit notes) split on blank-line paragraph
    boundaries directly, one chunk per natural paragraph. RecursiveCharacterText-
    Splitter was tried first and rejected: it always *merges* adjacent short
    splits back together up to chunk_size, so no single fixed chunk_size gave a
    clean 1-paragraph-per-chunk mapping across files whose paragraphs are
    naturally different lengths (measured: ~265 chars/paragraph in the rules
    file vs. ~450-500 in the prose notes) -- either short rule sections got
    merged together (chunk_size=800) or long prose paragraphs got fragmented
    mid-thought (chunk_size=400).
    The fallback cap for an oversized paragraph (determine_paragraph_max_len)
    is *derived from the corpus itself* rather than a hand-picked constant --
    it measures the single longest real paragraph across all prose files and
    pads it, so it adapts automatically as the corpus grows (new circuit notes,
    longer paragraphs) instead of silently going stale like a hardcoded number
    would.
  - The two flat price-list files (ROW_CHUNK_FILES) use row-level chunking
    instead -- one chunk per driver/constructor line -- because a single
    price lookup should match one precise small chunk, not an arbitrary
    slice of an unrelated 22-line list.

Run once before chains.py / team_builder.py, and re-run whenever data/raw/ changes
(e.g. after a race weekend, when prices and standings update):
    python src/ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from f1_data import parse_constructor_prices, parse_driver_prices

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PERSIST_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "f1_fantasy_2026"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim, retrieval-tuned (asymmetric query/passage), was all-mpnet-base-v2 (768-dim, symmetric)

ROW_CHUNK_FILES = {"02_driver_prices.txt", "03_constructor_prices.txt"}

# Bounds for determine_paragraph_max_len: MARGIN pads the observed longest
# paragraph so ordinary length variance doesn't trip the fallback splitter on
# every ingest; FLOOR stops a corpus of unusually short files (e.g. only
# one-line price notes) from computing a pathologically tiny, over-eager cap.
PARAGRAPH_MAX_LEN_MARGIN = 1.25
PARAGRAPH_MAX_LEN_FLOOR = 600


def load_prose_documents(source_dir=DATA_DIR):
    docs = []
    for path in sorted(Path(source_dir).glob("*.txt")):
        if path.name not in ROW_CHUNK_FILES:
            docs.extend(TextLoader(str(path)).load())
    return docs


def determine_paragraph_max_len(
    prose_docs,
    margin: float = PARAGRAPH_MAX_LEN_MARGIN,
    floor: int = PARAGRAPH_MAX_LEN_FLOOR,
) -> int:
    """Derive the paragraph-splitter's fallback cap FROM the actual corpus,
    instead of a hand-picked constant: find the single longest real paragraph
    across all prose files and pad it by `margin`, floored at `floor`. Re-run
    automatically on every ingest, so it stays correct as content changes --
    a new circuit-notes file with a longer paragraph raises the cap on its own,
    no manual re-tuning needed."""
    longest = 0
    for doc in prose_docs:
        for para in doc.page_content.split("\n\n"):
            para = para.strip()
            if para:
                longest = max(longest, len(para))
    return max(int(longest * margin), floor)


def split_into_paragraphs(doc, max_len: int):
    """One chunk per blank-line-separated paragraph, no merging of adjacent
    short paragraphs. A paragraph that individually exceeds max_len falls back
    to recursive character splitting."""
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=max_len, chunk_overlap=100)
    chunks = []
    for para in doc.page_content.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            chunks.append(Document(page_content=para, metadata=doc.metadata))
        else:
            chunks.extend(
                Document(page_content=sub, metadata=doc.metadata)
                for sub in fallback_splitter.split_text(para)
            )
    return chunks


def build_row_chunks(source_dir=DATA_DIR):
    """One Document per driver/constructor price line, prefixed with a label so
    each tiny chunk is self-contained when retrieved on its own."""
    chunks = []
    driver_file = Path(source_dir) / "02_driver_prices.txt"
    if driver_file.exists():
        for name, info in parse_driver_prices(driver_file).items():
            text = f"F1 Fantasy 2026 driver price: {name} ({info['team']}): ${info['price']}M"
            chunks.append(Document(page_content=text, metadata={"source": str(driver_file)}))
    constructor_file = Path(source_dir) / "03_constructor_prices.txt"
    if constructor_file.exists():
        for team, price in parse_constructor_prices(constructor_file).items():
            text = f"F1 Fantasy 2026 constructor price: {team}: ${price}M"
            chunks.append(Document(page_content=text, metadata={"source": str(constructor_file)}))
    return chunks


def validate_chunks(chunks, max_len: int, label: str = "chunks"):
    """Cross-check the chunking output against the cap that produced it. Any
    empty chunk, or any chunk exceeding max_len, means the splitter logic has a
    bug (the fallback splitter is configured with chunk_size=max_len, so it
    should be structurally impossible for a chunk to exceed it) -- fail loudly
    at ingest time rather than silently shipping a broken chunk that only
    surfaces later as a bad retrieval. Also prints the achieved length
    distribution so drift is visible on every run."""
    if not chunks:
        raise SystemExit(f"No {label} produced -- check the source files.")

    lengths = [len(c.page_content) for c in chunks]
    empty = [c for c, n in zip(chunks, lengths) if n == 0]
    oversized = [c for c, n in zip(chunks, lengths) if n > max_len]

    if empty:
        raise SystemExit(f"{len(empty)} empty {label} produced -- check the splitter logic.")
    if oversized:
        worst = max(oversized, key=lambda c: len(c.page_content))
        raise SystemExit(
            f"{len(oversized)} {label} exceed the {max_len}-char cap "
            f"(worst: {len(worst.page_content)} chars, source={worst.metadata.get('source', '?')}) "
            f"-- the fallback splitter should guarantee this never happens; investigate."
        )

    print(
        f"  [validate_chunks] {len(chunks)} {label}: min={min(lengths)} max={max(lengths)} "
        f"mean={sum(lengths) / len(lengths):.0f} chars, all within the {max_len}-char cap"
    )


def build_vector_store(
    source_dir=DATA_DIR,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
):
    prose_docs = load_prose_documents(source_dir)
    row_chunks = build_row_chunks(source_dir)
    if not prose_docs and not row_chunks:
        raise SystemExit(f"No .txt documents found in {source_dir}")

    max_len = determine_paragraph_max_len(prose_docs)
    print(f"Dynamic paragraph cap: {max_len} chars (derived from the longest real paragraph)")

    prose_chunks = []
    for doc in prose_docs:
        prose_chunks.extend(split_into_paragraphs(doc, max_len=max_len))
    validate_chunks(prose_chunks, max_len, label="prose chunks")
    validate_chunks(row_chunks, max_len, label="row chunks")

    chunks = prose_chunks + row_chunks
    print(
        f"Loaded {len(prose_docs)} prose documents -> split into {len(prose_chunks)} chunks; "
        f"{len(row_chunks)} row-level chunks from {len(ROW_CHUNK_FILES)} tabular files"
    )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Chroma.from_documents() ADDS to an existing collection of the same name
    # rather than replacing it -- every prior re-ingest during this project
    # silently duplicated the whole corpus on top of what was already there
    # (discovered: 167 stored vectors for what should have been 59, including
    # 108 entries still carrying the pre-rename "f1-fantasy-rag" path).
    # Delete first so re-ingesting is idempotent, not additive.
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    ).delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print(f"Persisted {len(chunks)} chunks to Chroma at {persist_directory}")
    return vector_store


def get_vector_store(persist_directory=PERSIST_DIR, collection_name=COLLECTION_NAME):
    """Load an existing persisted vector store (assumes ingest already ran)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    build_vector_store()
