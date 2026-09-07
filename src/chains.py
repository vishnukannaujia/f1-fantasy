"""
Plain retrieval-augmented Q&A chain for single-fact F1 Fantasy questions
("what's Hamilton's price?", "what does the DRS Boost do?").

This is the right tool for single-fact lookups: top-k similarity search over the
Chroma store built by ingest.py. It is NOT the right tool for team-building
(picking a full 5-driver + 2-constructor roster under the $100M budget cap) --
that needs the *entire* price table simultaneously, not a similarity-ranked
subset of chunks. See team_builder.py and ARCHITECTURE.md for that path.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from ingest import get_vector_store
from observability import langfuse_callbacks

load_dotenv()

MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def get_llm():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # Without an explicit max_tokens, langchain_anthropic's low default leaves no
    # room once the model spends part of its budget on extended thinking for a
    # complex prompt -- observed hitting stop_reason="max_tokens" with a
    # completely EMPTY visible response (team_builder.py hit this first; then
    # eval/learnings_loop.py hit the same thing calling this same function with
    # a longer analysis prompt). Any caller doing more than build_rag_chain's
    # short Q&A prompts needs this headroom, so it's the default here, not
    # something each caller has to remember to set.
    return ChatAnthropic(model=MODEL_NAME, max_tokens=8192)


def extract_text(content) -> str:
    """AIMessage.content is a plain str, or a list of content blocks (e.g. an
    extended-thinking block alongside the text block) depending on whether the
    model reasoned before answering. Normalize either shape to plain text."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


RAG_SYSTEM_PROMPT = (
    "You are an F1 Fantasy assistant answering questions using retrieved context "
    "about the current 2026 F1 season: fantasy scoring rules, driver/constructor "
    "prices, championship standings, recent form, and circuit notes. Answer ONLY "
    "using the context below. Cite specific numbers (prices, points, positions) "
    "where relevant. If the context does not contain the answer, say so explicitly "
    "instead of guessing -- in particular, don't guess at current prices, "
    "standings, or race results from your own training data, since those are "
    "certainly out of date.\n\nContext:\n{context}"
)


NOT_FOUND_MESSAGE = (
    "I don't have any information relevant to that question in the F1 Fantasy 2026 "
    "knowledge base (rules, prices, standings, recent form, circuit notes). This might be "
    "outside what this assistant covers, or phrased in a way that didn't match anything -- "
    "try rephrasing, or ask about F1 Fantasy rules/prices/standings/circuit notes directly."
)

DEFAULT_SCORE_THRESHOLD = 0.60  # relevance score = 1 - cosine_distance; see ARCHITECTURE.md
# for how this number was picked: measured real distance scores for the 15-query eval set
# (legitimate worst case 0.363) against clearly off-topic probes (0.44+), threshold sits in
# the gap. This is a COARSE pre-filter for obviously unrelated questions, not a fact-checker
# -- a topically-adjacent-but-factually-absent question (e.g. asking about a season this
# corpus doesn't cover) scores well inside the "legitimate" range and correctly still reaches
# the LLM, which is the layer actually equipped to notice the fact isn't there.


def build_rag_chain(
    k: int = 4,
    vector_store=None,
    system_prompt: str = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """Retrieval-augmented chain: retrieve top-k chunks above score_threshold, then answer
    grounded in them. Below-threshold retrieval short-circuits before the LLM call entirely
    (see NOT_FOUND_MESSAGE) -- a deterministic gate, not just a prompt instruction hoping the
    model notices empty/irrelevant context on its own."""
    llm = get_llm()
    vector_store = vector_store if vector_store is not None else get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    def answer(question: str) -> str:
        docs = retriever.invoke(question)
        if not docs:
            print(f"  [quality gate] nothing scored above {score_threshold} for {question!r} -- skipping LLM call")
            return NOT_FOUND_MESSAGE
        messages = prompt.invoke({"context": format_docs(docs), "question": question})
        return extract_text(llm.invoke(messages).content)

    rag_chain = RunnableLambda(answer).with_config(callbacks=langfuse_callbacks())
    # LangSmith needs no wiring here -- it's fully automatic via env vars once
    # the langsmith package is installed, and traces the retriever.invoke() /
    # llm.invoke() calls made inside `answer` individually even though they're
    # called imperatively rather than composed via the `|` operator. Langfuse
    # needs this explicit CallbackHandler attached, which is why the two look
    # different here even though both trace every call this chain makes.
    return rag_chain, retriever
