"""
Shared tracing wiring for running LangSmith and Langfuse side by side on the
same real calls, so they can be compared directly rather than on different
runs.

The two integrate differently, which is why this module exists instead of
each caller repeating the logic:
  - LangSmith is fully automatic once LANGSMITH_TRACING=true and
    LANGSMITH_API_KEY are set -- LangChain auto-instruments itself, no code
    change needed at any call site.
  - Langfuse needs an explicit CallbackHandler attached to each chain/graph
    invocation (`config={"callbacks": [...]}`), so this module centralizes
    building that handler once.

Both are opt-in via env vars and off by default -- tracing stays fully absent
with zero config, matching the project's existing LANGSMITH_TRACING pattern.
"""

import contextlib
import os


def get_langfuse_handler():
    """Returns a Langfuse CallbackHandler if LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are both set, else None."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def langfuse_callbacks() -> list:
    """A list suitable for `config={"callbacks": langfuse_callbacks()}` --
    empty (a no-op) when Langfuse isn't configured."""
    handler = get_langfuse_handler()
    return [handler] if handler else []


def langsmith_session_metadata(session_id: str) -> dict:
    """LangSmith has no first-class 'session' field the way Langfuse does --
    its own "session" is actually just its old name for "project". The
    equivalent here is ordinary searchable metadata: filter the LangSmith UI
    on `metadata.session_id = <value>` to find the matching trace."""
    return {"session_id": session_id}


def langfuse_session(session_id: str):
    """Context manager to wrap an invoke() call so every observation Langfuse
    records inside it (including from the LangChain CallbackHandler) is
    tagged with the same session_id -- a real no-op (contextlib.nullcontext)
    when Langfuse isn't configured, not a conditional branch callers need to
    write themselves.

    Usage:
        session_id = str(uuid.uuid4())
        with langfuse_session(session_id):
            result = app.invoke(inputs, config={
                "metadata": langsmith_session_metadata(session_id),
                "callbacks": langfuse_callbacks(),
            })
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return contextlib.nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(session_id=session_id)
