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
