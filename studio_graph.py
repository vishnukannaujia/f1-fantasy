"""
Module-level compiled-graph export for LangGraph Studio (langgraph dev reads
this via langgraph.json's "graphs" mapping). Kept separate from
src/team_builder.py rather than adding a module-level `graph = build_graph()`
there -- team_builder.py is also imported by eval/ scripts and the web app
purely for its functions (parsing, validation), and building the graph eagerly
on every such import is an unnecessary side effect those callers don't need.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from team_builder import build_graph  # noqa: E402

graph = build_graph()
