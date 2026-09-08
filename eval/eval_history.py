"""
Shared helper: append a one-line summary of each eval script's run to
eval_history.jsonl, so pass-rate over time is trackable across sessions
without a full CI pipeline. See agent-harness-components skill, component 6
(Evals): "a previously-100%-passing suite that silently drops is worth
investigating immediately, not batching into later cleanup" -- this makes
that drop visible in the file's own history, across sessions, rather than
only in whichever single terminal happened to run the suite that day.

Intentionally lightweight: a local, git-committed, append-only log, not a
dashboard or an alerting system -- proportionate to a personal-scale project.
This does NOT replace actually re-running the suite (that's still what
catches a regression); it only makes the pattern over time visible
afterward.

Usage: call record_run(script_name, passed, total) once at the end of an
eval script's main(), before the pass/fail exit branch, so a run gets logged
whether it passed or failed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent / "eval_history.jsonl"


def record_run(script: str, passed: int, total: int) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else None,
    }
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
