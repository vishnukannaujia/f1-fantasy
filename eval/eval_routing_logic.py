"""
Unit-level checks for web/app.py's routing contract: is_team_building_request
decides whether a chat message goes to the fast plain Q&A chain or the slow
(live web search + LLM) team_builder.py graph. No LLM calls, no network,
deterministic, runs in milliseconds -- this is component 1 (Scope & task
contract) of the agent-harness-components skill: a routing bug here silently
sends users to the wrong (much slower, or much less capable) half of the app
with no visible error.

Run:
    python eval/eval_routing_logic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "web"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import is_team_building_request  # noqa: E402
from eval_history import record_run  # noqa: E402

CHECKS_RUN, CHECKS_FAILED = 0, []


def check(name, condition, detail=""):
    global CHECKS_RUN
    CHECKS_RUN += 1
    if condition:
        print(f"  PASS -- {name}")
    else:
        CHECKS_FAILED.append(name)
        print(f"  FAIL -- {name}{': ' + detail if detail else ''}")


TEAM_BUILDING_MESSAGES = [
    "Build me a team for the Spanish Grand Prix at Madrid.",
    "Who should I captain this week?",
    "Recommend a lineup for this race",
    "What team formation should I pick?",
    "give me a roster for this weekend",
]

QA_MESSAGES = [
    "What is George Russell's price?",
    "What does the Wildcard chip do?",
    "How many free transfers are allowed?",
    "Who leads the drivers championship?",
    "What is the DNF penalty in a normal Grand Prix?",
    "Which team is favorite at Monza?",
    "What team is George Russell on?",
]


def test_team_building_messages_route_to_team_builder():
    for msg in TEAM_BUILDING_MESSAGES:
        check(f"routes to team_builder: {msg!r}", is_team_building_request(msg) is True)


def test_plain_qa_messages_route_to_qa():
    for msg in QA_MESSAGES:
        check(f"routes to qa: {msg!r}", is_team_building_request(msg) is False)


def test_known_false_positive_team_and_for_coincidence():
    # KNOWN CONTRACT GAP, not a regression: the keyword regex matches "team"
    # anywhere followed later by "for"/"pick"/"build" ANYWHERE after it, with no
    # requirement that they're part of the same phrase. A plain factual question
    # that happens to contain both words gets misrouted to the slow
    # team_builder graph. Asserting the CURRENT (undesirable) behavior here
    # rather than silently tightening the regex -- this test's job is to make
    # the gap visible and catch it if it gets worse (e.g. a regex "fix" that
    # accidentally makes false positives MORE common), not to pretend it's
    # fixed. See ARCHITECTURE.md known gaps.
    msg = "What team does Max Verstappen drive for?"
    check(
        "KNOWN GAP: 'team...for' coincidence misroutes a plain factual question to team_builder",
        is_team_building_request(msg) is True,
        "if this now returns False, the regex was tightened -- update this test's expectation, don't just delete it",
    )


def test_empty_and_whitespace_only():
    check("empty string -> not team-building", is_team_building_request("") is False)
    check("whitespace-only -> not team-building", is_team_building_request("   ") is False)


def main():
    print("=== web/app.py routing-contract unit checks (no LLM calls) ===\n")
    for fn in [
        test_team_building_messages_route_to_team_builder,
        test_plain_qa_messages_route_to_qa,
        test_known_false_positive_team_and_for_coincidence,
        test_empty_and_whitespace_only,
    ]:
        fn()

    record_run("eval_routing_logic", CHECKS_RUN - len(CHECKS_FAILED), CHECKS_RUN)
    print(f"\n{CHECKS_RUN - len(CHECKS_FAILED)}/{CHECKS_RUN} checks passed")
    if CHECKS_FAILED:
        print(f"FAILED: {CHECKS_FAILED}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
