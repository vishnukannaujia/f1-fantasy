"""
Unit-level checks for team_builder.py's hand-written, non-LLM logic: parsing the
model's structured proposal text, resolving names against the real price
tables, and the budget/roster-shape validation that gates whether a proposal is
accepted or sent back for a retry.

This is the layer eval_chunking.py calls "structural checks" applied to
team_builder.py instead of ingest.py: no LLM calls, no network, deterministic,
runs in milliseconds -- and arguably more important to have than the retrieval
eval, since a parsing or budget-math bug here would silently corrupt every
team recommendation regardless of how good the underlying LLM reasoning is.

Run:
    python eval/eval_team_builder_logic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from team_builder import (  # noqa: E402
    BUDGET_CAP,
    MAX_RETRIES,
    _lookup,
    _parse_proposal,
    route_after_validation,
    validate_budget,
)

FAKE_DRIVER_PRICES = {
    "Lando Norris": {"team": "McLaren", "price": 26.4},
    "Kimi Antonelli": {"team": "Mercedes", "price": 26.0},
    "Fernando Alonso": {"team": "Aston Martin", "price": 6.8},
    "Valtteri Bottas": {"team": "Cadillac", "price": 3.0},
    "Lance Stroll": {"team": "Aston Martin", "price": 3.0},
    "Max Verstappen": {"team": "Red Bull Racing", "price": 27.5},
}
FAKE_CONSTRUCTOR_PRICES = {
    "McLaren Mastercard F1 Team": 31.3,
    "Cadillac F1 Team": 3.0,
    "Scuderia Ferrari HP": 26.9,
}

VALID_PROPOSAL = """
DRIVERS:
1. Lando Norris
2. Kimi Antonelli
3. Fernando Alonso
4. Valtteri Bottas
5. Lance Stroll
CONSTRUCTORS:
1. McLaren Mastercard F1 Team
2. Cadillac F1 Team
CAPTAIN: Lando Norris
REASONING: Norris and Antonelli are the form picks, the rest are budget fillers.
"""

CHECKS_RUN, CHECKS_FAILED = 0, []


def check(name, condition, detail=""):
    global CHECKS_RUN
    CHECKS_RUN += 1
    if condition:
        print(f"  PASS -- {name}")
    else:
        CHECKS_FAILED.append(name)
        print(f"  FAIL -- {name}{': ' + detail if detail else ''}")


def make_state(proposal_text, retries=0, validation_errors=None):
    return {
        "proposal_text": proposal_text,
        "driver_prices": FAKE_DRIVER_PRICES,
        "constructor_prices": FAKE_CONSTRUCTOR_PRICES,
        "retries": retries,
        "validation_errors": validation_errors or [],
    }


def test_parse_proposal_well_formed():
    drivers, constructors, captain, reasoning = _parse_proposal(VALID_PROPOSAL)
    check("parses 5 drivers", drivers == ["Lando Norris", "Kimi Antonelli", "Fernando Alonso", "Valtteri Bottas", "Lance Stroll"], f"got {drivers}")
    check("parses 2 constructors", constructors == ["McLaren Mastercard F1 Team", "Cadillac F1 Team"], f"got {constructors}")
    check("parses captain", captain == "Lando Norris", f"got {captain!r}")
    check("parses reasoning", "form picks" in reasoning, f"got {reasoning!r}")


def test_parse_proposal_missing_captain():
    text = VALID_PROPOSAL.replace("CAPTAIN: Lando Norris\n", "")
    _, _, captain, _ = _parse_proposal(text)
    check("missing captain line -> captain is None", captain is None, f"got {captain!r}")


def test_lookup_exact_and_case_insensitive():
    key, info = _lookup("Lando Norris", FAKE_DRIVER_PRICES)
    check("exact match found", key == "Lando Norris" and info["price"] == 26.4)

    key, info = _lookup("lando norris", FAKE_DRIVER_PRICES)
    check("case-insensitive match found", key == "Lando Norris" and info["price"] == 26.4)

    key, info = _lookup("Nonexistent Driver", FAKE_DRIVER_PRICES)
    check("unknown name returns None", key is None and info is None)


def test_validate_budget_valid_team():
    state = make_state(VALID_PROPOSAL)
    result = validate_budget(state)
    expected_total = 26.4 + 26.0 + 6.8 + 3.0 + 3.0 + 31.3 + 3.0  # drivers + constructors
    check("valid team -> no errors", result["validation_errors"] == [], f"got {result['validation_errors']}")
    check("valid team -> correct total cost", abs(result["total_cost"] - expected_total) < 0.01, f"got {result['total_cost']}")
    check("valid team -> captain resolved", result["captain"] == "Lando Norris")


def test_validate_budget_over_cap():
    over_budget_text = VALID_PROPOSAL.replace("2. Cadillac F1 Team", "2. Scuderia Ferrari HP")
    state = make_state(over_budget_text)
    result = validate_budget(state)
    over_by = result["total_cost"] - BUDGET_CAP
    check("over-budget team -> flagged", any("exceeds" in e for e in result["validation_errors"]), f"got {result['validation_errors']}")
    check("over-budget team -> cost computed correctly", over_by > 0, f"total={result['total_cost']}")


def test_validate_budget_wrong_roster_shape():
    too_few_drivers = VALID_PROPOSAL.replace("5. Lance Stroll\n", "")
    state = make_state(too_few_drivers)
    result = validate_budget(state)
    check("4 drivers -> flagged as wrong count", any("exactly 5 drivers" in e for e in result["validation_errors"]), f"got {result['validation_errors']}")


def test_validate_budget_unknown_driver():
    bad_name_text = VALID_PROPOSAL.replace("Lance Stroll", "Not A Real Driver")
    state = make_state(bad_name_text)
    result = validate_budget(state)
    check("unknown driver name -> flagged", any("not found" in e for e in result["validation_errors"]), f"got {result['validation_errors']}")


def test_validate_budget_captain_not_in_roster():
    bad_captain_text = VALID_PROPOSAL.replace("CAPTAIN: Lando Norris", "CAPTAIN: Max Verstappen")
    state = make_state(bad_captain_text)
    result = validate_budget(state)
    check("captain outside the 5 drivers -> flagged", any("must be one of the 5" in e for e in result["validation_errors"]), f"got {result['validation_errors']}")


def test_route_after_validation():
    check(
        "no errors -> route to finalize",
        route_after_validation({"validation_errors": [], "retries": 0}) == "finalize",
    )
    check(
        "errors + retries below cap -> route to retry",
        route_after_validation({"validation_errors": ["x"], "retries": 0}) == "retry",
    )
    check(
        "errors + retries at cap -> route to finalize (give up)",
        route_after_validation({"validation_errors": ["x"], "retries": MAX_RETRIES}) == "finalize",
    )


def main():
    print("=== team_builder.py logic unit checks (no LLM calls) ===\n")
    for fn in [
        test_parse_proposal_well_formed,
        test_parse_proposal_missing_captain,
        test_lookup_exact_and_case_insensitive,
        test_validate_budget_valid_team,
        test_validate_budget_over_cap,
        test_validate_budget_wrong_roster_shape,
        test_validate_budget_unknown_driver,
        test_validate_budget_captain_not_in_roster,
        test_route_after_validation,
    ]:
        fn()

    print(f"\n{CHECKS_RUN - len(CHECKS_FAILED)}/{CHECKS_RUN} checks passed")
    if CHECKS_FAILED:
        print(f"FAILED: {CHECKS_FAILED}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
