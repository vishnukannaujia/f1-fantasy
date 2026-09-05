"""
Predictions ledger: score a saved pre-race prediction (predictions/*.json) against
the real result once the race weekend is over, and keep a running track record
across races. This turns the one-off held-out backtest (eval_prediction_backtest.py,
against the already-known Zandvoort result) into an ongoing, repeatable practice --
every race weekend adds a real data point instead of relying on a single n=1
retrospective test.

Usage (after a race has actually happened):
    python eval/predictions_tracker.py score predictions/2026-09-06_italian_gp.json \\
        --finishing-order "Lando Norris,Kimi Antonelli,George Russell,Lewis Hamilton,Charles Leclerc,Oscar Piastri" \\
        --dnfs "Driver Name" --penalties "note about any post-race verdict/DSQ"

    python eval/predictions_tracker.py summary   # prints the track record across
                                                   # every scored race so far
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = ROOT / "predictions"


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(path: Path, record: dict):
    Path(path).write_text(json.dumps(record, indent=2) + "\n")


def score_formation(formation: dict, finishing_order: list) -> dict:
    predicted_drivers = set(formation["drivers"])
    actual_top6 = finishing_order[:6]
    top6_overlap = len(predicted_drivers & set(actual_top6))

    captain = formation["captain"]
    captain_finish = (finishing_order.index(captain) + 1) if captain in finishing_order else None

    return {
        "top6_overlap": f"{top6_overlap}/{len(predicted_drivers)}",
        "captain_finishing_position": captain_finish,
        "captain_scored_well": captain_finish is not None and captain_finish <= 6,
    }


def cmd_score(args):
    path = Path(args.file)
    record = load(path)

    finishing_order = [name.strip() for name in args.finishing_order.split(",")]
    actual_result = {
        "finishing_order": finishing_order,
        "dnfs": [name.strip() for name in args.dnfs.split(",")] if args.dnfs else [],
        "penalties_or_verdicts": args.penalties or "",
    }

    scoring = {}
    for formation in record["formations"]:
        scoring[formation["label"]] = score_formation(formation, finishing_order)

    record["actual_result"] = actual_result
    record["scoring"] = scoring
    save(path, record)

    print(f"=== Scored: {record['race']} ({record['circuit']}, {record['race_date']}) ===\n")
    print(f"Actual finishing order (top 6): {', '.join(finishing_order[:6])}")
    if actual_result["dnfs"]:
        print(f"DNFs: {', '.join(actual_result['dnfs'])}")
    if actual_result["penalties_or_verdicts"]:
        print(f"Verdicts/penalties: {actual_result['penalties_or_verdicts']}")
    print()

    for formation in record["formations"]:
        s = scoring[formation["label"]]
        print(f"[{formation['label']}] predicted confidence: {formation.get('confidence_pct', '?')}%")
        print(f"  top6 driver overlap: {s['top6_overlap']}")
        print(f"  captain ({formation['captain']}) finished: P{s['captain_finishing_position']}" if s["captain_finishing_position"] else f"  captain ({formation['captain']}) did not finish in the top 6")
        print()

    print(f"Saved scoring back to {path}")


def cmd_summary(args):
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    if not files:
        print("No prediction records found in predictions/.")
        return

    print("=== Prediction track record across all races ===\n")
    for f in files:
        record = load(f)
        status = "SCORED" if record.get("scoring") else "pending (race not yet run/scored)"
        print(f"{record['race']} ({record['race_date']}) -- {status}")
        if record.get("scoring"):
            for label, s in record["scoring"].items():
                print(f"  [{label}] top6 overlap: {s['top6_overlap']}, captain finish: P{s['captain_finishing_position'] or '-'}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="score a prediction file against the real result")
    p_score.add_argument("file", help="path to predictions/*.json")
    p_score.add_argument("--finishing-order", required=True, help="comma-separated top finishers, P1 first")
    p_score.add_argument("--dnfs", default="", help="comma-separated DNF driver names")
    p_score.add_argument("--penalties", default="", help="freeform note on post-race verdicts/penalties")
    p_score.set_defaults(func=cmd_score)

    p_summary = sub.add_parser("summary", help="print the track record across all scored races")
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
