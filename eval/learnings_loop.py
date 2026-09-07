"""
The feedback loop: extract new, generalizable lessons from a SCORED prediction
file (predictions/*.json after predictions_tracker.py score has run) and
append them to learnings/learnings.json -- which team_builder.py's
load_static_context reads and includes in every future prediction's prompt.

This replaces hand-editing PROPOSAL_INSTRUCTIONS after each race (which is
what happened once, manually, after the Zandvoort backtest) with a repeatable
process: real result comes in -> this script proposes grounded lessons ->
a human reviews and appends -> every future team_builder.py run automatically
picks them up, no code change required.

Deliberately NOT fully automatic -- proposes candidate learnings and prints
them for review by default; --append is required to actually write them.
Feeding a bad or overfit "lesson" into every future prediction unreviewed is a
worse failure mode than a manual step, given how few data points (races) this
system has to learn from.

Usage:
    python eval/learnings_loop.py predictions/2026-09-06_italian_gp.json
    python eval/learnings_loop.py predictions/2026-09-06_italian_gp.json --append
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from chains import extract_text, get_llm  # noqa: E402

LEARNINGS_PATH = ROOT / "learnings" / "learnings.json"


def load_prediction(path: Path) -> dict:
    record = json.loads(Path(path).read_text())
    if not record.get("actual_result") or not record.get("scoring"):
        raise SystemExit(f"{path} has not been scored yet -- run predictions_tracker.py score first.")
    return record


def load_learnings() -> list:
    if LEARNINGS_PATH.exists():
        return json.loads(LEARNINGS_PATH.read_text())
    return []


def save_learnings(learnings: list):
    LEARNINGS_PATH.write_text(json.dumps(learnings, indent=2) + "\n")


def build_discrepancy_summary(record: dict) -> str:
    actual = record["actual_result"]
    lines = [
        f"Race: {record['race']} ({record['circuit']}, {record['race_date']})",
        f"Actual finishing order (top 6): {', '.join(actual['finishing_order'][:6])}",
        f"DNFs: {', '.join(actual['dnfs']) if actual['dnfs'] else 'none'}",
        f"Verdicts/penalties: {actual.get('penalties_or_verdicts', 'none noted')}",
        "",
        "How each predicted formation did, compared to what its own reasoning claimed:",
    ]
    for formation in record["formations"]:
        s = record["scoring"].get(formation["label"], {})
        lines.append(f"\n[{formation['label']}] (stated confidence: {formation.get('confidence_pct', '?')}%)")
        lines.append(f"  Captain: {formation['captain']} -> actually finished P{s.get('captain_finishing_position', '?')}")
        lines.append(f"  Top-6 overlap: {s.get('top6_overlap', '?')}")
        lines.append(f"  Original reasoning: {formation.get('reasoning', '(not recorded)')}")
    return "\n".join(lines)


def propose_learnings(record: dict, existing: list) -> list:
    """Each proposed learning is classified NEW / REFINES #<id> / CONFIRMS #<id>
    against the existing ledger, so the caller can update evidence weight
    instead of appending flat, potentially-duplicate or -contradictory
    entries. With only a handful of races scored so far, most genuine findings
    should be NEW or REFINES; CONFIRMS should be rare until the season has
    given the same pattern multiple independent chances to show up."""
    existing_text = "\n".join(f"- (#{e['id']}) {e['lesson']}" for e in existing) or "(none yet)"
    summary = build_discrepancy_summary(record)

    prompt = f"""You are analyzing a fantasy sports prediction system's real race outcome to extract
lessons for its own future reasoning -- this is a feedback loop: whatever you propose here gets fed
into the same system's prompt for every future prediction, weighted by how much evidence supports it.
Be conservative: with only a handful of races scored so far, most single-race findings are hypotheses,
not proven rules. Do not propose anything that could just as easily be explained by ordinary race-day
variance (an incident, a strategy call) rather than a real, repeatable pattern in the REASONING itself.

EXISTING LEARNINGS ALREADY IN THE SYSTEM:
{existing_text}

WHAT ACTUALLY HAPPENED THIS RACE, vs. what each formation's own reasoning predicted:
{summary}

Propose 0 to 3 learnings. For EACH one, classify its relationship to the existing learnings above:
- NEW: a genuinely independent finding not covered by anything existing.
- REFINES #<id>: this race's evidence narrows or qualifies an existing lesson to a specific
  condition (e.g. "X, but only when Y") -- it does NOT mean the old lesson was wrong, both still apply
  in their respective circumstances. Use this instead of proposing something that would otherwise look
  like it contradicts an existing lesson.
- CONFIRMS #<id>: this race's evidence is the SAME underlying pattern as an existing lesson playing out
  again, independently. Only use this if it's genuinely the same mechanism, not just a superficially
  similar outcome.

Reply with each learning in exactly this format, separated by a line containing only "---":

RELATION: <NEW, or REFINES #<id>, or CONFIRMS #<id>>
FINDING: <what specifically happened, citing concrete facts from above>
LESSON: <the specific, actionable instruction to feed into future predictions -- written as an
instruction TO the prediction system, not a description of the race>
"""
    llm = get_llm()
    text = extract_text(llm.invoke(prompt).content)

    proposed = []
    for block in text.split("---"):
        block = block.strip()
        if not block or "FINDING:" not in block:
            continue
        relation_raw = block.split("RELATION:", 1)[1].split("FINDING:")[0].strip() if "RELATION:" in block else "NEW"
        finding = block.split("FINDING:", 1)[1].split("LESSON:")[0].strip()
        lesson = block.split("LESSON:", 1)[1].strip() if "LESSON:" in block else ""
        if not (finding and lesson):
            continue

        relation, related_id = "NEW", None
        m = re.search(r"(REFINES|CONFIRMS)\s*#?(\d+)", relation_raw, re.IGNORECASE)
        if m:
            relation, related_id = m.group(1).upper(), int(m.group(2))
        proposed.append({"relation": relation, "related_id": related_id, "finding": finding, "lesson": lesson})
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_file")
    parser.add_argument("--append", action="store_true", help="actually append proposed learnings to learnings.json")
    args = parser.parse_args()

    record = load_prediction(Path(args.prediction_file))
    existing = load_learnings()
    proposed = propose_learnings(record, existing)

    if not proposed:
        print("No new generalizable learnings proposed from this race (existing learnings already cover it, or nothing here rises above single-race noise).")
        return

    print(f"=== {len(proposed)} candidate learning(s) proposed from {record['race']} ===\n")
    by_id = {e["id"]: e for e in existing}
    next_id = max((e["id"] for e in existing), default=0) + 1
    new_id_counter = 0
    for p in proposed:
        rel_label = p["relation"] if p["relation"] == "NEW" else f"{p['relation']} #{p['related_id']}"
        print(f"[{rel_label}]")
        print(f"  FINDING: {p['finding']}")
        print(f"  LESSON:  {p['lesson']}")
        if p["relation"] == "CONFIRMS" and p["related_id"] in by_id:
            target = by_id[p["related_id"]]
            print(f"  -> would raise lesson #{p['related_id']}'s race_count from {target.get('race_count', 1)} "
                  f"to {target.get('race_count', 1) + 1}"
                  + (" and upgrade it to 'confirmed'" if target.get("race_count", 1) + 1 >= 2 else ""))
        print()

    if args.append:
        for p in proposed:
            if p["relation"] == "CONFIRMS" and p["related_id"] in by_id:
                target = by_id[p["related_id"]]
                target["race_count"] = target.get("race_count", 1) + 1
                if target["race_count"] >= 2:
                    target["confidence"] = "confirmed"
                target.setdefault("confirming_sources", []).append(
                    f"{record['race']} ({record['race_date']})"
                )
                continue
            # NEW or REFINES (or CONFIRMS pointing at an id that doesn't exist,
            # which shouldn't happen but degrades safely to a fresh entry)
            existing.append({
                "id": next_id + new_id_counter,
                "added": record["race_date"],
                "confidence": "preliminary",
                "race_count": 1,
                "refines": p["related_id"] if p["relation"] == "REFINES" else None,
                "source": f"Real scored result: {record['race']} (eval/predictions_tracker.py)",
                "finding": p["finding"],
                "lesson": p["lesson"],
            })
            new_id_counter += 1
        save_learnings(existing)
        print(f"Updated {LEARNINGS_PATH} -- team_builder.py will include this in every future prediction.")
    else:
        print("Not appended (dry run). Re-run with --append to commit these to learnings.json.")


if __name__ == "__main__":
    main()
