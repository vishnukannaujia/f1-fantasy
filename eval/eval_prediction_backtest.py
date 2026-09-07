"""
Held-out backtest: predict a race that already happened, using only the context
that would have been available BEFORE it, then score the prediction against the
real result. This is the standard way to validate a forecasting system before
trusting it on a live, unresolved event (the actual point of this eval: establish
confidence in team_builder.py's reasoning ahead of a real, unresolved upcoming race).

Held-out race: the 2026 Dutch Grand Prix at Zandvoort (the most recent race in our
corpus). PRE_ZANDVOORT_CONTEXT below is built from fresh research into the
state of the season *as of right after the Hungarian Grand Prix* (the round
before Zandvoort) -- standings, recent form, and Zandvoort's own circuit
characteristics -- deliberately NOT including anything about the Zandvoort
result itself. GROUND_TRUTH is the real result, already known and present in
data/raw/05_recent_form_and_last_race.txt.

This is necessarily a harder, noisier prediction than team-building: race
outcomes have real variance (incidents, strategy, weather) that no amount of
pre-race context resolves. The scoring below is intentionally lenient
accordingly (set overlap, not just exact order) -- see the printed report for
what "good" looks like here.

Run:
    python eval/eval_prediction_backtest.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from team_builder import extract_text, get_llm  # noqa: E402

PRE_ZANDVOORT_CONTEXT = """
2026 F1 SEASON CONTEXT -- AS OF THE END OF ROUND 11 (Hungarian Grand Prix),
BEFORE the Dutch Grand Prix at Zandvoort (Round 12).
Source: aggregated from racingnews365.com and the-race.com 2026 Hungarian GP
coverage, retrieved 2026-09-05.

HUNGARIAN GRAND PRIX RESULT (Round 11, most recent race): Lando Norris took his
first Grand Prix win of the season, finishing ahead of Max Verstappen and Kimi
Antonelli.

DRIVERS' CHAMPIONSHIP STANDINGS heading into the summer break / Dutch GP:
1. Kimi Antonelli (Mercedes) -- leads the championship.
2. Lewis Hamilton (Ferrari) -- 50 points behind Antonelli.
3. George Russell (Mercedes) -- 9 points behind Hamilton (59 behind Antonelli).
4. Charles Leclerc (Ferrari) -- 138 points.
5. Lando Norris (McLaren) -- 128 points, just boosted by his Hungary win.

FORM HEADING IN: Norris's Hungary win is his first of the season and comes as
a jump for McLaren; Mercedes (Antonelli, Russell) and Ferrari (Hamilton,
Leclerc) have been the more consistent front-runners across the season so far.

CIRCUIT NOTES: THE DUTCH GRAND PRIX, CIRCUIT ZANDVOORT (upcoming, Round 12)
Source: aggregated from f1chronicle.substack.com, formula1.com, and
motorsportweek.com 2026 Dutch GP previews.

Zandvoort is a short (4.259 km), narrow, old-school circuit with two steeply
banked corners (Turn 3 and Turn 14, banked at 19 and 18 degrees). It requires
HIGH aerodynamic downforce, similar to Budapest (the Hungarian GP venue just
raced) -- the opposite demand of a low-downforce power circuit. Overtaking is
notoriously difficult here: a single real straight and a narrow track surface
mean genuine passing opportunities are rare. This tends to reward whichever
driver/team qualifies best and manages tires and strategy well over the race
distance, rather than rewarding a car that can overtake its way forward from a
poor grid slot. The circuit is demanding on tire energy despite its short lap
length, due to its high-speed corner sequences.
""".strip()

PREDICTION_PROMPT = """
Based ONLY on the context above (standings and form as of the end of Round 11,
and the Dutch Grand Prix / Zandvoort circuit notes), predict the top 6
finishers of the upcoming Dutch Grand Prix, in order from P1 to P6.

Do not guess at information not given above. Reason briefly about why, given
Zandvoort's characteristics (favors qualifying pace + strategy over overtaking)
and each driver's recent form and championship position.

Reply in EXACTLY this format:

PREDICTION:
1. <driver name>
2. <driver name>
3. <driver name>
4. <driver name>
5. <driver name>
6. <driver name>
REASONING: <2-4 sentences>
"""

# The real result, already known (data/raw/05_recent_form_and_last_race.txt) --
# used ONLY for scoring, never shown to the model.
GROUND_TRUTH = [
    "Lando Norris",
    "Kimi Antonelli",
    "George Russell",
    "Lewis Hamilton",
    "Charles Leclerc",
    "Oscar Piastri",
]


def get_prediction() -> tuple[list, str]:
    llm = get_llm()
    prompt = f"{PRE_ZANDVOORT_CONTEXT}\n\n{PREDICTION_PROMPT}"
    text = extract_text(llm.invoke(prompt).content)

    predicted, reasoning_lines, in_prediction, in_reasoning = [], [], False, False
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("PREDICTION:"):
            in_prediction, in_reasoning = True, False
            continue
        if upper.startswith("REASONING:"):
            in_prediction, in_reasoning = False, True
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                reasoning_lines.append(rest)
            continue
        if in_reasoning and stripped:
            reasoning_lines.append(stripped)
            continue
        m = re.match(r"^\d+\.\s*(.+)$", stripped)
        if m and in_prediction:
            predicted.append(m.group(1).strip())

    return predicted, " ".join(reasoning_lines)


def score(predicted: list, ground_truth: list) -> dict:
    exact_matches = sum(
        1 for i, name in enumerate(predicted[: len(ground_truth)]) if i < len(ground_truth) and name == ground_truth[i]
    )
    top6_overlap = len(set(predicted[:6]) & set(ground_truth))
    winner_correct = bool(predicted) and predicted[0] == ground_truth[0]
    podium_predicted_set = set(predicted[:3])
    podium_actual_set = set(ground_truth[:3])
    podium_overlap = len(podium_predicted_set & podium_actual_set)
    return {
        "exact_position_matches": f"{exact_matches}/{len(ground_truth)}",
        "top6_set_overlap": f"{top6_overlap}/{len(ground_truth)}",
        "podium_set_overlap": f"{podium_overlap}/3",
        "winner_correct": winner_correct,
    }


def main():
    print("=== Held-out backtest: predict Round 12 (Dutch GP, Zandvoort) using only pre-race context ===\n")
    predicted, reasoning = get_prediction()

    print("PREDICTED (model, given only pre-Zandvoort context):")
    for i, name in enumerate(predicted, 1):
        print(f"  {i}. {name}")
    print(f"\nReasoning: {reasoning}\n")

    print("ACTUAL RESULT (already known, held out from the model):")
    for i, name in enumerate(GROUND_TRUTH, 1):
        print(f"  {i}. {name}")
    print()

    metrics = score(predicted, GROUND_TRUTH)
    print("=== Scoring (intentionally lenient -- race outcomes have real variance) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(
        "\nInterpretation: this measures whether the reasoning process (standings + form + "
        "circuit fit) produces a plausible, grounded prediction -- not whether it can predict "
        "race-day variance (incidents, strategy calls, weather) that no pre-race context resolves. "
        "A strong podium/top-6 overlap here is the basis for trusting the same reasoning approach "
        "in team_builder.py for a real, unresolved upcoming race."
    )


if __name__ == "__main__":
    main()
