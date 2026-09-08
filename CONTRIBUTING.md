# Contributing to f1-fantasy

Thanks for your interest! Contributions are welcome.

## License & CLA

- f1-fantasy is licensed under **Apache-2.0** (see `LICENSE`).
- Before your first contribution is merged, you must agree to the
  [Contributor License Agreement](CLA.md). To sign, add a line to
  `CONTRIBUTORS.md` in your pull request:
  ```
  - Your Name <email> — I have read and agree to the CLA (YYYY-MM-DD)
  ```
  (Later this may be replaced by an automated CLA check such as CLA Assistant.)

## Workflow

1. Open an issue describing the change (bug or feature) before large work.
2. Fork, branch from `main`, keep changes focused.
3. Match the existing design: retrieval-only lookups go through `chains.py`,
   anything needing a hard constraint (budget, roster shape) goes through
   `team_builder.py`'s LangGraph — see `ARCHITECTURE.md` for the full rationale
   before adding a new data source or capability.
4. Run the eval suite relevant to what you touched — see `EVALS.md` for what
   each script checks:
   ```bash
   python eval/eval_chunking.py             # if you touched ingest.py or the corpus
   python eval/eval_team_builder_logic.py    # if you touched team_builder.py's logic
   python eval/eval_generation.py            # if you touched chains.py or prompts
   python eval/eval_quality_gate.py
   python eval/eval_routing_logic.py
   python eval/eval_prompt_injection.py
   ```
   Add a test case to the relevant script alongside any change to that logic —
   every non-trivial design decision here traces to a specific eval result, see
   `EVALS.md`'s "Decisions actually made via these evals" section for the
   standard this project holds itself to (including documenting a rejected
   idea with evidence, not just adopted ones).
5. Open a PR referencing the issue.

## Good first contributions

See `ARCHITECTURE.md`'s "Known gaps" section for an honest, current list —
it's kept up to date and is the best source of what's actually needed, but a
few standing ones:

- `refresh_data.py` — automate the weekly `data/raw/*.txt` refresh (currently
  manual research + file edits before each race weekend)
- A second held-out backtest (e.g. a different race) to check whether the
  hot-streak-weighting prompt fix generalizes beyond the two races it's been
  validated against so far
- Extending prompt-injection coverage in `eval/eval_prompt_injection.py` beyond
  the two adversarial cases currently tested
- Real Langfuse trace verification (keys are supported in `.env` but have
  never been added, so the integration has only been tested against its
  no-op fallback path)
