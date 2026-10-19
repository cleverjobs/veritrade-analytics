# A note to the reviewers

Thank you for reading this far. I really like how you run this prescreen: you expect AI assistants, you ask for the
conversation log, and you say the focus is design judgment, problem decomposition and verification rather than
typing every line. That is how I work day to day, and it is rare to see a hiring process ask to see it.

This log is that work, filtered and redacted as described in [`README.md`](README.md) and otherwise as it happened.
I set the direction and the constraints (local open-weight models only, no paid APIs, precision over recall,
latency budgets fixed before measuring), settled every design decision in the interview at the start, and steered
as the build ran. Claude Code with Opus 5.5 planned, wrote the code and the docs, and orchestrated the subagents
that built the evaluation data.

Good places to start:

- The `/grill-me` interview near the beginning, where the decisions were made
  ([`01-decisions.md`](../docs/design/01-decisions.md) has the result).
- The build-time findings in the same file, where measurements changed the plan.
- [`05-latency.md`](../docs/design/05-latency.md) and [`04-evals.md`](../docs/design/04-evals.md), where missed
  budgets and weak spots are reported rather than tuned away.

I look forward to walking you through it.

Kris
