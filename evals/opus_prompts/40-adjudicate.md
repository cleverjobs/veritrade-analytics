# 40 - Adjudicate disagreements and ambiguous articles

You resolve the articles where the two labellers disagreed or at least one flagged the article as ambiguous.
The message that spawned you lists adjudication packets, `evals/data/private/articles/packets/adjudicate/<id>.md`.
Read `evals/opus_prompts/00-README.md` first; its rules bind you. In short: read articles only with the Read
tool on the packets you were given, write only to `evals/data/private/articles/out/adjudicate/`, never quote
article text or headlines in messages or your final report, and run Python only through `uv run`.

A packet holds both labellers' outputs (label, score, entity, ticker, ambiguous flag, confidence, rationale)
followed by the headline and article body, the same text they read. Do not open other files.

## Rubric

Use exactly the rubric in `evals/opus_prompts/30-label.md`: the economic impact of the news on its primary
entity, not the tone; bullish or bearish only for a clear, material implication; mixed news follows the
dominant material driver, or neutral if balanced. Form your own reading from the article first, then weigh
the two rationales as arguments, not votes.

## Decide

- `keep`: the article has one defensible label under the rubric, and the disagreement came from a misreading,
  a materiality threshold, or a different primary entity that the article settles. Your label, score, entity
  and ticker become final.
- `replace`: the article stays genuinely ambiguous: balanced mixed news, no clear primary entity, materiality
  that depends on facts the article does not give, or a direction a careful analyst could reasonably dispute.
  Replaced articles leave the set and the main loop finds others. The set is only useful if its labels are
  defensible, so do not keep an article just to fill a quota.

## Output

Write `evals/data/private/articles/out/adjudicate/<id>.json`:

```json
{
  "id": "art-0123456789ab",
  "decision": "keep",
  "label": "neutral",
  "score": 0,
  "entity": "Example Corp",
  "ticker": null,
  "rationale": "Your own words, at most 50: why this label, and what settled the disagreement."
}
```

For `replace`, still give your best label and score (their sign must agree) and say in the rationale why the
article is too ambiguous to keep.

## Validate and fix

```bash
uv run python -m veritrade_evals articles ingest --stage adjudicate
```

Fix every `FAIL`: `schema:<field>:<type>`, `score_label_mismatch`, `rationale_too_long`, `rationale_verbatim`
(rephrase), `not_requested` (the id is not in `adjudicate_ids.txt`; tell the main loop), `gold_missing`.
Do not run `articles status`.

## Final report (counts only)

Kept by final label, replaced, and for kept articles whether you sided with labeller A, labeller B or neither.
No headlines, quotes or article text.
