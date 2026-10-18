# 30 - Label an article (labeller A or labeller B)

You are one of two independent labellers. The message that spawned you says whether you are **labeller A**
(write to `out/label_a/`) or **labeller B** (write to `out/label_b/`), and lists label packet paths,
`evals/data/private/articles/packets/label/<id>.md`. Read `evals/opus_prompts/00-README.md` first; its rules
bind you. In short: read articles only with the Read tool on the packets you were given, write only to your own
output folder, never quote article text or headlines in messages or your final report, and run Python only
through `uv run`.

## Independence

Your label must be your own reading of the article. Do not open anything under
`evals/data/private/articles/out/` except your own outputs, nor gold, adjudication or full packets, candidates,
fetch records or HTML. Do not look up the company, its share price or later events; judge the article as of its
publication. A packet holds an id, the headline and the article body (long articles are cut at a paragraph
boundary after about 1,200 words; a note says how much is not shown).

## Rubric (the same one the service's prompts use)

Judge the **economic impact of the news on its primary entity** (the company or asset the article is mainly
about), not the emotional tone of the writing.

- **bullish**: a clear, material positive implication for the entity's value, for example results or guidance
  above expectations, a major contract or approval, a takeover offer at a premium.
- **bearish**: a clear, material negative implication, for example results or guidance below expectations,
  large losses, investigations, recalls, defaults, downgrades.
- **neutral**: no clear material direction: routine or scheduled announcements, commentary without new facts,
  minor news.

Mixed news follows the dominant material driver; if positives and negatives are balanced, choose neutral.
Dramatic or emotional wording is not a signal, and neither is the outlet's general mood about the market.
Only call bullish or bearish when the implication is both clear and material; when in doubt, neutral.

## Fields

- `label`: `bullish`, `bearish` or `neutral`.
- `score`: an integer from -2 to 2 whose sign matches the label (0 if and only if neutral).
  2 = major, unambiguous positive news; 1 = clear but moderate positive implication; 0 = no clear material
  direction (routine, minor or balanced news); -1 = clear but moderate negative implication; -2 = major,
  unambiguous negative news.
- `entity`: the primary entity as commonly written. `ticker`: the ticker if stated or unambiguous, else `null`.
- `rationale`: at most 50 words, your own words, naming the driver of your call. No quotes; never 10 or more
  consecutive words from the article.
- `ambiguous`: `true` when a careful analyst could reasonably pick a different class: genuinely mixed news,
  unclear materiality, no clear primary entity, or a story whose direction depends on facts it does not give.
  Ambiguous articles go to an adjudicator, so flag honestly rather than forcing a call.
- `confidence`: 1 (guess) to 5 (certain).

## Output

Write `evals/data/private/articles/out/label_a/<id>.json` (labeller A) or `.../out/label_b/<id>.json`
(labeller B):

```json
{
  "id": "art-0123456789ab",
  "label": "bearish",
  "score": -1,
  "entity": "Example Corp",
  "ticker": "EXMP",
  "rationale": "Your own words: the driver and why it is material.",
  "ambiguous": false,
  "confidence": 4
}
```

## Validate and fix

After writing all outputs, run the ingest for your stage (`label_a` or `label_b`):

```bash
uv run python -m veritrade_evals articles ingest --stage label_a
```

Fix every `FAIL` among your ids: `schema:<field>:<type>` (missing or wrong field), `score_label_mismatch`
(the score's sign must match the label, 0 only for neutral), `rationale_too_long` (over 50 words),
`rationale_verbatim` (10 consecutive words shared with the article; rephrase), `gold_missing` (tell the main
loop). Do not pass `--agreement` and do not run `articles status`.

## Final report (counts only)

Labels by class, how many you flagged ambiguous, mean confidence, any ids that still fail. No headlines,
quotes or article text.
