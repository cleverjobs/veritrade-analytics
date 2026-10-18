# 50 - Judge candidate summaries

You judge short summaries of financial news articles written by small local models. The message that spawned
you lists judge packets, `evals/data/private/summary/judge/packets/<id>.md`. Read
`evals/opus_prompts/00-README.md` first; its rules bind you, and the ones that matter here are repeated below.

Each packet holds one article (title and body, the gold text) and two to five candidate summaries labelled
`S1`, `S2`, ... The labels are anonymised and shuffled per article: you do not know, and must not try to find
out, which model wrote which summary. Judge each summary on its own merits against the article.

## Rules (publisher rights, a public transcript, and a blind judgement)

1. Read article text only with the Read tool, and only from the packets you were given. Do not open any other
   file under `evals/data/private/` (caches, blinding maps, other packets, other judges' outputs), and do not
   grep, cat or print packets in a shell.
2. Write only `evals/data/private/summary/judge/out/<id>.json`, one file per packet.
3. Never quote the article, its title or a summary in messages, your final report or file names. Refer to
   articles by id and to summaries by label. Hallucination notes are in your own words; the validator rejects
   any note that shares a run of 10 consecutive words with the article or a summary.
4. No WebFetch, WebSearch or browser. Judge from the packet alone, not from what you know about the company.
5. Python only through `uv run ...` from the repository root; no inline Python.

## What the summaries were asked to do

At most 3 sentences and 60 words; lead with the market-moving fact (what happened, to which company or asset);
copy key numbers exactly; name the company; no speculation, opinion or facts that are not in the article.
Length, sentence count and the presence of the company name are checked by code, so do not score them. Judge
content only.

## Scores per summary

**faithfulness** (1-5): is every claim supported by the article?

- 5: every claim is supported; every number, name, date and direction matches the article exactly.
- 4: one minor slip a trader would shrug off: a rounded number, a slightly loose word, a small unsupported
  detail that changes nothing material.
- 3: one material problem: a wrong key number, an unsupported claim that matters, a misattributed fact, or a
  hedge in the article stated as certain.
- 2: several material problems, or one that misleads about the direction or size of the news.
- 1: mostly unsupported, contradicts the article, or is about the wrong company or event.

**coverage** (1-5): does it carry the market-moving facts a trader needs? Decide for yourself, from the
article, what those are (usually the event, its direction and its key numbers).

- 5: the main market-moving fact and its key numbers, plus the most important supporting point.
- 4: the main fact and its key numbers; a secondary point is missing.
- 3: the main fact, but without the numbers or the driver that make it actionable.
- 2: secondary details only; the main market-moving fact is missing or buried.
- 1: almost nothing a trader could use.

**hallucinations**: a list, one entry per claim, number, name or date in the summary that the article does
not support or that contradicts it. Each entry at most 25 words, in your own words, saying what is wrong
(for example "gives the quarterly loss as 12 million; the article reports 21 million"). Numbers and names may
appear in a note; runs of the article or summary text may not. A rounded number that keeps the meaning is not
a hallucination; it can cost a point of faithfulness on its own. Empty list when there is nothing to report.

**leads_with_market_moving_fact**: `true` when the first sentence states the main market-moving fact.

Consistency rules the validator enforces: faithfulness 5 requires an empty hallucination list, and
faithfulness 3 or lower requires at least one hallucination entry. Give identical summaries identical
judgements, and the same problem the same penalty in every summary of one article.

## Output

Write `evals/data/private/summary/judge/out/<id>.json` with one entry per label in the packet:

```json
{
  "id": "art-0123456789ab",
  "summaries": [
    {
      "label": "S1",
      "faithfulness": 5,
      "coverage": 4,
      "hallucinations": [],
      "leads_with_market_moving_fact": true
    },
    {
      "label": "S2",
      "faithfulness": 3,
      "coverage": 3,
      "hallucinations": ["states revenue rose 8%; the article reports 6%"],
      "leads_with_market_moving_fact": false
    }
  ]
}
```

## Validate and fix

```bash
uv run python -m veritrade_evals.summary_eval judge-check --split dev --id art-0123456789ab
```

Repeat `--id` for each of your packets (use the split named in the spawning message). It prints `ok <id>` or
`FAIL <id>` with codes, never which model is behind a label. Fix every `FAIL`:

- `missing_output`, `id_mismatch`, `schema:<field>:<type>`: the file or a field is missing or malformed.
- `S<n>:missing_label`, `S<n>:unknown_label`, `S<n>:duplicate_label`: exactly one entry per packet label.
- `S<n>:hallucination_too_long`, `S<n>:hallucination_empty`: at most 25 words, never blank.
- `S<n>:hallucination_verbatim`: rephrase the note in your own words.
- `S<n>:faithfulness_5_with_hallucinations`, `S<n>:low_faithfulness_without_hallucinations`: make the score
  and the list agree.
- `S<n>:stale_summary`: the packet is out of date; stop and tell the main loop.

Do not run `judge-ingest`; the main loop does.

## Final report (counts only)

Packets judged, and per packet the faithfulness and coverage scores by label plus the number of
hallucination entries. No article text, headlines, summaries or quotes.
