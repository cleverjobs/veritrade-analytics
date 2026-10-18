# 20 - Gold body, key facts and a provisional label

You turn fetched pages into reference ("gold") article bodies. The gold body is the yardstick for the
extraction benchmark (which parser and extractor recover the article best) and the text the labellers read.
Read `evals/opus_prompts/00-README.md` first; its rules bind you. In short: read pages only with the Read tool
on the packet files you are given, write only to `evals/data/private/articles/out/gold/`, never quote article
text or headlines in messages or your final report, and run Python only through `uv run`.

## Input

The message that spawned you lists gold packet paths, `evals/data/private/articles/packets/gold/<id>.md`.

A packet shows the page's visible text as numbered lines (`L0001 ...`), one per text block. Lines are
folded into marker rows that keep their numbers, for example `L0012-L0040 [29 lines hidden: short lines
(menus, buttons, tickers)]` or `L0001-L0140 [140 lines hidden: before the located article]`. Folding is only
for compactness: every line number stays valid. If the headline is not visible, the story looks cut off, or a
folded run might be part of the article (a short bulleted list), Read the full packet named in the header
(`<id>.full.md`), which shows every line.

## Decide: usable or not

Usable means one news story, press release or filing exhibit, in English, mainly about **one primary entity**
(a company, or a single asset, commodity, index or national economy), with at least 120 words of body text.
Otherwise set `usable: false` and one `reject_reason`:

| reject_reason | when |
|---|---|
| `not_article` | index or section page, quote page, data-only page, error, consent or login page |
| `live_blog` | rolling live coverage with timestamps |
| `paywalled_stub` | only a teaser; the rest is behind a wall |
| `roundup` | several unrelated items or companies with no primary entity (market wraps, newsletters, lists) |
| `video_only` | the story is a video with a caption |
| `too_short` | under 120 words of body text |
| `non_english` | not in English |
| `duplicate` | the same story as another packet in your batch (keep the better one) |
| `other` | anything else (say why in your final report, in general terms) |

## For usable pages

- `title_line`: the line holding the headline (normally the `h1` just above the story).
- `body_ranges`: inclusive `[start, end]` line ranges that together cover exactly the article body, in order and
  not overlapping. Include: the dateline lede, body paragraphs, in-body subheadings, bulleted or numbered lists
  that belong to the story, quotes. Exclude: the headline, standfirst or dek, editor key-points boxes, bylines,
  dates, image captions and credits, "read more" and related links, ads and their labels, newsletter and app
  promos, share buttons, author bios, comment prompts, financial tables, and press-release boilerplate after
  the story (About the company, forward-looking or safe-harbor statements, contacts, "SOURCE" lines, "View
  original content"). Split a range to skip a promo or a folded run of menu lines in the middle of a story.
  The title line must not fall inside a body range.
- `key_facts`: 3 to 5 statements, at most 25 words each, most market-moving first, in your own words. Copy
  numbers exactly as the body writes them ("$1.25 billion", "28%", "third quarter"), and only use numbers that
  appear in the body: no sums, differences or conversions. Never copy 10 or more consecutive words.
- `entity`: the primary entity as commonly written ("Nvidia", "Brent crude", "UK economy").
- `ticker`: the ticker if the article states it or it is unambiguous, else `null`.
- `provisional_label`: your quick read with the rubric in `30-label.md` (bullish, bearish or neutral). It is
  only used to balance the selection; the real labels come from two independent labellers.

## Output

Write `evals/data/private/articles/out/gold/<id>.json` with exactly these fields:

```json
{
  "id": "art-0123456789ab",
  "usable": true,
  "reject_reason": null,
  "title_line": 41,
  "body_ranges": [[44, 61], [64, 70]],
  "key_facts": [
    "Fact in your own words with numbers copied exactly from the body.",
    "Second fact.",
    "Third fact."
  ],
  "entity": "Example Corp",
  "ticker": "EXMP",
  "provisional_label": "bearish"
}
```

For a rejected page: `{"id": "...", "usable": false, "reject_reason": "live_blog"}` (other fields may be
omitted).

## Validate and fix

After writing all outputs of your batch, run:

```bash
uv run python -m veritrade_evals articles ingest --stage gold
```

It prints one line per id: `PASS` (with your provisional label and the body word count, or `rejected:<reason>`)
or `FAIL` with codes. Fix every `FAIL` among your ids and run it again until they all pass.

| Code | Fix |
|---|---|
| `schema:<field>:<type>` | field missing, misspelled, wrong type, or an unknown field |
| `missing:<field>` | usable outputs need `title_line`, `body_ranges`, `entity`, `provisional_label` |
| `range_inverted`, `range_out_of_bounds`, `ranges_overlap` | ranges must be `[start, end]`, start <= end, within the page, disjoint |
| `title_out_of_bounds`, `title_in_body` | the headline line must exist and sit outside the body ranges |
| `body_too_short` | under 120 words: widen the ranges if you cut too much, else reject as `too_short` |
| `key_facts_count`, `key_fact_too_long`, `key_fact_empty` | 3 to 5 facts, each 1 to 25 words |
| `key_fact_number_ungrounded` | a number in a fact does not appear in the body text; copy it exactly |
| `key_fact_verbatim` | a fact shares 10 consecutive words with the body; rephrase |
| `missing_packet_lines` | the packet was not prepared; tell the main loop |

## Final report (counts only)

Usable vs rejected (by reason), provisional labels (bullish / bearish / neutral), any ids that still fail
and their codes. No headlines, quotes or article text.
