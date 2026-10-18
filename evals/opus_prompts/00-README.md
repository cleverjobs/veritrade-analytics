# Opus prompts: the 90-article evaluation set

Public sentiment datasets are sentences and headlines; the service reads whole articles. This folder holds the
prompts for the Claude Opus subagents that build a set of 90 real articles: 10 primary sources x 3 classes
(bullish, bearish, neutral) x 3 articles, split 30 dev / 60 test. There is no human check: two independent
labellers and an adjudicator stand in for one (decisions Q5 and R5 in `.context/docs/design/01-decisions.md`).

The agents never fetch anything themselves. They drive the deterministic CLI
`uv run python -m veritrade_evals articles <command>` (code in `evals/src/veritrade_evals/articles*.py` and
`discovery.py`), read private packet files, and write JSON outputs that the CLI validates.

## Pipeline

| Step | Who (every spawn `model: "opus"`) | Prompt | Commands | Result |
|---|---|---|---|---|
| 1 discover | one agent per source | `10-discover.md` | `discover`, `candidates`, `fetch` | candidates, raw HTML, fetch records |
| 2 gold | agents over batches of gold packets | `20-gold.md` | `prepare --stage gold`, `ingest --stage gold` | gold body, key facts, provisional label |
| 3 label | labeller A and labeller B, separate agents | `30-label.md` | `prepare --stage label`, `ingest --stage label_a` / `label_b` | labels, agreement, `adjudicate_ids.txt` |
| 4 adjudicate | one agent | `40-adjudicate.md` | `prepare --stage adjudicate`, `ingest --stage adjudicate` | keep or replace |
| 5 refill | main loop | - | `status`, then steps 1-4 for the cells that are short | every cell has enough eligible articles |
| 6 finalize | main loop | - | `finalize`, `verify` | committed `evals/data/articles/{articles.jsonl,stats.json}` |
| 7 judge summaries | judge agents over batches of judge packets | `50-judge.md` | `summary_eval judge-prepare`, `judge-check`, `judge-ingest` | blind 1-5 faithfulness and coverage per candidate summary |

Step 7 uses `uv run python -m veritrade_evals.summary_eval <command>` and runs after `generate` and `score`
have produced the candidate summaries (decision R6). Judges see each article with its candidate summaries
under shuffled labels `S1..Sn`; the label-to-model map stays in a private file only `judge-ingest` reads.

Only the main loop runs `prepare`, `finalize`, `verify`, `ingest --agreement`, `judge-prepare` and
`judge-ingest`. Discovery and gold agents may
run `status`; labellers and the adjudicator do not, so their judgements stay independent.

An article is **eligible** when its gold output is usable, it was published within 60 days before 2026-10-18
(2026-08-19 to 2026-10-18), both labellers produced valid outputs, and either they agree with neither flagging
`ambiguous`, or the adjudicator decided `keep` (its label is then final). `finalize` picks 3/3/3 per source; a
source that cannot fill a class may give up one article there (minimum 2) while other sources take a 4th, so
the set is still 30/30/30. The exact algorithm is printed into `stats.json`.

## Rules for every agent (publisher rights and the public transcript)

This build session's transcript is published, and the repository is public.

1. Article text, headlines, snippets and raw HTML live only under `evals/data/private/` (gitignored).
2. Read article text only with the Read tool, and only from the private files your prompt names (packets, and
   for discovery the candidates view). Do not grep, cat or print article files in a shell.
3. Write your outputs only to `evals/data/private/articles/out/<stage>/<id>.json`.
4. Never quote article text, headlines or snippets in messages, final reports, file names or committed files.
   Refer to articles by id (`art-` + 12 hex characters) and to sources by key. Key facts and rationales are
   written in your own words; the CLI rejects any that share a run of 10 consecutive words with the article.
5. Never use WebFetch, WebSearch or a browser on publisher pages. All fetching goes through the CLI, which uses
   the serving fetcher (SSRF guard, consent handling), honours robots.txt and waits at least 1 s between
   requests to one domain (EDGAR: 5 requests per second with a declared contact user agent).
6. Python only through `uv run ...` from the repository root; no inline Python.

## Private layout (`evals/data/private/articles/`)

| Path | Written by | Contents |
|---|---|---|
| `candidates/<source>.jsonl` | `discover` | id, url, title, published date, how it was found |
| `views/<source>.tsv` | `candidates` | compact id / date / title-or-slug view for an agent to Read |
| `html/<id>.html`, `fetched/<id>.json` | `fetch` | raw page; status, words, extraction method, publish date |
| `packets/gold/<id>.md`, `<id>.full.md`, `<id>.lines.json` | `prepare --stage gold` | numbered page lines |
| `packets/label/<id>.md`, `packets/adjudicate/<id>.md` | `prepare` | headline + gold body (+ both labels) |
| `out/{gold,label_a,label_b,adjudicate}/<id>.json` | agents | outputs, schemas in the stage prompts |
| `gold/<id>.txt` | `ingest --stage gold` | the validated gold body |
| `adjudicate_ids.txt` | `ingest --stage label_*` | ids where the labellers disagree or one is unsure |
| `cache/`, `discovery_log.jsonl` | `discover` | archive responses (6 h), per-run archive counts |

## Private layout (`evals/data/private/summary/`)

| Path | Written by | Contents |
|---|---|---|
| `preds/<endpoint>/<model>/`, `manifest.json` | `generate` | generated summaries per candidate, keyed by run config |
| `scores/<split>.jsonl` | `score` | per-summary check results and key-fact cosines |
| `spotcheck/<split>.md`, `review/` | `spot-check`, ad-hoc reviews | facts next to summary sentences for hand labelling |
| `judge/packets/<id>.md`, `judge/packets_<split>.txt` | `judge-prepare` | article plus anonymised candidate summaries |
| `judge/blinding/<id>.json` | `judge-prepare` | label-to-candidate map; judges never open it |
| `judge/out/<id>.json` | judge agents | outputs, schema in `50-judge.md` |
| `judge/ingested/<split>.jsonl` | `judge-ingest` | de-blinded scores that `report` aggregates |

## Main-loop runbook

```bash
# 1. discovery agents (one per source, 10-discover.md), then check the counts
uv run python -m veritrade_evals articles status

# 2. gold
uv run python -m veritrade_evals articles prepare --stage gold
#    spawn gold agents over batches of packet paths (20-gold.md); they run `ingest --stage gold` themselves

# 3. labels (the two labellers never see each other's outputs or the provisional label)
uv run python -m veritrade_evals articles prepare --stage label
#    spawn labeller A and labeller B over the same packet list (30-label.md)
uv run python -m veritrade_evals articles ingest --stage label_b --agreement   # writes adjudicate_ids.txt

# 4. adjudication
uv run python -m veritrade_evals articles prepare --stage adjudicate
#    spawn the adjudicator (40-adjudicate.md)

# 5. refill short cells, then
uv run python -m veritrade_evals articles finalize
uv run python -m veritrade_evals articles verify

# 7. summary judge (after summary_eval generate + score on the split)
uv run python -m veritrade_evals.summary_eval judge-prepare --split dev
#    spawn judge agents over batches of the paths in judge/packets_dev.txt (50-judge.md)
uv run python -m veritrade_evals.summary_eval judge-ingest --split dev
uv run python -m veritrade_evals.summary_eval report --split dev
```
