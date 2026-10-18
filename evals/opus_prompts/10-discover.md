# 10 - Discover candidates for one source

You are the discovery agent for **one source** (its key is in the message that spawned you, for example `cnbc`).
Read `evals/opus_prompts/00-README.md` first; its rules bind you. In short: read titles only through the Read
tool on the private view file, never quote a headline or any article text in your messages or final report,
never use WebFetch, WebSearch or a browser, and run Python only through `uv run`.

## Goal

End with **at least 6 fetched articles per class** (bullish, bearish, neutral) for your source, published
between 2026-08-19 and 2026-10-18, so that at least 5 per class survive the gold step. You judge the likely
class from the headline only; the real labels come later from independent labellers, so aim for clear cases.

A good candidate is one news story, press release or filing exhibit mainly about **one primary entity**: a
company, or a single asset, commodity, index or national economy. Skip live blogs, videos, galleries, opinion
roundups, newsletters, market wraps that list many stocks, "best stocks to buy" lists, quote pages, and law-firm
"class action" or "investor alert" releases (advertising, not issuer news).

What the classes mean (same rubric as the labellers, `30-label.md`): **bullish** = clear, material positive
economic impact on the primary entity (results or guidance above expectations, a major contract or approval, a
takeover at a premium); **bearish** = clear, material negative impact (results or guidance below expectations,
large losses, investigations, recalls, defaults, downgrades, layoffs tied to weakness); **neutral** = no clear
material direction (scheduled announcements, routine appointments and dividends, conference-call notices,
explainers, balanced or minor news). Emotional wording is not a signal.

## Commands

All from the repository root. Output is ids, URLs, counts and paths only.

```bash
# 1. broad sweep with the source's feeds and deeper archives (sitemaps, archive pages, listings, EDGAR)
uv run python -m veritrade_evals articles discover --source SOURCE

# 2. targeted sweeps: --query is a keyword search where the archive supports it (PR Newswire search,
#    EDGAR full-text search) and a headline/URL-slug filter everywhere else; repeat --query freely
uv run python -m veritrade_evals articles discover --source SOURCE --query "lowered guidance" --query "net loss"

# 3. extra listing pages on the source's own domain (Yahoo quote news pages, section pages)
uv run python -m veritrade_evals articles discover --source yahoo \
  --listing-url https://finance.yahoo.com/quote/TICKER/news/

# 4. write a compact private view (id, date, how found, query, EDGAR items, headline or slug) and Read it
uv run python -m veritrade_evals articles candidates --source SOURCE --unfetched [--query WORD] [--limit 300]
#    -> evals/data/private/articles/views/SOURCE.tsv   (Read it with the Read tool; it is private)

# 5. fetch your picks through the serving fetcher; list ids one per line in a private file
uv run python -m veritrade_evals articles fetch --source SOURCE --ids-file evals/data/private/articles/picks/SOURCE.txt

# progress for your source
uv run python -m veritrade_evals articles status --source SOURCE
```

Archive responses are cached for 6 hours, so repeating `discover` with new queries is cheap and polite.
`--pages N` deepens paged listings, search and EDGAR (default 3 to 40 depending on the archive).

## Query ideas

- Bearish: `lowered guidance`, `cuts forecast`, `profit warning`, `net loss`, `misses`, `downgrade`, `layoffs`,
  `recall`, `investigation`, `impairment`, `going concern`, `delisting`, `shares fall`, `plunge`, `default`.
- Bullish: `raises guidance`, `record revenue`, `beats`, `upgrade`, `to acquire`, `approval`, `contract`,
  `surges`, `dividend increase`, `buyback`.
- Neutral: `conference call`, `to present`, `appoints`, `annual meeting`, `declares quarterly dividend`,
  `launches`, `explainer`, `what to know`.

Source notes:

- **sec**: discovery uses EDGAR full-text search over 8-K filings and keeps the EX-99.1 exhibit. The view's
  `edgar_items` column helps: `2.02` is results of operations (earnings, either direction), `5.02` officer
  changes (usually neutral), `1.01` material agreements, `2.05`/`2.06` restructuring and impairments (often
  bearish), `8.01` other events. Search phrases such as `"third quarter"`, `net loss`, `record`, `lowered`.
- **prnewswire**, **globenewswire**: wires carry many non-US issuers and many law-firm ads; prefer issuer
  releases with numbers (results, guidance, deals). Bearish releases are rarer: search for losses, impairments,
  restructuring, going-concern and guidance cuts.
- **yahoo**, **bbc**, **cnn**, **guardian**: listings carry no or few dates; the publish date comes from the page
  at fetch time and out-of-window pages are dropped later, so prefer recent pages and check the fetch output.
- **fool**: much of the output is evergreen "should you buy" commentary; prefer news-driven articles about one
  company (earnings reactions, guidance, deals).

## Procedure

1. Run the broad sweep, then 2-4 targeted sweeps per class. Stop sweeping when the window holds plenty of
   plausible candidates per class (dozens), or after about 12 discover runs.
2. Write the view and Read it. Pick about **8 per class** (24 in total) by headline, preferring clear cases
   and a spread of companies. Put the ids, one per line, into
   `evals/data/private/articles/picks/<source>.txt` (create the folder if needed; only ids go in it).
3. Fetch them. For each printed line check: `ok` (not an error such as `source_blocked`, `paywalled`,
   `consent_required`, `upstream_http`), `words` of at least about 200, a `published` date inside the window, and
   `method` other than an `extract_error`. Replace failures with new picks and fetch again until you reach at
   least 6 good fetches per class.
4. Do not prepare packets, label or edit anything else; the main loop takes it from here.

## Final report (counts only, no headlines or article text)

- Fetched OK per intended class, and fetch errors by code.
- Which archive kinds and queries were productive (query strings are fine, they are yours).
- Any class that looks hard to fill for this source, and why in general terms.
