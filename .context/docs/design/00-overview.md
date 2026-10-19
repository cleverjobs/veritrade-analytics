# 00 - Overview

VeriTrade analysts lose minutes reading breaking news to gauge sentiment. This PoC takes a public financial-news
URL and returns a short summary plus a Bullish / Bearish / Neutral signal, optimised for latency and for
**precision of directional calls** (a wrong call costs a trade; a missed call costs little).

```
browser ──► FastAPI (same origin: /api + static Next.js export)
              │ fetch (httpx, HTTP/2, SSRF guard, consent/paywall/block detection, single-flight cache)
              │ extract (parser x extractor registry, lede/full scope, token cap)
              ├─► /api/signal   decision model (/v1/systemone, class probabilities) or tiny LLM
              ├─► /api/summary  small LLM, JSON or SSE stream
              └─► /api/analyze  one small LLM -> {score, entity, summary}, JSON or SSE stream
                         ▼
                      Ollama (native Metal on macOS, or the bundled container)
```

Two architectures are compared head to head:

- **Split**: a fast signal model (a decision model or a tiny LLM) answers `/signal` while a summariser streams
  `/summary` in parallel.
- **Combined**: one haiku-grade model produces score + summary in a single call (`/analyze`).

## Sources

Any public URL is accepted. The source profiles and the evaluation cover the final 10, chosen by a live probe
([02-source-probe.md](02-source-probe.md), [`sources.toml`](../../../backend/src/veritrade/data/sources.toml)):
Yahoo Finance, CNBC, The Motley Fool, PR Newswire, GlobeNewswire, SEC EDGAR 8-K EX-99.1, CNN Business, Fortune,
The Guardian Business and BBC Business. AP News, Investing.com, Benzinga and Business Wire refuse a plain client at
their CDN and are rejected rather than evaded.

## Outcome

The defaults were picked by the rules in [`selection.md`](../../../evals/results/selection/selection.md): `/signal`
serves `qwen3.5:0.8b-q8_0` (dev300 F0.5 0.730, lede p50 178 ms), and `/summary` and `/analyze` both serve
`qwen3.5:4b-q4_K_M` (judge faithfulness 4.40 of 5; `/analyze` dev300 F0.5 0.752, complete p50 2,584 ms), so the
bundled profile pulls just those two models, 4.34 GB. The 9B decision models score higher
(`clef-flash:9b-q8_0` dev300 F0.5 0.874, mostly through recall: its bullish / bearish precision is 0.867 / 0.905
against 0.914 / 0.918 for the 0.8B) but miss the 500 ms `/signal` budget (p50 659 ms), so the default trades recall
for speed, and the 9B models stay selectable per request on native Ollama. Each model's frozen threshold lives in
[`models.toml`](../../../backend/src/veritrade/data/models.toml) (table in
[04](04-evals.md#4-models-prompts-and-confidence-policies)); the UI pickers, every signal card and `GET /api/models`
show it, and a call below it returns "No signal" (`neutral` with `abstained: true`). Details:
[01 selection outcome](01-decisions.md#selection-outcome), [04-evals.md](04-evals.md), [05-latency.md](05-latency.md).

The one-time holdout on unseen test splits ([`holdout.md`](../../../evals/results/holdout/holdout.md)) leaves the
three defaults as frozen, with nothing re-tuned: test-article F0.5 0.801 for `/signal` (dev 0.685) and 0.871 for
`/analyze` (dev 0.880), and `/summary` judge faithfulness 4.37 (dev 4.40). Two weak spots remain: the
`/signal` default inverts 1 of 30 directional calls (3.3%, above the 2% cap as a point estimate, as on the dev
articles, 1 of 14), and `/analyze` summaries degrade on unseen articles (faithfulness 4.17, material errors 25%
against 10% on dev).

## Reading order

| Doc | Content |
|---|---|
| [`01-decisions.md`](01-decisions.md) | Every settled design decision and why, build-time findings, selection outcome |
| [`02-source-probe.md`](02-source-probe.md) | Which publishers a plain Python client can fetch, and the final top 10 |
| [`03-extraction.md`](03-extraction.md) | Parser x extractor matrix results |
| [`04-evals.md`](04-evals.md) | Datasets, prompt grid, precision knobs, Opus-labelled articles, results |
| [`05-latency.md`](05-latency.md) | Latency budgets vs measurements, native vs container |
| [`90-optional-backlog.md`](90-optional-backlog.md) | What we would do with more time |
