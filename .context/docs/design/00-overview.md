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

- **Split**: a decision model answers `/signal` fast while a summariser streams `/summary` in parallel.
- **Combined**: one haiku-grade model produces score + summary in a single call (`/analyze`).

Reading order:

| Doc | Content |
|---|---|
| `01-decisions.md` | Every settled design decision and why |
| `02-source-probe.md` | Which publishers a plain Python client can fetch, and the final top 10 |
| `03-extraction.md` | Parser x extractor matrix results |
| `04-evals.md` | Datasets, prompt grid, precision knobs, Opus-labelled articles, results |
| `05-latency.md` | Latency budgets vs measurements, native vs container |
| `90-optional-backlog.md` | What we would do with more time |
