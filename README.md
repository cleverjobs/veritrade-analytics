# VeriTrade Analytics - real-time financial news analysis (PoC)

Paste a public financial-news URL and get a Bullish / Bearish / Neutral signal in about 0.2 s and a 2-3 sentence
summary that starts streaming in about 1.1 s (native Metal, warm models, fetch excluded). A FastAPI service fetches
and extracts the article and runs open-weight models locally through Ollama, tuned for **precision of directional
calls**, with every model, prompt and threshold chosen by a committed evaluation.

> **This is a proof of concept, and rigor is balanced per aspect.** The signal metric, the data splits, the latency
> budgets and the one-time holdout are strict; the prompt grid is deliberately lean; article labels and summary
> judgements come from Claude Opus subagents with no human check. The limits are listed
> [at the end](#limitations-and-next-steps).

| The brief asks for | Where |
|---|---|
| Design decisions | [Design decisions](#design-decisions), [`01-decisions.md`](.context/docs/design/01-decisions.md) |
| Latency trade-offs | [Latency trade-offs](#latency-trade-offs), [`05-latency.md`](.context/docs/design/05-latency.md) |
| How to run | [Quickstart](#quickstart), [Using it](#using-it) |
| How to test | [Testing](#testing) |
| AI tools and the conversation log | [AI tools used](#ai-tools-used), [`.context/transcript/`](.context/transcript/) |

The design docs read in order from [`00-overview.md`](.context/docs/design/00-overview.md); each number below links
to the committed result it comes from.

## Results at a glance

Per-endpoint defaults, picked by the rules in [`selection.md`](evals/results/selection/selection.md). Latency is
native Metal on an Apple M4 Pro, warm, concurrency 1, excluding fetch ([`latency.md`](evals/results/latency/latency.md)).

| Endpoint | Default model | Quality on dev | Warm p50 latency | Budget (p50) |
|---|---|---|---|---|
| `/api/signal` (lede) | `qwen3.5:0.8b-q8_0` | dev300 F0.5 0.730 [0.651, 0.795], 0.0% inversions | 178 ms (p95 195) | <= 500 ms: met |
| `/api/summary` | `qwen3.5:4b-q4_K_M` | Opus judge faithfulness 4.40 [4.17, 4.60] of 5, material errors 10% [0%, 20%] | TTFT 1,128 ms, complete 2,199 ms | TTFT <= 1,500 ms, complete <= 4,000 ms: met |
| `/api/analyze` | `qwen3.5:4b-q4_K_M` | dev300 F0.5 0.752 [0.664, 0.834], 0.0% inversions; faithfulness 4.47 | signal 1,098 ms, TTFT 1,292 ms, complete 2,584 ms | as `/summary`: met |

- Two model files serve all three endpoints: 4.34 GB against the 10 GB bundle cap.
- Extraction (`selectolax` + `cascade`, [03](.context/docs/design/03-extraction.md)): body F1 0.959 [0.942, 0.973]
  on 90 articles (0.929 before three profiles were tuned on the same set, the honest estimate), p50 1.2 ms.
- Live fetch, outside the budgets: median of the per-source medians 148 ms, slowest source 552 ms.
- The higher-scoring 9B decision models miss the 500 ms `/signal` budget ([the trade-off](#the-signal-trade-off)).

These are dev numbers. The frozen configs then run once on unseen test splits
([`04-evals.md`](.context/docs/design/04-evals.md#8-holdout)).
<!-- PENDING:holdout one sentence per endpoint: the winner's F0.5 [95% CI] and inversion rate on articles_test and tfns_val against its dev figure, and the /summary winner's Opus faithfulness on the 60 test articles, linking evals/results/holdout/holdout.md -->

## Quickstart

Requirements: Docker with Compose 2.20 or later (the `docker compose` plugin), Homebrew for the macOS default, and
for local development [uv](https://docs.astral.sh/uv/) plus Node 24 with pnpm (via corepack).

### 1. macOS (default): native Ollama plus the app container

Docker Desktop cannot pass the Apple GPU into a Linux container, so Ollama runs on the host with Metal.

```bash
brew install ollama && brew services start ollama
ollama pull qwen3.5:0.8b-q8_0 && ollama pull qwen3.5:4b-q4_K_M   # the two defaults, 4.34 GB
docker compose up -d                                              # builds the app image on the first run
open http://localhost:8000
```

- `scripts/dev_setup.py native` installs Ollama and sets the measured service env (`OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=3`; see
  [`preflight.json`](evals/results/latency/preflight.json)); `scripts/dev_setup.py pull --set eval` pulls the whole
  eval roster ([`models-eval.txt`](deploy/models-eval.txt), about 36 GB). Run both with `uv run python`.
- On Linux with a native Ollama, start it with `OLLAMA_HOST=0.0.0.0` so the container reaches it through
  `host.docker.internal`.

### 2. Bundled CPU Ollama (Linux, CI, no native install)

```bash
docker compose --profile bundled --env-file deploy/bundled.env up -d
```

An `ollama/ollama` container pulls the two defaults listed in [`bundled.env`](deploy/bundled.env) (written by
`veritrade_evals select --write`) into a named volume on the first start, and the app waits for the pull. CPU-only
inference was roughly 5-15x slower than Metal on the build machine ([indicative](#native-vs-docker-cpu)), so this is
a portability path, not a latency path. Leave `OLLAMA_CPUS` unset: it is a CPU quota, not a thread count, and a
quota below the visible cores can slow llama.cpp down rather than just cap it.

### 3. Local development without Docker

```bash
uv sync --all-packages
VT_CORS_ORIGINS=http://localhost:3000 uv run uvicorn veritrade.api.app:create_app --factory --port 8000 --reload
cd frontend && pnpm install && NEXT_PUBLIC_API_BASE=http://localhost:8000 pnpm dev   # second shell, port 3000
# or serve the built UI from FastAPI: pnpm build in frontend/, then start uvicorn with VT_STATIC_DIR=frontend/out
```

### Where it answers, and how to stop

- **UI**: http://localhost:8000 (Analyze) and http://localhost:8000/benchmarks/ (the eval tables from
  `evals/results/benchmarks.json`, written by `veritrade_evals report`; a labelled sample shows when it is absent).
- **API**: `http://localhost:8000/api/...`, interactive docs at `/docs`, contract in
  [`backend/openapi.json`](backend/openapi.json). Health: `curl -s http://localhost:8000/api/health`.
- **Stop**: `docker compose down`; for the bundled stack `docker compose --profile bundled --env-file
  deploy/bundled.env down` (add `-v` to delete the pulled models); `brew services stop ollama` for native Ollama.
- **Settings** are environment variables ([`config.py`](backend/src/veritrade/config.py),
  [`compose.yaml`](compose.yaml)): `VT_PORT` (host port, 8000), `VT_OLLAMA_BASE_URL`, `VT_MODELS` (restricts the
  allowlist in [`models.toml`](backend/src/veritrade/data/models.toml)), `VT_DEFAULT_MODEL_SIGNAL` / `_SUMMARY` /
  `_ANALYZE`, and `SEC_USER_AGENT` (the declared contact for SEC EDGAR's fair-access policy; set your own).

## Using it

### The UI

- Paste any public HTML article URL (the 10 tested sources are listed under the input) and press **Analyze**.
  *Split* mode (default) fires `/api/signal` and `/api/summary` together, so the signal card renders first;
  *Combined* makes one `/api/analyze` call. **Streaming** (on by default) uses the SSE endpoints, off the JSON ones.
- Pickers filled from `GET /api/models` choose a model per endpoint (models not pulled are disabled), the extractor
  (10), the parser (4) and, in split mode, the scope.
- Results: the signal card (label, abstention with the direction the model leaned, policy, confidence), the
  streamed summary, the article line (source, extraction method) and a latency bar that puts each request's server
  stage timings on one client time axis, with time-to-signal and first-token marks.

### The API

| Method and path | Request body | Response |
|---|---|---|
| `POST /api/signal` | `url`, `model`, `parser`, `extractor`, `scope` (default `lede`), `no_cache` | `meta`, `signal`, `timings` |
| `POST /api/summary` | `url`, `model`, `parser`, `extractor`, `no_cache` (scope is always `full`) | `meta`, `summary`, `truncated`, `timings` |
| `POST /api/summary/stream` | as `/api/summary` | SSE: `meta -> summary_delta* -> done` |
| `POST /api/analyze` | as `/api/signal`, `scope` default `full` | `meta`, `signal`, `entity`, `summary`, `truncated`, `timings` |
| `POST /api/analyze/stream` | as `/api/analyze` | SSE: `meta -> sentiment -> summary_delta* -> done` |
| `GET /api/models` | - | Allowlist (kind, endpoints, installed, size), per-endpoint defaults, parsers, extractors, scopes |
| `GET /api/health` | - | `ok` / `degraded`, Ollama reachability and version, backend label (`native-metal` / `docker-cpu`) |

Only `url` is required; `model` must be on the allowlist; `parser` and `extractor` default to `selectolax` and
`cascade` (`GET /api/models` lists the 4 parsers and 10 extractors); `no_cache: true` bypasses both caches.
`signal` holds `sentiment`, `abstained`, `policy` and, by policy, `confidence`, `score` (-2..2) or `probabilities`.
`timings` splits each request into fetch, consent, parse, extract and inference (wall, TTFT, load, prompt eval,
eval, tokens) and reports both cache outcomes.

```bash
URL='https://...'   # any public article URL
curl -s http://localhost:8000/api/signal -H 'content-type: application/json' -d "{\"url\": \"$URL\"}"
curl -sN http://localhost:8000/api/analyze/stream -H 'content-type: application/json' -d "{\"url\": \"$URL\"}"
```

**SSE order.** `meta` (article meta, fetch / parse / extract timings) once extraction is done; on `/analyze/stream`
one `sentiment` event as soon as the score closes (served field order: sentiment-first); `summary_delta` text
chunks; `done` with the full JSON response. A failure before the stream opens is a JSON error with its HTTP status,
after that one `error` event; a closed connection stops the upstream generation.

**Errors** share one envelope, `{"error": {"code", "message", "hint", "retryable", "upstream_status"}}`
([`errors.py`](backend/src/veritrade/errors.py), [`contracts.py`](backend/src/veritrade/api/contracts.py)); other
request-body validation errors, such as an unknown `parser`, `extractor` or `scope` or an extra field, return
FastAPI's standard 422 `{"detail": [...]}`.

| HTTP | Codes | When |
|---|---|---|
| 400 | `invalid_url`, `blocked_destination`, `model_not_allowed` | Malformed URL; SSRF guard (private or reserved address, a port other than 80/443); model not on the allowlist or disabled by `VT_MODELS`, or not served on that endpoint |
| 422 | `paywalled`, `consent_required`, `extraction_empty`, `unsupported_content`, `too_large` | Fetched, but no usable article: paywall, consent wall, under 40 words, not HTML (PDF, feed), over 10 MB |
| 502 | `source_blocked`, `upstream_http`, `schema_violation` | Publisher refused (CDN bot wall, 401/403/429/451, with a hint to a syndicated copy); other upstream HTTP error; malformed model output |
| 503 | `model_unavailable`, `llm_unreachable` | Model not pulled (the hint gives the `ollama pull` command); Ollama down |
| 504 | `upstream_timeout`, `inference_timeout` | Fetch deadline (8 s); inference timeout (90 s) |

## Architecture

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

A request is checked against the model allowlist; one pooled HTTP/2 client fetches the page behind an SSRF guard
that vets every resolved address and redirect hop, and bot walls, consent walls and paywalls become typed errors.
Extraction (default `selectolax` + `cascade`: JSON-LD `articleBody`, then the per-source profile in
[`sources.toml`](backend/src/veritrade/data/sources.toml), then a text-density fallback) cuts the `lede` or `full`
input. Ollama gets the model's frozen prompt with a JSON schema, greedy decoding and `think: false`, and the
model's confidence policy turns the answer into a label or an abstention (`backend/src/veritrade/`).

## Design decisions

Every decision was settled in an interview before any code was written; the full table, with the question each one
answers, is in [`01-decisions.md`](.context/docs/design/01-decisions.md).

- **Open-weight models on a local Ollama** ([Q1](.context/docs/design/01-decisions.md#round-1)): everything runs and
  reproduces with no API key or paid service. Native Metal is the macOS default because Docker Desktop cannot reach
  the Apple GPU; a `bundled` CPU profile covers Linux and CI.
- **Precision first** (Q2): a wrong directional call costs a trade, a missed one little, so selection uses F0.5 macro
  over bullish and bearish with inversions capped at 2% ([04](.context/docs/design/04-evals.md#1-objective-and-metrics)).
- **Abstain rather than guess** (Q3): thresholds on decision-model class probabilities, on label-token probabilities
  (`/signal`, plus `directional_tau`) or on an ordinal -2..+2 score (`/analyze`, works mid-stream); every signal
  reports `abstained`. The frozen defaults never abstain (dev300 coverage 100%): the 0.8B runs at `min_p` 0 with
  tau 0.28 and `/analyze` at \|s\| >= 1 on integer scores, so they hold back by answering neutral; only the 9B
  decision models abstain (dev300 coverage 72.9% and 26.3%).
- **Split and combined, both served** (Q8): a fast signal model next to a summariser, or one model for score, entity
  and summary; the UI switches and both were measured ([05](.context/docs/design/05-latency.md#split-against-combined)).
- **The quick signal reads the lede** (Q4): headline plus opening paragraphs, body capped at 200 tokens; `/signal`
  answers in about 7 tokens, so prompt length is its latency.
- **Stream, sentiment first** (R3): the schema puts the score first and an incremental scanner emits it as it closes.
- **One origin** ([R1](.context/docs/design/01-decisions.md#round-2)): FastAPI serves the static Next.js export; no
  proxy hop, no Node at runtime.
- **Extraction measured, not assumed** (Q10, R4): 22 parser x extractor rows on 90 real articles; the served
  `selectolax` + `cascade` is best or tied on quality and has the lowest tail of any row that finds bodies (p95
  2.4 ms, [03](.context/docs/design/03-extraction.md)).
- **Fetch like a browser, never evade** (Q18-Q20): standard headers, Yahoo consent answered with reject-all, typed
  errors; four CDN-blocked publishers were replaced, not evaded ([02](.context/docs/design/02-source-probe.md)).
- **Budgets before measurements** (R7): set first (see [the table](#latency-trade-offs)); a miss is reported, never moved.
- **Select by rules in code, test once** (Q6, Q7, R5-R8): lean grid, one error pass, freeze; `select` writes the
  defaults; the holdout runs once behind a guard keyed by the frozen config's hash ([04](.context/docs/design/04-evals.md)).
- **Publisher rights in a public repo** (Q11): skeleton test fixtures (real DOM, synthetic text); no article or
  dataset text is committed, only ids, URLs, hashes, labels and numbers.

## Latency trade-offs

Full analysis: [`05-latency.md`](.context/docs/design/05-latency.md). Native suite: 528 samples, 0 errors, cold n=3
and warm n=30 per cell on the 30 dev articles served from memory.

| Endpoint (default) | Budget, p50 | Measured p50 | Measured p95 |
|---|---|---|---|
| `/signal` lede, `qwen3.5:0.8b` | <= 500 ms | 178 ms | 195 ms |
| `/summary` SSE, `qwen3.5:4b` | TTFT <= 1,500 ms, complete <= 4,000 ms | TTFT 1,128 ms, complete 2,199 ms | complete 4,232 ms |
| `/analyze` SSE, `qwen3.5:4b` | TTFT <= 1,500 ms, complete <= 4,000 ms | signal 1,098 ms, TTFT 1,292 ms, complete 2,584 ms | complete 4,627 ms |

The budgets bind the p50 and every default meets them. The tails do not: the slowest requests are the long articles;
in every full-scope cell the two slowest of 30 carry about 2,550-2,980 prompt tokens (1,990-2,200 for `clef-flash`,
whose article input is capped at 2,000 tokens) against 825-1,074 at p50.

- **Split or combined.** Split delivers the signal at 178 ms while the summary streams (first text 1,128 ms);
  combined delivers it at 1,098 ms, six times later, but from one model and one prompt instead of two resident models
  (1.02 + 3.32 GB) and two prefills, and it scores higher on the article dev set (0.880 vs 0.685, overlapping CIs at
  n=30; dev300 0.752 vs 0.730). Split's two concurrent requests on one GPU were not measured, so its times are a lower bound.
- **Streaming.** SSE does not change the total (-2.9% to +1.1% against JSON at p50) but shows text at 38-54% of it.
  Sentiment-first, the streamed score lands 194-249 ms before the first summary text and 1.4-1.5 s before a JSON
  client would see it.
- **Lede scope.** The 0.8B answers in 178 / 195 ms (p50 / p95) on the lede against 231 / 532 ms on the full
  article, and the 9B tails grow 4-5x at full scope: lede keeps `/signal` predictable as well as fast.
- **Where the time goes.** `/signal` is prefill: the 0.8B spends 126 ms on prompt eval and 43 ms writing a 7-token
  answer. The 4B summariser prefills at about 800 tokens/s, so prompt eval is 1,029 of its 1,128 ms TTFT. Parse and
  extract take 1.2-1.6 and 1.6-2.5 ms, under 2% of the fastest `/signal`; the extractor matters through tokens
  instead (704-822 `full` tokens for competent extractors, 2,116 for `strip-tags`). A cold model load adds 0.7-1.3 s,
  paid only after a start or 30 idle minutes (`keep_alive` 30m).
- **Network and caches.** Fetch adds a median of the per-source medians of 148 ms, slowest 552 ms (GlobeNewswire),
  so a lede signal lands about 0.33 s after submit on a typical source (a rough sum of medians). A pooled HTTP/2
  client keeps connections alive for 60 s; the single-flight fetch cache (10 min, 512 entries, 200 MB; errors 60 s)
  downloads a page once for split mode's two requests; results are cached for an hour; Yahoo's consent cookie is
  reused (857 ms once, then about 120 ms).

### The `/signal` trade-off

| Model | dev300 F0.5 [95% CI] | Articles dev F0.5 | Lede p50 / p95 | File | Outcome |
|---|---|---|---|---|---|
| `qwen3.5:0.8b-q8_0` | 0.730 [0.651, 0.795] | 0.685 | 178 / 195 ms | 1.02 GB | served default |
| `clef-flash:9b-q8_0` | 0.874 [0.811, 0.937] | 0.848 | 659 / 807 ms | 10.58 GB | misses 500 ms; alone over the 10 GB bundle cap |
| `nimble:9b-q4_K_M` | 0.843 [0.773, 0.901] | 0.895 | 1,061 / 1,182 ms | 5.63 GB | misses 500 ms |

The 9B models score 0.11-0.14 higher on dev300, mostly through recall, but neither fits the budget and clef-flash
alone exceeds the bundle cap. The 0.8B is precise when it calls (dev300 precision 0.91 / 0.92) but cautious: on the
article dev set it catches 10 of 20 directional articles against 17 for clef-flash, one inversion each. The budget
was reported, never moved; both 9B models stay selectable per request (`model`, native Ollama) for 0.5-0.9 s more.

### Native vs docker-cpu

The bundled Ollama runs llama.cpp on CPU, measured through the bundled app over HTTP (warm n=10 per cell, fetch
subtracted). Two passes were discarded, and neither is committed:

- **Pass 1**: the JSON rows were biased by Ollama's prompt cache (prompt eval 118-171 ms), because each JSON cell
  re-sent prompts its SSE twin had just evaluated. Fix: a reload hook (`container_reload` in
  [`latency.py`](evals/src/veritrade_evals/latency.py)) runs `docker exec ... ollama stop` before every warm phase.
- **Pass 2**: invalidated by unrelated workloads on the same Mac (load average 50-64 against 6.9 at the native
  preflight): `qwen3.5:0.8b` decode went from about 31 ms to about 2.8 s per token and 4 `/analyze` requests timed
  out, because llama.cpp's 14 spin-barrier threads on a 14-vCPU Docker VM over a 14-core host stall whenever the
  host takes cores.

Pass 1's valid rows, indicative only (n=10, not committed): CPU-only Docker is roughly 5-15x slower than Metal on
this machine and would miss every budget, so the bundled profile is for Linux and CI.

| Endpoint | Transport | docker-cpu p50 | x native |
|---|---|---|---|
| `/signal` 0.8B lede | JSON | 2,657 ms | 14.9x |
| `/summary` 4B | SSE | TTFT 8,768 ms, complete 14,405 ms | 7.8x / 6.6x |
| `/analyze` 4B | SSE | signal 8,644 ms, TTFT 9,147 ms, complete 12,106 ms | 7.9x / 7.1x / 4.7x |

The quiet-host re-run uses the same protocol, reload hook included, and its rows go to the `docker-cpu` section of
[`latency.md`](evals/results/latency/latency.md).
<!-- PENDING:docker-cpu warm n=10 rows from the quiet-host re-run (signal 0.8b lede JSON; summary and analyze qwen3.5:4b SSE): complete p50, TTFT p50, time to signal p50, ratio to native and the preflight load average, linking evals/results/latency/latency.md -->

## Evaluation

Full method and results: [`04-evals.md`](.context/docs/design/04-evals.md). The harness imports the serving code, so
what is measured is what is served.

- **Metric**: F0.5 macro over bullish and bearish, a 2% inversion cap, and an operating point at the best F0.5
  under the cap; 95% stratified bootstrap CIs (B = 1,000) on the headline metrics.
- **Tweets and sentences** ([`splits/`](evals/data/splits/), ids, hashes and labels only): `dev300` for tuning (100
  per class from Twitter Financial News train, reweighted to its 20.2 / 15.1 / 64.7% prior); Twitter FNS validation
  (2,388), Financial PhraseBank AllAgree (2,259) and FinEntity single-label paragraphs (851) held out.
- **Articles**: 90 recent articles from the 10 sources, 3 per class per source (Yahoo 4 bearish and BBC 2, keeping
  30 per class), split 30 dev / 60 test, labelled by
  two independent Opus labellers plus an adjudicator. On the 157 double-labelled pool articles they agree on 96.2%
  (kappa 0.943); 24 ambiguous articles were replaced ([`stats.json`](evals/data/articles/stats.json)).
- **Prompt grid**: 33 prompt cells on dev300 (prompt version x output schema or field order per model), each swept
  over confidence policy and threshold; one error-analysis pass, then a freeze into
  [`frozen.json`](evals/results/frozen.json) and `models.toml`. The largest lever was a decision rule, not a prompt:
  `directional_tau` lifts the 0.8B by +0.218 [+0.141, +0.307] F0.5 ([grid notes](evals/results/grid/notes.md)). On `/analyze`, `qwen3.5:4b` leads dev300 with 0.752 against 0.685
  (`ministral-3:3b`) and 0.647 (`qwen3:4b-instruct-2507`).
- **Summaries**: deterministic checks (length, every number grounded in the source, entity named), key-fact
  coverage with `qwen3-embedding:0.6b` at cosine 0.60, and blind Opus judges (faithfulness, coverage,
  hallucinations, lead) on the 30 dev articles ([`summary.md`](evals/results/summary/summary.md)). Extraction is
  scored separately against Opus gold bodies ([`summary.md`](evals/results/extraction/summary.md)).
- **Reproducing**: `uv run python -m veritrade_evals --help` lists every step; `datasets fetch` re-downloads the
  pinned public datasets behind the tweet, sentence and paragraph splits, but the article-level runs (signal on
  articles, extraction, summaries, latency) need the stored HTML and Opus gold bodies in the gitignored
  `evals/data/private/`, which publisher rights keep out of the repo, so for those a clone can only read the
  committed results.

Summary quality of the served pair on the dev articles (n = 30):

| Endpoint, model | Checks pass | Numbers grounded | Key facts (0.60) | Opus faithfulness [95% CI] | Coverage | Hallucination flag | Material errors [95% CI] |
|---|--:|--:|--:|---|--:|--:|---|
| `/summary`, `qwen3.5:4b-q4_K_M` | 90% | 93% | 47% | 4.40 [4.17, 4.60] | 4.57 | 50% | 10% [0%, 20%] |
| `/analyze`, `qwen3.5:4b-q4_K_M` | 93% | 97% | 57% | 4.47 [4.20, 4.70] | 4.63 | 40% | 10% [0%, 20%] |

The hallucination flag counts any unsupported detail, minor slips included; a material error is faithfulness <= 3
(a wrong key number, an unsupported claim that matters). The `/summary` runner-up, `qwen3.5:2b-q4_K_M`, is faster
(TTFT 484 ms) but scores 3.97 with 30% material errors, so the 4B serves both summary endpoints.

**Holdout.** The frozen configs run once on the test splits (`articles_test`, `tfns_val`, `fpb_allagree`,
`finentity_single`), with `qwen3.5:9b-q4_K_M` as an untuned quality ceiling, and Opus judges the winners' summaries
on the 60 test articles. Nothing is re-tuned on these numbers ([protocol](.context/docs/design/04-evals.md#8-holdout)).
The one-time run is in progress; its results are added here when it completes.
<!-- PENDING:holdout replace the in-progress sentence above with a short table: per endpoint winner, F0.5 [95% CI], inversion rate and coverage on articles_test, tfns_val, fpb_allagree and finentity_single with the dev F0.5 and gap; the qwen3.5:9b ceiling; Opus faithfulness and material error rate of the two summary winners on the test articles; link evals/results/holdout/holdout.md -->

## Testing

| Layer | Where | What it guards |
|---|---|---|
| Backend unit | `backend/tests/unit` | Extractors on skeleton fixtures under all four parsers, the SSRF guard (address classes, DNS rebinding, ports), block / consent / paywall detection, prompts, confidence policies, the stream scanner split at every byte offset, the Ollama client, the API contract |
| Backend integration | `backend/tests/integration` | The ASGI app against fake publishers (skeleton pages) and a scripted Ollama: every endpoint in JSON and SSE, event order, single-flight and result caches, the error table, failures mid-stream, client disconnects |
| Evals and scripts | `evals/tests/unit`, `scripts/tests` | Metrics, splits, grid, freeze, selection, the holdout guard, latency aggregation, the report, transcript redaction |
| Frontend | `frontend/tests` | Vitest + Testing Library: SSE parser, components, and MSW integration (signal before the summary completes, JSON fallback, errors, abort on resubmit, benchmarks page) |
| End to end | `frontend/e2e` | Playwright on the compose stack: bundled CPU Ollama with the CI models, an nginx fixture site answering for the 10 publisher hostnames, split and combined streaming, a decision model with a switched extractor, a signal check per source profile, the benchmarks page |
| Live smoke | `backend/tests/live` | One real article per source; informational, never gates |

```bash
uv sync --all-packages
(cd frontend && pnpm install)            # frontend deps for check.py
uv run python scripts/check.py --fast    # ruff, format, mypy, OpenAPI drift, backend unit + evals + scripts tests, frontend lint/format/typecheck/test
uv run python scripts/check.py --full    # adds backend integration tests with a 75% coverage floor, and the frontend build
uv run pytest -m live backend/tests/live -rA   # live publishers, opt-in
```

End to end on the CI stack, which pulls `qwen3.5:0.8b-q8_0` and `tev1:0.8b-q8_0` (a CI-only decision model that
exercises the decision-model path and is never evaluated); CPU inference is slow, so test timeouts are 300 s.

```bash
docker compose -f compose.yaml -f compose.e2e.yaml up -d --wait
(cd frontend && pnpm install && pnpm exec playwright install chromium && pnpm exec playwright test)
docker compose -f compose.yaml -f compose.e2e.yaml down
```

**CI** ([`ci.yml`](.github/workflows/ci.yml), on push to `main` and pull requests): `backend` runs
`scripts/check.py --backend --full`; `frontend` checks the generated API types against `backend/openapi.json`, then
lint, format, typecheck, tests and the build; `e2e` builds the image, restores the cached Ollama models, starts the
stack, warms both models and runs Playwright, uploading logs on failure. [`live-smoke.yml`](.github/workflows/live-smoke.yml)
fetches the real publishers weekly and never gates, because publishers change markup and block datacenter IPs.

## AI tools used

- **Claude Code CLI** (2.1.30x) with **Claude Opus 5.5** (`claude-opus-5-5`) as the main loop, high reasoning effort
  and plan mode.
- **`/grill-me` skill**: Claude interviewed Kris until every decision was settled, before any code (Q1-Q20, R1-R10
  and follow-ups in [`01-decisions.md`](.context/docs/design/01-decisions.md)). The approach was handpicked from
  [`.context/openai-research.md`](.context/openai-research.md), a research and architecture proposal Kris provided.
- **Workflow tool** (multi-agent orchestration, "ultracode") and **Claude Code subagents**, every one pinned to Opus
  5.5, each owning a disjoint set of files. Opus subagents also produced the eval data that needs judgment: article
  discovery, gold bodies and key facts, two independent labellers, an adjudicator and blind summary judges (prompts
  in [`evals/opus_prompts/`](evals/opus_prompts/00-README.md)).
- **No API keys or paid services** anywhere. Serving uses only open-weight local models through Ollama.

Kris set the direction and settled every decision; Claude Code planned, wrote the code, tests, eval harness and docs,
and ran the evaluations. The full, redacted conversation (main loop, subagents, workflows) is in
[`.context/transcript/`](.context/transcript/): rendered HTML and Markdown, raw JSONL, and `meta.json` with the
harness, model, effort, subagents and token usage.

## Limitations and next steps

- **One model family labels and judges.** Article labels, gold bodies, key facts and summary judgements all come
  from Claude Opus with no human check, and with 30 dev articles the served rules' CIs span about 0.2-0.5 F0.5.
- **Precision costs recall.** The `/signal` default caught 10 of 20 directional dev articles, so users will see many
  neutrals; the 9B decision models catch more but miss the latency budget.
- **The inversion cap is unproven on articles.** On the 30 dev articles the `/signal` default inverts 1 of its 14
  directional calls (7.1% against the 2% cap); selection passes it only because the 95% CI at n=30 reaches 0%, so
  the holdout is the real check ([`selection.md`](evals/results/selection/selection.md)).
- **Tuned on tweets.** Prompts and thresholds were tuned on 300 tweets and checked on balanced article sets; real
  traffic is likely neutral-heavy, so served precision is likely lower ([threats](.context/docs/design/04-evals.md#9-threats-to-validity)).
- **Summaries still slip.** The `/summary` default has an unsupported detail (minor slips included) in 50% of dev
  summaries and a material error in 10% [0%, 20%]; numbers are not re-checked against the source at serve time.
- **Latency is one machine at concurrency 1.** Tails exceed the budgets, split mode's concurrent pair and multi-user
  load were not measured, and the docker-cpu figures are indicative.
- **Coverage and hardening.** 10 profiled English sources (others fall back to JSON-LD and density, unmeasured),
  no PDFs; no authentication or inbound rate limiting, and the caches are in-process (per replica).

Next, from [`90-optional-backlog.md`](.context/docs/design/90-optional-backlog.md): a DSPy-style prompt optimiser;
distilling the 9B decision model into the sub-second tier; a progressive `/signal` that upgrades the 0.8B verdict
when a 9B model answers; analysing articles on publish; serving the number-grounding check; a GPU compose profile.

## Layout

| Path | What |
|---|---|
| `backend/` | FastAPI service (`veritrade` package): fetch, extract, inference, API; serving config in `src/veritrade/data/` |
| `evals/` | Evaluation and benchmarking harness (`veritrade_evals` package), committed splits and results, Opus prompts |
| `frontend/` | Next.js 16 static export, served by FastAPI |
| `scripts/` | Setup, quality gate, probes, exports |
| `docker/`, `deploy/`, `compose.yaml`, `compose.e2e.yaml` | App image, bundled Ollama pull, e2e fixture site, model lists |
| `.github/workflows/` | CI and the weekly live smoke |
| `.context/` | Brief, research, design docs, AI conversation transcript |

## License

MIT, see [`LICENSE`](LICENSE).
