# 05 - Latency

Question (R7): how fast does each endpoint answer, where does the time go, and which trade-offs shaped the
served defaults? This chapter builds on [00-overview.md](00-overview.md) (endpoints, split / combined),
[01-decisions.md](01-decisions.md) (R7, Q1), [03-extraction.md](03-extraction.md) and [04-evals.md](04-evals.md)
(candidates, selection). Native numbers: [latency.json](../../../evals/results/latency/latency.json), table in
[latency.md](../../../evals/results/latency/latency.md), session `20261018T200825Z`. As a PoC (R2), the suite
measures one machine at one concurrency, in depth; what it leaves out is under "Levers".

## Budgets

Set before any measurement (R7), for native Metal, warm models, excluding fetch: **`/signal` lede p50 <= 500 ms;
`/summary` and `/analyze` TTFT p50 <= 1,500 ms and complete p50 <= 4,000 ms**. A missed budget is reported,
never moved. `/signal` at full scope has no budget (n/a); JSON variants are checked on completion only. The
selection rule (01, Q2) takes the most accurate candidate inside the budget, so these numbers and
[selection.md](../../../evals/results/selection/selection.md) decide the served defaults.

## Method

From [latency.py](../../../evals/src/veritrade_evals/latency.py) and the method block of `latency.md`:

- **App**: the real FastAPI app (`create_app`) in process over an ASGI transport, with a local fetcher serving
  the stored HTML of the 30 dev articles from memory, so the network stays out. Parser `selectolax`, extractor
  `cascade` (the serving default), `no_cache=true` so neither cache can answer.
- **Concurrency 1**, models grouped so each loads once, every other model unloaded first. **Cold**, n=3 per
  cell: the first 3 dev articles, the model unloaded before each request.
- **Warm**, n=30 per cell, one per dev article. Before each cell's warm phase the model is unloaded, then warmed
  on a synthetic page that is not a dev article: Ollama reuses a cached prompt prefix, so warming on a measured
  article, or a JSON cell re-sending its SSE twin's prompts, would make prompt eval look free.
- **Clocks** server side from request receipt (the in-process client agreed within 3 ms). TTFT is the first
  `summary_delta` event; time to signal is the `sentiment` event (for JSON, the total). **Stages**: parse,
  extract, inference wall, load, prompt eval, eval. Decision models answer through `/v1/systemone`, which
  reports wall time and input tokens only, hence their `-` columns.
- **Machine** ([preflight.json](../../../evals/results/latency/preflight.json)): Apple M4 Pro, 14 cores, 48 GB,
  macOS 26.6.2, Ollama 0.40.6 as a brew service with `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=30m`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=3`. No Docker container ran; 1-minute load average
  6.89; the preflight warned that swap held 23,200 of 25,600 MB (free memory 70%). **Output**: 528 samples (16
  cells x 33), 0 errors, ids and numbers only in the
  [raw JSONL](../../../evals/results/latency/raw/20261018T200825Z.jsonl).

## Native results

Warm p50 / p95 in ms; stage columns are warm p50s. Copied from `latency.md`.

| endpoint | model | scope | transport | n warm | p50 ms | p95 ms | TTFT p50 | to signal p50 | cold p50 | parse | extract | inference | prompt eval | eval | budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| /signal | `qwen3.5:0.8b-q8_0` | lede | JSON | 30 | 178 | 195 | - | 178 | 992 | 1 | 2 | 173 | 126 | 43 | met (p50 excl. fetch 178/500) |
| /signal | `qwen3.5:0.8b-q8_0` | full | JSON | 30 | 231 | 532 | - | 231 | 1,077 | 1 | 2 | 226 | 178 | 41 | n/a |
| /signal | `clef-flash:9b-q8_0` | lede | JSON | 30 | 659 | 807 | - | 659 | 6,064 | 2 | 2 | 654 | - | - | missed (p50 excl. fetch 659/500) |
| /signal | `clef-flash:9b-q8_0` | full | JSON | 30 | 1,893 | 4,007 | - | 1,893 | 8,091 | 2 | 2 | 1,889 | - | - | n/a |
| /signal | `nimble:9b-q4_K_M` | lede | JSON | 30 | 1,061 | 1,182 | - | 1,061 | 2,219 | 2 | 2 | 1,057 | - | - | missed (p50 excl. fetch 1,061/500) |
| /signal | `nimble:9b-q4_K_M` | full | JSON | 30 | 2,321 | 5,242 | - | 2,321 | 3,194 | 2 | 2 | 2,316 | - | - | n/a |
| /summary | `qwen3.5:2b-q4_K_M` | full | SSE | 30 | 1,272 | 2,094 | 484 | - | 2,305 | 2 | 2 | 1,267 | 423 | 827 | met (complete p50 1,272/4,000, TTFT p50 484/1,500) |
| /summary | `qwen3.5:2b-q4_K_M` | full | JSON | 30 | 1,310 | 2,071 | - | - | 2,541 | 2 | 2 | 1,305 | 417 | 869 | met (complete p50 1,310/4,000) |
| /summary | `qwen3.5:4b-q4_K_M` | full | SSE | 30 | 2,199 | 4,232 | 1,128 | - | 2,882 | 2 | 2 | 2,194 | 1,029 | 1,075 | met (complete p50 2,199/4,000, TTFT p50 1,128/1,500) |
| /summary | `qwen3.5:4b-q4_K_M` | full | JSON | 30 | 2,194 | 4,211 | - | - | 2,891 | 2 | 2 | 2,190 | 1,018 | 1,070 | met (complete p50 2,194/4,000) |
| /analyze | `ministral-3:3b` | full | SSE | 30 | 2,242 | 3,976 | 1,051 | 802 | 3,253 | 2 | 2 | 2,236 | 713 | 1,361 | met (complete p50 2,242/4,000, TTFT p50 1,051/1,500) |
| /analyze | `ministral-3:3b` | full | JSON | 30 | 2,219 | 3,974 | - | 2,219 | 3,516 | 2 | 2 | 2,213 | 713 | 1,350 | met (complete p50 2,219/4,000) |
| /analyze | `qwen3:4b-instruct-2507-q4_K_M` | full | SSE | 30 | 2,234 | 4,979 | 1,201 | 1,118 | 3,479 | 2 | 2 | 2,229 | 862 | 1,236 | met (complete p50 2,234/4,000, TTFT p50 1,201/1,500) |
| /analyze | `qwen3:4b-instruct-2507-q4_K_M` | full | JSON | 30 | 2,243 | 4,969 | - | 2,243 | 3,403 | 2 | 2 | 2,237 | 868 | 1,242 | met (complete p50 2,243/4,000) |
| /analyze | `qwen3.5:4b-q4_K_M` | full | SSE | 30 | 2,584 | 4,627 | 1,292 | 1,098 | 3,570 | 1 | 2 | 2,578 | 992 | 1,539 | met (complete p50 2,584/4,000, TTFT p50 1,292/1,500) |
| /analyze | `qwen3.5:4b-q4_K_M` | full | JSON | 30 | 2,580 | 4,582 | - | 2,580 | 3,544 | 2 | 2 | 2,575 | 992 | 1,504 | met (complete p50 2,580/4,000) |

### What meets and what misses

- **Every generative candidate meets its budgets at p50.** The R7 sanity gate in 01 predicted a TTFT miss for
  `qwen3.5:4b` (2.4 s on ~1.6k-token synthetic inputs); real dev articles after the cascade are shorter (an
  825-token `/summary` prompt at p50), so its TTFT lands at 1,128 ms.
- **Both 9B decision models miss the `/signal` budget** (659 and 1,061 ms; see the trade-off section below).
- **The tails are outside the budgets, which bind only the p50.** Streamed TTFT p95 is 1,108 ms for the 2B and
  2,469 to 2,995 ms for the 3-4B models; complete p95 passes 4 s for `/summary` 4B (4,232) and for `/analyze` on
  both 4B models (4,627 and 4,979). The tail is the long articles: in the raw samples, the two slowest of 30
  requests in every full-scope cell carry about 2,550 to 2,980 prompt tokens (1,990 to 2,200 for `clef-flash`,
  whose article input is capped at 2,000 tokens) against 825 to 1,074 at p50, and write longer.
- In each `qwen3.5:2b` cell, 1 of 30 warm summaries hit `num_predict` and was served cut at a sentence with
  `truncated: true` (01); no other cell truncated. All 165 streamed samples, cold and warm, kept the event
  order (`meta` first, `done` last, one `sentiment` before `done` on `/analyze`).

### Where the time goes

Warm p50s from `latency.json`; per-token rates are ratios of p50s, so approximate.

| endpoint | model | prompt tokens | prompt eval ms | output tokens | eval ms | prompt eval share | decode ms/token |
|---|---|---|---|---|---|---|---|
| /signal lede | `qwen3.5:0.8b-q8_0` | 530 | 126 | 7 | 43 | 75% | 6.1 |
| /summary SSE | `qwen3.5:2b-q4_K_M` | 825 | 423 | 92 | 827 | 34% | 9.0 |
| /summary SSE | `qwen3.5:4b-q4_K_M` | 825 | 1,029 | 68 | 1,075 | 49% | 15.8 |
| /analyze SSE | `ministral-3:3b` | 979 | 713 | 92 | 1,361 | 34% | 14.8 |
| /analyze SSE | `qwen3:4b-instruct-2507-q4_K_M` | 970 | 862 | 88 | 1,236 | 41% | 14.0 |
| /analyze SSE | `qwen3.5:4b-q4_K_M` | 963 | 992 | 96 | 1,539 | 39% | 16.0 |

- `/signal` is a prefill problem: 7 output tokens (label schema, `num_predict` 16), so prompt length decides
  the time. That is why the quick signal reads the lede (01, Q4).
- The summarisers differ. The 2B spends two thirds decoding; the 4B prefills at about 800 tokens/s against
  about 1,950 for the 2B, so prompt eval is half its time and almost all of its TTFT (1,029 of 1,128 ms).
  Shorter inputs move TTFT; shorter outputs move completion. Warm load is under 1 ms, and everything outside
  inference (parse, extract, input, response) is 3 to 6 ms.

### Streaming, JSON and field order

Same model, same articles: the transport does not change the total (SSE minus JSON is -2.9% to +1.1% at p50),
but SSE shows text at 38 to 54% of it (`/summary` 4B: 1,128 of 2,199 ms). The UI defaults to streaming.

**Sentiment-first field order** (R3) is what makes the streamed signal useful in `/analyze`. The incremental
scanner emits the `sentiment` event as soon as the score closes: 194 to 249 ms before the first summary text,
and 1.4 to 1.5 s before a JSON client would see it (signal 802 / 1,098 ms streamed against 2,219 / 2,580 ms JSON
for `ministral-3:3b` / `qwen3.5:4b`). `qwen3:4b-instruct-2507` ignores schema property order and writes the
entity first (01), so its signal leads the summary by only 83 ms.

### Lede against full scope

| model | lede p50 / p95 | full p50 / p95 | prompt tokens lede / full |
|---|---|---|---|
| `qwen3.5:0.8b-q8_0` | 178 / 195 | 231 / 532 | 530 / 1,074 |
| `clef-flash:9b-q8_0` | 659 / 807 | 1,893 / 4,007 | 286 / 831 |
| `nimble:9b-q4_K_M` | 1,061 / 1,182 | 2,321 / 5,242 | 448 / 1,003 |

Full scope costs the 0.8B little at the median but widens every tail: its p95 crosses 500 ms, and the 9B tails
grow 4 to 5 times. Lede keeps `/signal` predictable as well as fast.

### Cold start, parse and extract

- **Cold**: the model load adds 0.7 to 1.3 s over warm for the generative models (load p50 785 to 1,328 ms),
  0.9 to 1.2 s for `nimble` and 5.4 to 6.2 s for `clef-flash` (a 10.58 GB file); n=3, so indicative. A cold 4B
  misses TTFT (2,132 ms for `/summary`). Every call sends `keep_alive` 30m, so only the first request after a
  start or 30 idle minutes pays it.
- **Parse and extract**: 1.2 to 1.6 ms and 1.6 to 2.5 ms at p50 in every cell, under 2% of even the fastest
  `/signal`, so the parser is not a latency lever (03: every parser yields the same body; `selectolax` is
  chosen for speed and footprint). The extractor matters through tokens instead: competent extractors send
  704 to 822 `full` tokens against 2,116 for `strip-tags` (03), and prompt eval scales with that.

## Network

The suite excludes fetch on purpose. Live cost from the source probe ([02-source-probe.md](02-source-probe.md),
EU residential ISP, 2026-10-18): median of the per-source medians 148 ms, slowest 552 ms.

| source | fetch ms (median) | source | fetch ms (median) |
|---|---|---|---|
| bbc | 26 | globenewswire | 552 |
| cnbc | 161 | guardian | 116 |
| cnn | 35 | prnewswire | 34 |
| fool | 202 | sec | 135 |
| fortune | 367 | yahoo | 477 |

As a rough sum of medians (not measured end to end), a lede signal lands about 0.33 s after submit on a
typical source and about 0.73 s on the slowest; on 4 of the 10 sources the fetch takes longer than the signal.
What the service does about it:

- **One pooled HTTP/2 client** (`veritrade.fetch.client`): keep-alive connections (60 s) let a second article
  from the same publisher skip TCP and TLS setup. Deadline 8 s, connect 3 s.
- **Single-flight fetch cache** (`veritrade.fetch.cache`): split mode fires `/signal` and `/summary` at once,
  and the second request joins the first one's in-flight download (`timings.cache.fetch` is `"joined"`), so
  the page is fetched once. Pages live 10 minutes (512 entries, 200 MB); typed fetch errors live 60 s, so a
  refusing publisher is not hit per request.
- **Result cache** (`veritrade.services.cache`): finished responses keyed by (endpoint, url, model, prompt
  version, parser, extractor, scope) for 1 hour; a hit returns with zero stage timings and `meta.cached`.
- **Consent cookie**: the Yahoo reject-all form costs 857 ms once; the shared cookie jar skips it afterwards
  (about 120 ms, 02).

## Split against combined

The UI offers both, split by default. Defaults first, then the other combined candidates; each row measured
alone at concurrency 1.

| arrangement | signal | first summary text | summary complete | resident model files |
|---|---|---|---|---|
| split: `/signal` 0.8B lede + `/summary` `qwen3.5:4b` SSE | 178 | 1,128 | 2,199 | 2 (1.02 + 3.32 GB) |
| combined: `/analyze` `qwen3.5:4b` SSE (default) | 1,098 | 1,292 | 2,584 | 1 (3.32 GB) |
| combined: `/analyze` `ministral-3:3b` SSE | 802 | 1,051 | 2,242 | 1 (2.95 GB) |
| combined: `/analyze` `qwen3:4b-instruct-2507` SSE | 1,118 | 1,201 | 2,234 | 1 (2.50 GB) |

- **Split** gives the fastest signal, about 0.2 s, while the summary streams, at the cost of two prompts, two
  resident models and two prefills of the same article.
- **Combined** yields the signal at about 0.8 to 1.1 s and the summary by about 2.2 to 2.6 s, from one model
  and one prompt. Its signal is not worse: `qwen3.5:4b` scores dev300 F0.5 0.752 against 0.730 for the 0.8B,
  and 0.880 against 0.685 on the 30 dev articles (wide CIs, [selection.md](../../../evals/results/selection/selection.md)).
- **Not measured**: split mode's two requests running at once on one GPU; the split row joins two isolated
  measurements, so its summary times are a lower bound for real use.

## /signal: accuracy against latency

| model | dev300 F0.5 [95% CI] | articles dev F0.5 [95% CI] | lede p50 / p95 ms | cold p50 ms | file GB | outcome |
|---|---|---|---|---|---|---|
| `qwen3.5:0.8b-q8_0` | 0.730 [0.651, 0.795] | 0.685 [0.473, 0.833] | 178 / 195 | 992 | 1.02 | served default |
| `clef-flash:9b-q8_0` | 0.874 [0.811, 0.937] | 0.848 [0.708, 0.963] | 659 / 807 | 6,064 | 10.58 | misses 500 ms; alone over the 10 GB bundle cap (R8) |
| `nimble:9b-q4_K_M` | 0.843 [0.773, 0.901] | 0.895 [0.771, 0.976] | 1,061 / 1,182 | 2,219 | 5.63 | misses 500 ms |

The 9B decision models are 0.11 to 0.14 F0.5 better on dev300 and neither fits the budget, so the 0.8B wins as
the only candidate inside it. The budget was reported, not moved to admit them. Both stay on the allowlist: on
native Ollama a request picks one with the `model` parameter and trades about 0.5 to 0.9 s for a higher F0.5,
mostly recall. Whether the gap holds on unseen data is a holdout question ([04-evals.md](04-evals.md)).

<!-- PENDING:holdout articles_test and tfns_val F0.5 of the three /signal candidates at lede scope, to show whether the 9B accuracy lead holds on unseen data -->

## docker-cpu

Docker Desktop cannot pass Apple's GPU into a Linux container (01, Q1), so the bundled Ollama runs llama.cpp on
CPU. `latency docker-cpu` measures the bundled defaults over HTTP (`/signal` 0.8B lede JSON; `/summary` and
`/analyze` `qwen3.5:4b`, SSE and JSON), warm n=10 per cell; the app fetches live, so fetch and consent are
subtracted, and it warms up on an 11th dev article.

Two earlier passes were discarded, and neither is committed:

1. **Pass 1 found a prompt-cache bias.** The bundled Ollama has no host port, so the native unload was not
   available, and each JSON cell re-sent the prompts its SSE twin had just evaluated on the same articles;
   those rows showed prompt eval of 118 to 171 ms. Fix: a reload hook (`container_reload` in `latency.py`) runs
   `docker exec ... ollama stop <model>` before every warm phase and waits until `ollama ps` drops it,
   mirroring the native unload.
2. **Pass 2 was invalidated by host contention.** Unrelated workloads on the same Mac pushed the load average
   to 50-64 (6.9 at the native preflight). `qwen3.5:0.8b` decode went from about 31 ms/token to about
   2.8 s/token and 4 `/analyze` requests timed out. Root cause: the bundled Ollama runs llama.cpp with 14
   threads on a 14-vCPU Docker VM over a 14-core host, and its spin-barrier threads stall whenever the host
   takes cores, so CPU inference collapses instead of slowing in proportion.

Indicative only (pass-1 rows the cache bias did not touch, n=10, not committed):

| endpoint | model | transport | complete p50 | TTFT p50 | to signal p50 | x native (complete / TTFT / signal) |
|---|---|---|---|---|---|---|
| /signal | `qwen3.5:0.8b-q8_0` lede | JSON | 2,657 | - | - | 14.9x |
| /summary | `qwen3.5:4b-q4_K_M` | SSE | 14,405 | 8,768 | - | 6.6x / 7.8x |
| /analyze | `qwen3.5:4b-q4_K_M` | SSE | 12,106 | 9,147 | 8,644 | 4.7x / 7.1x / 7.9x |

In pass 1, CPU-only Docker was roughly 5 to 15 times slower than Metal on this machine and would miss every
budget. That is why native Ollama is the macOS default and the bundled profile is for Linux and CI: a portability path, not
a latency path. `OLLAMA_CPUS` in `compose.yaml` sets a `cpus:` quota, which does not lower llama.cpp's thread
count, so a quota below the visible core count can make CPU inference slower rather than just capped; a cpuset
or `num_thread` is the fix (backlog). The re-run repeats pass 1's protocol, reload hook included, on a quiet
host, and its rows go in the `docker-cpu` section of `latency.md`.

<!-- PENDING:docker-cpu warm n=10 table from the quiet-host re-run (signal 0.8b lede JSON; summary and analyze qwen3.5:4b SSE and JSON): complete p50/p95, TTFT p50, time to signal p50, ratio to native, preflight load average -->

## Levers

Used:
- **Input and model size**: lede scope for `/signal` (half the tokens of full, p95 195 ms instead of 532); a
  cascade that sends about 720 `full` tokens where `strip-tags` sends about 2,100 (03); the smallest `/signal`
  model that clears the accuracy filters, answering in 7 tokens.
- **Perceived latency**: SSE, sentiment-first field order and an incremental JSON scanner.
- **Warm models**: `keep_alive` 30m on every call; `OLLAMA_MAX_LOADED_MODELS` 3 native (2 bundled) keeps both
  split-mode models resident; flash attention and a q8_0 KV cache in the Ollama env.
- **Bounded generation**: `think: false`, greedy decoding, per-endpoint `num_predict`, and the served summary
  cap of 3 sentences / 60 words (01).
- **Network and hops**: pooled HTTP/2, single-flight fetch cache, result cache, consent cookie reuse; the UI is
  served by the same FastAPI process (R1), and the fast parsers extract inline, without a thread hand-off.

Left: the latency items in [90-optional-backlog.md](90-optional-backlog.md) (progressive `/signal`, split-mode
concurrency, a GPU compose profile, CPU threads for the bundled Ollama, latency under load), plus:
- Preload the default models at app start, so the first request does not pay 0.7 to 1.3 s of load.
- Trim the 4B prompts: prompt eval is half of `/summary` and nearly all of its TTFT.
- Start the fetch when a URL is pasted, before submit, to hide part of the network cost.

## Reproduce

```bash
uv run python -m veritrade_evals latency preflight   # records the machine; fails while Docker containers run
uv run python -m veritrade_evals latency run         # native suite, resumable; writes latency.{json,md}
uv run python -m veritrade_evals latency docker-cpu --base-url http://localhost:8000   # bundled stack up first
uv run python -m veritrade_evals latency report      # re-aggregates the raw samples
```

The native run needs the private article HTML (`articles verify` passes) and an otherwise idle machine.
