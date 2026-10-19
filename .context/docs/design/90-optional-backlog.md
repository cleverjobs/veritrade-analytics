# 90 - Optional backlog

What we would do with more time. This is a PoC and rigor was balanced per aspect (R2 in
[01-decisions.md](01-decisions.md)), so each item below marks a place where we stopped on purpose. Every item gives
the reason and the expected payoff; items are ranked by impact within their group, highest first. Items that an
earlier doc or a results file already names as future work link back to it.

## Accuracy

1. **A DSPy-style prompt optimiser instead of the hand grid** (Q7). The lean grid tried up to 8 configs per model
   with synthetic shots. An optimiser over instructions, shot selection and the decision rule, scored on dev300 F0.5
   under the 2% inversion cap, searches far wider. The largest lever found by hand was a decision rule
   (`directional_tau`, +0.218 F0.5 for `qwen3.5:0.8b`, [grid notes](../../../evals/results/grid/notes.md)).
2. **Distil the 9B decision model into the sub-second tier.** Label a large unlabelled stream of headlines and ledes
   with `clef-flash:9b-q8_0` (dev300 F0.5 0.874), then fine-tune `qwen3.5:0.8b` or a small classifier head on its
   calls. Payoff: part of the 0.874 vs 0.730 gap at the 178 ms tier, without a 10.58 GB model
   ([selection.md](../../../evals/results/selection/selection.md)).
3. **Logprob-weighted confidence for streamed `/analyze`.** Streams already carry logprobs (undocumented, see
   [01 build-time findings](01-decisions.md#build-time-findings-that-adjusted-the-plan)), so the ordinal
   `|score| >= 1` rule could become a tunable probability threshold like `/signal`'s. It needs a sign-safe score
   encoding: the qwen3.5 tokenizer fuses the space and minus sign, which hid the sign of negative scores in the
   0.8b grid.
4. **Serve the number-grounding check.** The summary eval already checks every number against the source; running
   it on each served summary and flagging sentences with unsupported numbers needs no model call. The `/summary`
   winner still has an unsupported detail (minor slips included) in 50% and a material error in 10% of dev summaries
   ([summary eval](../../../evals/results/summary/summary.md)), and numbers are the slips a trader acts on.
5. **A rubric for mixed results.** Earnings and guidance tweets hold the most wrong calls for their size (13 in 46,
   against 14 in 128 for the catch-all bucket), which fits a beat alongside a cut outlook. An aspect-level output
   (results vs outlook) or an explicit mixed rule targets the content type the brief names.
6. **Decide price moves before tuning for them.** The rubrics ask for economic impact, so the generative models miss
   33-38 of 57 directional price-move tweets, while the `/analyze` models miss only 0-1 of the 5 directional
   price-move headlines in the article dev set ([articles dev notes](../../../evals/results/articles_dev/notes.md)).
   Whether a reported move is directional is a product call; the rubric sentence the error pass left out is ready
   if the answer is yes.
7. **Revisit dropped candidates when their bugs close.** Gemma 4 (think + format bug) and `qwen3:4b-instruct-2507`
   ignoring schema property order and `think:false` under a schema. Low effort, more combined-endpoint candidates.

## Latency and serving

1. **Analyse on publish, not on request.** The 10 sources already have `feeds` / `sections` in
   [`sources.toml`](../../../backend/src/veritrade/data/sources.toml), and the evals' discovery code reads them.
   Polling them and analysing new articles ahead of time turns a request into a cache hit; the cost is inference on
   articles nobody opens.
2. **Progressive `/signal`.** Return the 0.8b verdict at 178 ms, then replace it when a 9B decision model answers
   (clef-flash lede p50 659 ms, [latency.md](../../../evals/results/latency/latency.md)). On the article dev set the
   default catches 10 of 20 directional articles and clef-flash 17 of 20, with 1 inversion each
   ([articles dev](../../../evals/results/articles_dev/summary.md)).
3. **A GPU compose profile.** A `gpu` profile with a device reservation would give Linux hosts with a CUDA GPU the
   bundled path at native-like speed. CPU-only Docker measured roughly 5-15x slower than Metal on the build machine
   (indicative, [01 build-time findings](01-decisions.md#build-time-findings-that-adjusted-the-plan)).
4. **Split-mode concurrency on one Ollama.** The latency suite runs at client concurrency 1, yet split mode fires
   `/signal` and `/summary/stream` at once against two resident models on one GPU. Measure that pair, then tune
   `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` (3 native, 2 bundled). Payoff: the split-mode TTFT a user
   actually sees.
5. **A batching server for throughput.** Beyond a handful of analysts, a server with continuous batching (vLLM, or
   llama.cpp server with parallel slots) serves many concurrent streams per GPU. Cost: a second serving stack and
   model format beside Ollama.
6. **CPU threads for the bundled Ollama.** `OLLAMA_CPUS` in [`compose.yaml`](../../../compose.yaml) sets a CPU
   quota, which does not lower llama.cpp's thread count, so a quota below the visible cores can slow inference
   rather than cap it. Pin a `cpuset` or set `num_thread` to match. Payoff: predictable CPU latency in CI and on
   shared hosts.
7. **Trim the remaining story tails.** The cascade still leaks 13-15% of tokens on Yahoo, PR Newswire and
   GlobeNewswire, mostly after the story ([03](03-extraction.md)). That costs prompt-eval time and summary focus on
   the `full` scope, not the lede.
8. **Start `/signal` on a partial body.** The lede sits early in the HTML, so parsing a streamed prefix could start
   inference before the download ends. The payoff is bounded by transfer time, so mostly on large pages (CNN
   5.7 MB).

## Data and evaluation

1. **A human-checked article set and a larger test set.** The 90 articles are labelled by Opus only (two labellers
   and an adjudicator). With 10 per class on dev a gap needs about 0.2 F0.5 to clear zero, so most article
   comparisons are ties ([articles dev](../../../evals/results/articles_dev/summary.md)). A few hundred
   human-checked articles would separate finalists that tie today.
2. **Extraction beyond the top 10.** Any public URL is accepted, but the matrix covers only profiled sources; other
   publishers fall through to JSON-LD and `density`, unmeasured. A set of other publishers would show whether
   `trafilatura-precision`, the source-agnostic candidate in [03](03-extraction.md), should back the cascade there.
3. **Entity-level and multi-entity sentiment data.** Labels are for the article's primary entity, and the public
   sets are sentence- or headline-level (FinEntity only in its single-label rows). Entity-level data would measure
   the question an analyst asks ("bullish for whom?") and back entity-targeted requests.
4. **Fresh pages for the three tuned profiles.** GlobeNewswire, PR Newswire and Yahoo were tightened on the pages
   they are scored on, so the honest cascade F1 is 0.929 and 0.959 is in-sample ([03](03-extraction.md)).
5. **Resample the threshold choice.** Grid CIs hold the operating point fixed, so they are optimistic
   ([grid summary](../../../evals/results/grid/summary.md)). A nested bootstrap or a separate threshold split would
   give honest CIs on the precision knobs.
6. **A sharper key-fact metric.** Embedding coverage agrees with hand labels on 33 of 45 facts at cosine 0.60. A
   per-fact entailment check would track the Opus judge more closely and could gate summary prompts in CI.
7. **Latency on a dedicated host and under load.** Grid and extraction timings came from a shared workstation, and
   contention invalidated a docker-cpu pass. A dedicated runner plus a 1 / 4 / 16 client sweep would give p95s fit
   for capacity planning.
8. **A label-noise pass.** 20 directional dev300 tweets get no correct call from any model
   ([grid notes](../../../evals/results/grid/notes.md)); review them before reading that recall loss as model error.
9. **A second vantage point for the source probe.** One EU residential IP chose the final 10, and datacenter IPs
   meet stricter bot management ([02](02-source-probe.md)). Probe from where the service would run.

## Product and API

1. **Entity-targeted requests.** Accept a ticker or entity and answer for it, and return per-entity calls for
   multi-entity articles, where one story can be bullish for one company and bearish for another. Depends on data
   item 3.
2. **Licensed feeds for the blocked sources.** AP News, Investing.com, Benzinga and Business Wire refuse a plain
   client at their CDN, and evasion is ruled out (Q19). Licensed APIs or feeds would add four widely read sources.
3. **A per-request precision knob.** Thresholds are frozen per model; an override of `min_p` / `directional_tau`
   within the dev300 operating curve would let a desk trade recall for precision without a redeploy.
4. **Non-HTML content.** PDF exhibits and attachments return `unsupported_content`, yet earnings decks and some
   filings arrive that way.
5. **Non-English news.** Discovery skips non-English items and every prompt is English; EU and Asian sources need
   multilingual prompts and their own labelled set.

## Operations

1. **Auth and inbound rate limiting.** The API has no authentication and no per-client limit; the only limiter is
   outbound, for EDGAR ([`client.py`](../../../backend/src/veritrade/fetch/client.py)). API keys or SSO plus per-key
   quotas come before it leaves localhost. The SSRF guard is already in place.
2. **Source DOM drift monitoring beyond the weekly live smoke.**
   [`live-smoke.yml`](../../../.github/workflows/live-smoke.yml) runs weekly and never gates. Profiles depend on selectors, and Guardian, BBC and Yahoo hash class names. A
   daily canary over fresh feed items that alerts when the cascade falls back from `profile` to `density`, body
   word counts drop, or `source_blocked` rises would catch silent degradation. No backup sources remain.
3. **Observability.** Every response already carries per-stage timings. Exported as traces and histograms
   (OpenTelemetry) with alerts on the R7 budgets, they make the latency report a live one.
4. **A shared cache (Redis) with distributed single-flight.** Fetch and result caches are in-process, so each
   replica fetches the same breaking story and a restart empties them. A shared store with a short lock keeps one
   fetch per URL across replicas.
5. **Quality monitoring.** Shadow-score a sample of served `/signal` calls with a 9B decision model and judge a
   sample of summaries offline; alert when agreement drops. It catches model, prompt or publisher drift that
   latency metrics miss.
6. **Pin models by digest and gate upgrades.** Eval manifests record model digests. Serving only digests that
   passed `freeze` and the holdout keeps served behaviour equal to measured behaviour across `ollama pull` and
   Ollama upgrades.
