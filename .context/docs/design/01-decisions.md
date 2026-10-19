# 01 - Design decisions

Every decision below was settled with Kris in a `/grill-me` interview (2026-10-16 to 2026-10-18) before any code
was written. Background facts came from read-only research agents (local machine, Ollama library, Docker/CI, news
sources, datasets, transcript format) and were re-verified against primary sources where they were surprising
(the Ollama decision-model API). The full conversation is in `.context/transcript/`.

Guiding principle (Kris, R2): **this is a PoC - balance the level of rigor per aspect**, no exhaustive sweeps.

## Round 1

| # | Question | Decision | Why |
|---|---|---|---|
| Q1 | Where does the Ollama server run? | Native Ollama on the host is the documented default on macOS; a `bundled` compose profile runs `ollama/ollama` (CPU) for Linux/CI; `OLLAMA_BASE_URL` switches; benchmarks are labelled `native-metal` / `docker-cpu` | Docker Desktop cannot pass Apple's GPU into Linux containers (~0.2x prompt-eval, ~0.4x generation vs native); reviewers on Macs will have Ollama anyway |
| Q2 | What does "precision" mean, how is a winner picked? | Positives are directional calls (bullish/bearish); neutral is the safe default. Selection metric F0.5 macro over {bullish, bearish}; bullish<->bearish inversions capped at 2%; winner = best F0.5 inside the latency budget, else the Pareto knee | A wrong call costs a trade, a missed call costs little; F0.5 cannot be gamed by "always neutral" |
| Q3 | Sentiment output contract | 3-class `sentiment` + `abstained`; compare two precision knobs for generative models (label-token logprob threshold vs ordinal -2..+2 \|score\| threshold) and keep the better one; decision models threshold their native class probabilities | Ollama emits logprobs, but only for non-streaming calls; the ordinal knob also works when streaming |
| Q4 | What does the quick signal read? | `scope=lede` (headline + opening paragraphs) by default, `scope=full` available; both evaluated for accuracy and speed | Lede matches headline-level datasets and decision-model context limits; prompt length drives latency |
| Q5 | Article-level gold set | 90 real articles (10 sources x 3 classes x 3), labelled by Opus subagents, no human check | Public datasets are sentences/headlines; real requests are full articles |
| Q6 | Public datasets and splits | Dev: stratified 300 from Twitter Financial News train. Holdout: full Twitter validation + Financial PhraseBank AllAgree (downloaded at eval time, never committed, CC BY-NC-SA) + FinEntity single-label paragraphs | Twitter FNS already uses Bearish/Bullish/Neutral and has official splits; FPB is the cleanest cross-domain check |
| Q7 | Prompt tuning method | Automated grid per model, then one manual error-analysis pass, freeze, run holdout once. Optional backlog starts with a DSPy-style optimizer loop | Reproducible and explainable within a PoC budget |
| Q8 | API surface | `/api/signal`, `/api/summary`, `/api/analyze`, `/api/models`, `/api/health`; model allowlist; per-URL fetch cache with single-flight; result cache; timing breakdown in every response; SSE **and** JSON variants for `/summary` and `/analyze` | Streaming improves perceived latency; both variants make the comparison measurable |
| Q9 | How the UI is served | Reopened in round 2 (R1) | - |
| Q10 | Extraction | Full parser x extractor comparison matrix, switchable from the UI, results published | Extraction quality and speed both feed end-to-end latency |
| Q11 | Fixtures and publisher rights | Skeleton fixtures (real DOM structure, synthetic text) + scheduled live smoke; article texts never committed (URL + hash + labels only) | Respect each publisher's content rights in a public repo |
| Q12 | CI | Everything on GitHub Actions, including e2e with a real tiny model | Free for public repos |
| Q13 | Transcript | Redacted raw JSONL + rendered Markdown/HTML + `meta.json` under `.context/transcript/`, stripped of anything not suitable for a public repo | The brief asks for the AI log; Kris wants reviewers to read it |
| Q14 | Git workflow | Milestone commits straight to `main`, no feature branches; repo-local author Kristiyan Dimitrov; public repo `cleverjobs/veritrade-analytics` | Solo PoC |
| Q15 | Tooling | Python 3.13 uv workspace (`backend`, `evals`), ruff + mypy; Node 24 + pnpm, Next.js 16 strict TS, ESLint + Prettier, Vitest + RTL + MSW, Playwright | Conventional, fast |
| Q16 | Native Ollama install | `brew install ollama` + `brew services` | Metal acceleration for evals and the native path |
| Q17 | Model roster | See R2 and the follow-ups | - |
| Q18 | Top-10 sources | Yahoo Finance, CNBC, AP News, Investing.com, Motley Fool, Benzinga, PR Newswire, GlobeNewswire, SEC EDGAR 8-K EX-99.1, CNN Business; backups Fortune, Business Wire, Guardian, BBC; the live probe decides | Reviewer likelihood x accessibility; the brief names earnings reports, EDGAR is their primary source |
| Q19 | Fetch identity | Standard browser header set, no evasion (no captcha solving, proxies or TLS impersonation); typed `source_blocked` / `paywalled` / `consent_required` errors with hints; `SEC_USER_AGENT` for EDGAR | Neither end of the spectrum |
| Q20 | Yahoo EU consent wall | Submit "reject all" and cache the cookie; fall back to `consent_required` | Gets the article without granting tracking consent on anyone's behalf |

## Round 2

| # | Question | Decision | Why |
|---|---|---|---|
| R1 | UI serving topology | FastAPI serves the static Next.js export from the same origin (multi-stage Docker build); dev uses `pnpm dev` + CORS for localhost | No proxy hop on the latency path, no nginx, no Node at runtime; the brief requires a Python backend anyway |
| R2 | Decision models | `clef-flash:9b-q8_0` + `nimble:9b-q4_K_M`, with `qwen3.5:0.8b` as the generative baseline | PoC balance; clef-flash by request, nimble best published accuracy among the rest |
| R3 | Streaming `/analyze` | Typed SSE events `meta` -> `sentiment` -> `summary_delta*` -> `done`; JSON field order (sentiment-first vs summary-first) is an eval dimension | Sentiment arrives after a handful of output tokens when it is generated first |
| R4 | Extraction matrix + reference | 4 parsers x 4 own strategies + trafilatura (default, precision), readability-lxml, newspaper4k, justext, strip-tags; reference bodies produced by Opus, kept local (gitignored) | Comparable quality metric without publishing article text |
| R5 | Article-set procedure | Opus discovery via RSS/sections (last ~60 days, fetchable by our fetcher); two independent Opus labellers + adjudicator, ambiguous articles replaced; agreement reported; split 30 dev / 60 test | Stand-in for a human check |
| R6 | Summary evaluation | Deterministic checks (numbers grounded in source, entity named, length) + key-fact coverage via a local embedding model + Opus-as-judge on finalists | Cheap reproducible signal plus a strong judge where it matters |
| R7 | Latency budgets (native, warm, excl. fetch) | `/signal` lede p50 <= 500 ms; `/summary` and `/analyze` TTFT <= 1.5 s, complete p50 <= 4 s; unrealistic budgets are surfaced, not silently moved | Concrete selection gate |
| R8 | Bundled models | Per-endpoint winners if they total <= ~10 GB, else the best finalist that fits; all other Docker containers stopped during setup | Docker Desktop VM has 15.6 GiB |
| R9 | Transcript scope | Only this build session (main + subagents + workflows) | No other relevant sessions |
| R10 | Docs layout | `.context/docs/design/NN-*.md`, backlog in `90-optional-backlog.md`, eval reports in `evals/results/` | Book-order reading |

## Follow-ups

- Generative roster trimmed: summary {`qwen3.5:2b`, `qwen3.5:4b`}; combined {`qwen3.5:4b`, `qwen3:4b-instruct-2507`,
  `ministral-3:3b`}; `qwen3.5:9b` only as a holdout quality ceiling; Gemma 4 dropped (open think+format bug).
- Lean prompt grid: 2 rubric styles x few-shot {0, 4} x field order where relevant (<= 8 configs per model).
- Stopped Docker containers stay stopped; the restart command is handed back at the end.
- `tev1:0.8b-q8_0` is used only in CI to exercise the real decision-model path in containers; it is not evaluated.

## Build-time findings that adjusted the plan

Measured during the build (phases P1 to P5), recorded here so the deviations are traceable. Inputs for the Ollama
probe were synthetic (`evals/results/ollama_probe.json`, native Metal, warm, n=5).

- **Sources** (`02-source-probe.md`): AP News, Investing.com, Benzinga and Business Wire refuse a plain client at
  their CDN (Cloudflare / Akamai), so all four backups were promoted. Final 10: Yahoo Finance, CNBC, Motley Fool,
  PR Newswire, GlobeNewswire, SEC EDGAR EX-99.1, CNN Business, Fortune, Guardian Business, BBC Business.
- **R7 sanity gate passed**: each endpoint has at least one candidate inside its budget, so no budget moved.
  - `/signal`: `qwen3.5:0.8b` answers in 190 ms at ~510 prompt tokens. The 9B decision models take 1.3-1.5 s
    at the same size, so they are reported against the budget rather than the budget relaxed for them.
  - `/summary`: `qwen3.5:2b` shows first text at 0.93 s and finishes at 2.2 s on ~1.6k-token articles;
    `qwen3.5:4b` misses the 1.5 s TTFT budget at that size (2.4 s).
  - `/analyze`: `ministral-3:3b` reaches the score at 1.47 s and finishes at 3.5 s on ~1.75k tokens; the 4B
    qwen models miss TTFT at full-article size.
- **Logprobs**: Ollama 0.40.6 also returns logprobs on streamed chunks (undocumented). The knob comparison stays
  on non-streaming `/signal`; `/analyze` keeps the ordinal knob. Logprob-weighted streaming confidence goes to
  the backlog.
- **Field order**: `qwen3:4b-instruct-2507` ignores schema property order (always entity, score, summary) and
  ignores `think:false` when a schema is set, so the field-order dimension runs on `qwen3.5:4b` and
  `ministral-3:3b` only.
- **Fetch cap** raised from 5 MB to 10 MB: CNN article pages are 5.7-6.6 MB decompressed.
- **Sampling penalties zeroed**: the qwen3.5 Modelfiles ship `presence_penalty 1.5`, which still applies under
  greedy decoding. Few-shot answers end in `neutral` / `0`, so the penalty pushed generated labels away from
  neutral (with the penalty on, the generated `qwen3.5:0.8b` label disagreed with its own first-token argmax on 84
  of the 300 dev300 tweets for v2 and 62 for v4, [grid notes](../../../evals/results/grid/notes.md)). Every chat call
  now pins `presence_penalty 0`, `frequency_penalty 0`, `repeat_penalty 1`; `num_predict` still bounds runaway
  output.
- **Length-cut generations are salvaged**: a summary or analyze output that hits `num_predict` returns its fields
  with the summary cut to the last full sentence and `truncated: true`, instead of a 502 after a doubled-budget
  retry.
- **Summary length is enforced in serving** (R6 follow-up): no candidate reached the 0.8 deterministic pass rate
  on the dev articles, mostly on the 60-word cap. Rather than relax the gate, the service caps every summary at
  3 sentences / 60 words at a sentence boundary, and the frozen prompts ask for 2-3 sentences of at most 50 words
  that name the primary entity first.
- **Key-fact coverage threshold 0.60** (R6): a hand spot-check of 45 key facts on 10 dev articles
  ([`spotcheck.dev.json`](../../../evals/results/summary/spotcheck.dev.json)) matched 33 of 45 flags at a cosine
  of 0.60, a near tie with 0.55 (34 of 45, one more false "covered"); 0.75 marked no false "covered" but missed 20
  stated facts. 0.60 is primary and
  0.75 is reported as a conservative bound; the Opus judge remains the selection signal.
- **Docker-cpu pass 1 was biased by Ollama's prompt cache**: the pass ran the bundled app over HTTP (n=10 warm per
  cell), and each JSON cell re-sent prompts its SSE twin had just evaluated on the same articles, so its prompt eval
  read 118-171 ms. `container_reload` in `evals/src/veritrade_evals/latency.py` now stops the model inside the
  bundled container (`docker exec ... ollama stop`) before every warm phase, mirroring the native suite's unload.
- **Docker-cpu pass 2 was discarded after contention in part of the pass**: other workloads ran during it
  (another container was starting), its `/signal` cell decoded at about 2.4 s per token against about 35 ms in the
  clean pass, and 4 `/analyze` JSON requests timed out. Its 4B cells show the same per-token rates as the clean
  pass, but the whole pass was discarded. The host load average is no evidence here: a clean docker-cpu pass runs at
  35-75 by itself. The samples of both passes are not committed.
- **Docker-cpu measured 7.5-14.0x slower than Metal**: one clean pass on 2026-10-19 (warm n=10 per cell, reload
  hook on, 0 errors, laptop on battery from 67% to 46%) put every docker-cpu p50 over its budget. Against the native
  p50 of the same cell: `/signal` 0.8b lede 2,291 ms (12.9x), `/summary` 4B 7.5-9.5x (SSE TTFT 10,169 ms, complete
  20,820 ms; JSON 16,357 ms), `/analyze` 4B 9.5-14.0x (SSE signal 10,428 ms, complete 30,145 ms; JSON 36,093 ms)
  ([05](05-latency.md#docker-cpu), [`latency.md`](../../../evals/results/latency/latency.md#docker-cpu)). This is
  why native Ollama is the macOS default (Q1) and the bundled profile is for Linux and CI.
  - The figures vary about 2x between runs: pass 1's streaming cells (discarded, not committed) completed in
    14,405 ms (`/summary`) and 12,106 ms (`/analyze`), 1.4x and 2.5x faster than the clean pass on the same
    protocol. Power source, background load and how the VM's 14 threads land on performance and efficiency cores
    all move CPU inference in Docker on a laptop.
- **`OLLAMA_CPUS` is a quota, not a thread count**: compose's `cpus:` caps CPU time but does not lower llama.cpp's
  thread count, so a quota below the visible core count can make CPU inference slower, not just capped. A `cpuset`
  or `num_thread` is the fix ([90-optional-backlog.md](90-optional-backlog.md)).

## Selection outcome

`uv run python -m veritrade_evals select` applied the Q2 / R6 / R7 rules to the committed results, and
`select --write` set `[defaults]` in [`models.toml`](../../../backend/src/veritrade/data/models.toml) and the pull
list in [`deploy/bundled.env`](../../../deploy/bundled.env). Rules and per-candidate verdicts are in
[`selection.md`](../../../evals/results/selection/selection.md); latency is native-metal, warm, excluding fetch
([`latency.md`](../../../evals/results/latency/latency.md)). Every endpoint had an accuracy-eligible candidate inside
its budget, so no Pareto knee was needed.

| Endpoint | Winner, frozen config | Rule that picked it | Accuracy (dev) | Latency p50 |
|---|---|---|---|---|
| `/signal` | `qwen3.5:0.8b-q8_0`, v2-label, label logprobs with `directional_tau` 0.28, lede | Best dev300 F0.5 among candidates with dev300 inversions <= 2%, an articles-dev inversion CI reaching 2%, and lede p50 <= 500 ms | dev300 F0.5 0.730 [0.651, 0.795]; articles 0.685 | 178 ms |
| `/summary` | `qwen3.5:4b-q4_K_M`, prompt v2 | Deterministic pass rate >= 0.8 and both stream budgets, then Opus judge faithfulness | Faithfulness 4.40 vs 3.97 for `qwen3.5:2b-q4_K_M`, paired gap 0.43 [0.07, 0.77] ([summary eval](../../../evals/results/summary/summary.md)) | TTFT 1,128 ms, complete 2,199 ms |
| `/analyze` | `qwen3.5:4b-q4_K_M`, v1, ordinal \|s\| >= 1, sentiment-first | As `/signal`, with TTFT p50 <= 1.5 s and complete p50 <= 4 s | dev300 F0.5 0.752 [0.664, 0.834]; articles 0.880 | Signal 1,098 ms, TTFT 1,292 ms, complete 2,584 ms |

- **Bundle (R8)**: the winners alone, `qwen3.5:0.8b-q8_0` and `qwen3.5:4b-q4_K_M`, are 4.34 GB of model files
  against the 10 GB cap, because one 4B file serves both `/summary` and `/analyze`; no fallback combination was
  needed.
- **The `/signal` trade-off**: `clef-flash:9b-q8_0` has the best dev300 F0.5 (0.874 [0.811, 0.937]) but misses the
  500 ms lede budget (p50 659 ms), and its 10.58 GB file alone exceeds the 10 GB bundle cap; `nimble:9b-q4_K_M`
  (0.843, 1,061 ms) misses too. The selected `qwen3.5:0.8b-q8_0` scores 0.730 at 178 ms. On the article dev set it
  catches 10 of 20 directional articles against 17 for clef-flash, with one inversion each
  ([articles dev](../../../evals/results/articles_dev/summary.md)). The budget was reported, never moved (R7). Both
  9B decision models stay on the allowlist and can be chosen per request with the `model` parameter on native
  Ollama.
- **The winner's articles-dev inversion**: that one inversion is 1 of 14 directional calls (7.1%). The rule passes
  it because at n=30 the 95% CI reaches down to 0%; the point estimate alone would fail the 2% cap. On the 60 test
  articles it is 1 of 30 (3.3%, CI 0% to 10.3%), again above the cap with a CI that reaches 0%, so the holdout does
  not clear the doubt (below).
- **Provisional defaults replaced**: the budget-feasible defaults set after the P2 probe (`/summary`
  `qwen3.5:2b-q4_K_M`, `/analyze` `ministral-3:3b`, see `defaults_untouched` in
  [`frozen.json`](../../../evals/results/frozen.json)) gave way to `qwen3.5:4b-q4_K_M` on both. The probe's
  synthetic inputs gave about 1.6k-token prompts, where the 4B model missed TTFT; the dev articles after the served
  cascade give an 825-token `/summary` prompt at p50 ([`latency.json`](../../../evals/results/latency/latency.json);
  about 718 tokens of `full` article text, [03](03-extraction.md)), and there it meets both budgets.
- **Split vs combined**: split mode delivers the signal at 178 ms; the combined `/analyze` stream delivers it at
  1,098 ms, about six times later, but scores higher on the article dev set (0.880 [0.791, 0.989] vs
  0.685 [0.473, 0.833], overlapping CIs at n=30) and on the test articles (0.871 vs 0.801), and is close on dev300
  (0.752 vs 0.730).

**Holdout check.** The frozen configs ran once, on 2026-10-19, on the test splits (`articles_test`, `tfns_val`,
`fpb_allagree`, `finentity_single`), with `qwen3.5:9b-q4_K_M` as a holdout-only quality ceiling, and six blind Opus
judges scored the winners' summaries on the 60 test articles. Mid-run the native Ollama server hung (21 timed-out
rows in one cell); it was restarted with the same service environment and the run resumed from its caches, which
[`holdout_runs.json`](../../../evals/results/holdout_runs.json) records, and every cell finished with 0 errors.
Nothing was re-tuned on these numbers and the defaults stay as frozen. Full tables:
[04-evals.md](04-evals.md#holdout-results), [`holdout.md`](../../../evals/results/holdout/holdout.md).

| Default | Test articles F0.5 [95% CI], inversions | Articles dev | `tfns_val` F0.5 [95% CI], inversions | dev300 | `fpb_allagree` / `finentity_single` |
|---|---|---|---|---|---|
| `/signal` `qwen3.5:0.8b-q8_0`, lede | 0.801 [0.668, 0.886], 1/30 (3.3%) | 0.685, 1/14 | 0.648 [0.612, 0.679], 2/386 (0.5%) | 0.730, 0.0% | 0.700 / 0.711 |
| `/analyze` `qwen3.5:4b-q4_K_M`, full | 0.871 [0.804, 0.940], 0/42 | 0.880, 0/22 | 0.662 [0.631, 0.690], 4/527 (0.8%) | 0.752, 0.0% | 0.940 / 0.760 |
| `qwen3.5:9b-q4_K_M`, untuned ceiling | 0.814 [0.759, 0.880], 1/51 (2.0%) | - | - | - | 0.932 / 0.692 |

| Summaries, test (dev) | Opus faithfulness [95% CI] | Coverage | Material errors [95% CI] |
|---|---|---|---|
| `/summary` `qwen3.5:4b-q4_K_M` | 4.37 [4.17, 4.57] (4.40) | 4.50 (4.57) | 17% [7%, 27%] (10%) |
| `/analyze` `qwen3.5:4b-q4_K_M` | 4.17 [3.97, 4.37] (4.47) | 4.63 (4.63) | 25% [15%, 35%] (10%) |

- **`/signal`**: F0.5 is higher than on dev (0.801 vs 0.685, inside the wide dev CI, so not evidence of a gain), but
  the inversion point estimate is above the 2% cap again and bearish recall is 0.40 on both sets. Selection let it through because the articles-dev CI at n=30 reached 0%; the test
  result does not clear that doubt. On tweets and the cross-domain sets it stays under the cap (0.0-1.0%).
- **`/analyze`** generalises best of the served defaults: 0.871 against 0.880, no inversions. Its FPB 0.940 is far
  above dev300 because AllAgree sentences are unambiguous by construction; that gap compares domains and is not an
  improvement. On FinEntity it scores 0.760 against the 0.8B's 0.711.
- **Tweets**: every model, finalists included, lands 0.03-0.09 F0.5 below dev300, the expected optimism of picking
  prompts and thresholds on 300 dev tweets; each endpoint's dev300 leader still leads. The 9B decision models keep
  their accuracy lead (test articles 0.859 lede for both, 0.896 full for `nimble`; tweets 0.783 / 0.755 against
  0.648) and still miss the latency budget, so the `/signal` trade-off above stands.
- **The 9B ceiling** beats the 4B nowhere (0.814 vs 0.871 on articles, 0.692 vs 0.760 on FinEntity, a tie on FPB):
  no evidence the 4B leaves quality on the table at this prompt. It is untuned, so this is a floor for the 9B.
- **Summaries**: `/summary` holds on unseen articles; `/analyze` summaries degrade (faithfulness 4.17, 15 of 60 with
  a material error), and the served length cap shortened 25% of them (33% on dev)
  ([summary eval](../../../evals/results/summary/summary.md)).
