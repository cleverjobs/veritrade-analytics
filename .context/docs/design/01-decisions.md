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
