# 04 - Evaluation

How the models, prompts and thresholds behind `/signal`, `/summary` and `/analyze` were chosen, and how well they
do. The reasons behind each choice are in [`01-decisions.md`](01-decisions.md); the fetch and extraction path every
eval runs through is in [`02-source-probe.md`](02-source-probe.md) and [`03-extraction.md`](03-extraction.md).
Every number links to the committed result file it comes from. Rigor is balanced per aspect, as for the rest of this
PoC (R2): the signal metric, the splits and the one-time holdout are strict; the prompt grid is deliberately lean;
the article labels and summary judgements come from Claude Opus subagents with no human check.

```
dev300 grid (prompt x policy x threshold per model) -> one error-analysis pass -> articles dev check (transfer only,
never re-tuned) -> freeze -> summary eval on the frozen prompts -> latency suite -> select -> holdout, run once
```

## 1. Objective and metrics

A wrong directional call costs a trade, a missed one costs little (Q2). So the positives are directional calls, and
neutral or an abstention is the safe default. Definitions follow
[`metrics.py`](../../../evals/src/veritrade_evals/metrics.py):

- **Directional call**: an answered (not abstained) bullish or bearish prediction. An abstention is served as neutral
  with `abstained: true`. **Coverage** is the share of rows answered, **signal rate** the share with a directional
  call, **accuracy** the share whose served label (abstention -> neutral) equals gold.
- **Precision / recall** per class c in {bullish, bearish}: correct c calls / c calls and correct c calls / gold c
  rows. A class with no calls scores 0, not NaN, so "always neutral" scores F0.5 0.
- **F0.5 macro** (the selection metric): the mean of the two per-class F-beta scores at beta = 0.5, which weights
  precision above recall. Neutral is not a positive class.
- **Inversion rate**: bullish<->bearish flips (a directional call on a gold-directional row with the opposite label)
  over all directional calls. The cap is **2%**.
- **Operating point**: thresholds are swept over 0 and every distinct confidence value (the only points where calls
  change); the chosen point is the **max F0.5 among points with inversion <= 2%**, ties to higher coverage, then a
  lower threshold. A config with no point under the cap is "cap missed" and cannot be selected.
- **Prevalence weights** (dev300 only): each row weighs train prior / sample prior of its class, so rates estimate
  the real tweet mix; confusion and error counts stay raw. Articles and holdout sets are unweighted.
- **CIs**: 95% percentile bootstrap (B = 1,000, seed 20261018), resampled within each gold class, on the headline
  metrics only (F0.5 macro, inversion, coverage, signal rate, accuracy). Gaps between two configs on the same rows use
  a paired bootstrap. The threshold is not re-chosen per resample, so a CI at a tuned operating point is optimistic.

## 2. Datasets and splits

From the split headers in [`evals/data/splits/`](../../../evals/data/splits/) and [`stats.json`](../../../evals/data/articles/stats.json):

| split | source | role | n | bullish / bearish / neutral | weighting |
|---|---|---|--:|---|---|
| `dev300` | Twitter Financial News Sentiment (FNS), train | tuning: prompt grid, thresholds | 300 | 100 / 100 / 100 | train prior 20.2% / 15.1% / 64.7% |
| articles dev | Opus-labelled articles (section 3) | transfer check, error analysis, summary eval | 30 | 10 / 10 / 10 | none |
| articles test | Opus-labelled articles | holdout | 60 | 20 / 20 / 20 | none |
| `tfns_val` | Twitter FNS, validation | holdout, in-domain tweets | 2,388 | 475 / 347 / 1,566 | none (natural prior) |
| `fpb_allagree` | Financial PhraseBank, Sentences_AllAgree | holdout, cross-domain sentences | 2,259 | 570 / 303 / 1,386 | none |
| `finentity_single` | FinEntity, rows with one distinct label | holdout, cross-domain paragraphs | 851 | 181 / 180 / 490 | none |

- **dev300** comes from Twitter FNS train after dropping 69 rows whose normalised text is also in validation, 70
  duplicates and 4 rows with conflicting labels; then 100 per class with seed 20261018. Class weights are 0.605
  (bullish), 0.453 (bearish) and 1.942 (neutral). FPB and FinEntity positive / negative / neutral map to bullish /
  bearish / neutral; FPB collapses exact duplicates and drops sentences seen with two labels; FinEntity keeps rows
  with at least one entity annotation and exactly one distinct label.
- **Committed**: per split, the ids, the sha1 of the normalised text (NFKC, casefold, URLs and whitespace runs
  dropped), the labels, the selection rule and the dataset revision;
  [`datasets.lock.json`](../../../evals/data/datasets.lock.json) pins each dataset commit, file sha256 and licence. A
  split fails to load on a revision mismatch, a missing id, text-hash drift or label drift.
- **Never committed**: dataset text (FPB is CC BY-NC-SA 3.0), downloaded at eval time into the gitignored
  `evals/data/private/`, along with article HTML, gold bodies, headlines, predictions, summaries and judge packets.

## 3. The Opus-labelled article set

Public datasets hold sentences and headlines; the service reads whole articles (Q5, R5). The set has 90 real articles
from the 10 sources in [`02-source-probe.md`](02-source-probe.md), 3 classes x 3 per source, published 2026-08-19 to
2026-10-18. One seeded article per (source, class) cell goes to dev: **30 dev / 60 test**. Two quota adjustments keep
the 30/30/30 balance (Yahoo 4 bearish, BBC 2 bearish). Every Opus step ran as a Claude Code subagent pinned to Claude Opus
5.5, so no API key or paid service was used. Prompts and runbook: [`evals/opus_prompts/`](../../../evals/opus_prompts/00-README.md).

1. **Discover** (one agent per source): candidates from RSS and section pages, fetched by the deterministic
   `articles` CLI through the serving fetcher; agents never fetch anything themselves.
2. **Gold**: Opus picks the body as line ranges of the page, writes 3-5 key facts in its own words, and gives a
   provisional label.
3. **Label A / label B**: two independent agents apply the service's rubric (economic impact on the primary entity,
   not tone; neutral when in doubt), never seeing each other's output or the provisional label.
4. **Adjudicate**: one agent reviews every article where the labels differ or either labeller flagged ambiguity, and
   either keeps it (its label is final) or replaces it as genuinely ambiguous.
5. **Refill and finalize**: short cells repeat steps 1-4; `finalize` picks per cell, agreed articles first; `verify`
   re-checks the set.

| pool statistic | value |
|---|--:|
| articles with a usable gold body | 298 |
| labelled by both labellers | 157 |
| labels agree | 151 (raw agreement 96.2%) |
| Cohen's kappa (A vs B) | 0.943 |
| sent to adjudication (6 disagreements, 43 agreed but flagged ambiguous) | 49 (31.2%) |
| replaced by the adjudicator | 24 of 49 (49.0%) |
| adjudicated articles kept in the final 90 | 5 |

In the final 90, A and B agree on every label (kappa 1.0), by construction since agreed articles are picked first, so
the pool agreement is the meaningful figure. The committed [`articles.jsonl`](../../../evals/data/articles/articles.jsonl)
holds ids, URLs, sources, dates, both labels, the adjudication, entity and ticker, and rationales and key facts in
Opus's words; the CLI rejects any that share 10 consecutive words with the article, so no article text is committed.
**Caveats**: Opus-only labels with no human check; replacing ambiguous articles makes the set cleaner than real
traffic; with 10 articles per class on dev, a gap needs about 0.2 F0.5 to clear zero.

## 4. Models, prompts and confidence policies

**Roster** (R2 and its follow-ups). The split architecture serves `/signal` + `/summary`; the combined one `/analyze`.

| endpoint | models | kind |
|---|---|---|
| `/signal` | `clef-flash:9b-q8_0`, `nimble:9b-q4_K_M` | decision models (native class probabilities) |
| `/signal` | `qwen3.5:0.8b-q8_0` | generative baseline (label and ordinal schemas) |
| `/summary` | `qwen3.5:2b-q4_K_M`, `qwen3.5:4b-q4_K_M` | generative |
| `/analyze` | `qwen3.5:4b-q4_K_M`, `qwen3:4b-instruct-2507-q4_K_M`, `ministral-3:3b` | generative, one call -> score + entity + summary |
| `/analyze` (holdout only) | `qwen3.5:9b-q4_K_M` | quality ceiling, untuned (prompt v1, \|s\| >= 1) |

**Lean prompt grid** on dev300: 33 cells ([`grid/manifest.json`](../../../evals/results/grid/manifest.json)), all on
native Metal, Ollama 0.40.6, greedy (temperature 0, seed 0, sampling penalties neutralised; see the build-time
findings in `01-decisions.md`).

- Generative `/signal` and `/analyze`: rubric A (definitions) or B (impact questions) x 0 or 4 synthetic shots =
  v1-v4; for `qwen3.5:0.8b` each runs on both the label and the ordinal schema.
- Decision models: rubric or plain one-line question x described or bare criteria = v1-v4.
- `/analyze` follow-ups, paired against v1: v5 (the tightened summary rule) and `v1-summary-first` (the field-order
  check, on `qwen3.5:4b` and `ministral` only).

**Confidence policies** ([`confidence.py`](../../../backend/src/veritrade/inference/confidence.py)):

- **probability** (decision models): argmax of the native class probabilities; abstain when it is below `min_p`.
- **label-logprob** (generative `/signal`): first label-token probabilities, renormalised over the tokens the grammar
  accepts, with the same `min_p` rule. With **`directional_tau`**, a neutral argmax still yields the likelier
  directional class when that class holds p >= tau; below tau the row is a served neutral, not an abstention. On a
  first-token label collision the adapter re-asks with the ordinal schema.
- **ordinal** (`/analyze`, streams, fallback): a score from -2 to 2, directional only when \|score\| >= t. It is read
  mid-stream as soon as it closes and needs no logprobs (Q3); logprob-weighted streaming confidence is on the backlog.

**Scope and field order.** `/signal` reads `lede` by default (headline plus opening paragraphs, body capped at 200
tokens); `lede` and `full` are both scored on articles. `/analyze` reads `full` and emits the sentiment field first,
so the score streams after a handful of tokens (R3).

**Error analysis and freeze.** One hand-written read of the grid counts
([`grid/notes.md`](../../../evals/results/grid/notes.md)) drove the changes listed in section 5. The `freeze`
command then wrote [`frozen.json`](../../../evals/results/frozen.json) and
[`models.toml`](../../../backend/src/veritrade/data/models.toml):

| endpoint | model | frozen prompt | policy | threshold | field order |
|---|---|---|---|---|---|
| `/signal` | `qwen3.5:0.8b-q8_0` | `v2@ff82e957` (rubric A, 4 shots, label schema) | label-logprob + `directional_tau` | min_p 0, tau 0.28 (ordinal \|s\| >= 1 fallback) | - |
| `/signal` | `clef-flash:9b-q8_0` | `v4@fcc93f1b` (plain question, bare labels) | probability | min_p 0.80 | - |
| `/signal` | `nimble:9b-q4_K_M` | `v1@a1e34f63` (rubric, described criteria) | probability | min_p 0.97 | - |
| `/summary` | `qwen3.5:2b-q4_K_M`, `qwen3.5:4b-q4_K_M` | `v2@84ce3858` (tightened summary rule) | - | - | - |
| `/analyze` | `qwen3.5:4b-q4_K_M` | `v1@7daf4f28` | ordinal | \|s\| >= 1 | sentiment-first |
| `/analyze` | `qwen3:4b-instruct-2507-q4_K_M` | `v5@e2b57fc6` | ordinal | \|s\| >= 1 | sentiment-first |
| `/analyze` | `ministral-3:3b` | `v5@e2b57fc6` | ordinal | \|s\| >= 1 | sentiment-first |

Freeze rules: `/signal` keeps the dev300 best rule unless it clearly fails on the articles. `/analyze` takes v5
unless the whole paired 95% CI of its gap against v1 is below zero on dev300 or the articles, which rejected v5 for
`qwen3.5:4b` (articles -0.080 [-0.178, -0.014]). Summary-first needed a dev300 win without an articles loss; it never
got one.

## 5. Accuracy on dev

Frozen configs only. Articles use the dev300 threshold unchanged and the serving extraction (`selectolax` +
`cascade`). Sources: [`grid/summary.md`](../../../evals/results/grid/summary.md),
[`articles_dev/summary.md`](../../../evals/results/articles_dev/summary.md).

**`/signal`** (dev300 weighted; articles lede is the served default):

| model | dev300 F0.5 [95% CI] | P bull / P bear | inversion | coverage | signal rate | articles lede F0.5 [95% CI] | inv. | articles full F0.5 [95% CI] | inv. |
|---|---|---|--:|--:|--:|---|--:|---|--:|
| `qwen3.5:0.8b-q8_0` | 0.730 [0.651, 0.795] | 0.914 / 0.918 | 0.0% | 100.0% | 15.4% | 0.685 [0.473, 0.833] | 1/14 | 0.635 [0.329, 0.817] | 0/10 |
| `clef-flash:9b-q8_0` | 0.874 [0.811, 0.937] | 0.867 / 0.905 | 0.0% | 72.9% | 33.2% | 0.848 [0.708, 0.963] | 1/20 | 0.907 [0.820, 0.989] | 0/20 |
| `nimble:9b-q4_K_M` | 0.843 [0.773, 0.901] | 0.948 / 0.891 | 0.0% | 26.3% | 24.2% | 0.895 [0.771, 0.976] | 0/17 | 0.880 [0.761, 0.976] | 0/19 |

**`/analyze`** (full scope, sentiment-first, \|s\| >= 1, coverage 100% for all):

| model | config | dev300 F0.5 [95% CI] | P bull / P bear | inversion | signal rate | articles F0.5 [95% CI] | inv. |
|---|---|---|---|--:|--:|---|--:|
| `qwen3.5:4b-q4_K_M` | v1 | 0.752 [0.664, 0.834] | 0.877 / 0.821 | 0.0% | 21.4% | 0.880 [0.791, 0.989] | 0/22 |
| `ministral-3:3b` | v5 | 0.685 [0.589, 0.772] | 0.837 / 0.768 | 1.0% | 20.0% | 0.741 [0.611, 0.858] | 0/19 |
| `qwen3:4b-instruct-2507-q4_K_M` | v5 | 0.647 [0.553, 0.741] | 0.702 / 0.752 | 0.0% | 23.1% | 0.779 [0.646, 0.879] | 1/22 |

No served rule clearly fails on the articles (whole inversion CI above 2%, or whole coverage CI below half the dev300
coverage), so every dev300 threshold stands. What the error analysis found
([`grid/notes.md`](../../../evals/results/grid/notes.md),
[`error_patterns.json`](../../../evals/results/grid/error_patterns.json)):

- **Inversions are nearly absent**: across the six best configs one tweet is inverted, by `ministral`. Wrong calls
  are mostly gold-neutral tweets called directional (2-8 per model). The real loss is recall: 34 (`clef-flash`) to
  118 (`qwen3.5:0.8b`) directional tweets missed; after the catch-all bucket, the largest miss bucket for every
  generative model is price-move tweets (33-38 of 57 directional ones missed, against 10 for `clef-flash`), which
  the impact rubric reads as immaterial.
- **`directional_tau` is the largest single lever**: `qwen3.5:0.8b` parks most of its mass on neutral, and the tau
  rule lifts its F0.5 by +0.218 [+0.141, +0.307] (paired). It loses for the peaked decision models (-0.046 and
  -0.303), so they keep the plain rule.
- **Other changes**: rubric A beats B for every generative model; shots help `qwen3.5:0.8b` on the label schema but
  hurt every `/analyze` model, so all frozen `/analyze` prompts are zero-shot; summary-first never wins and delays
  the score (2-3x on articles). The price-move rubric sentence was not added: articles do not show the tweet miss
  (0-1 of 5 directional price-move headlines missed).
- **Knob comparison**: without tau, the logprob knob beats the ordinal one on `qwen3.5:0.8b` by +0.059 [-0.068,
  +0.187], not significant; with tau it wins clearly. The ordinal schema cannot use tau, because this tokenizer fuses
  `" -"` and hides the sign of negative scores.

## 6. Summary evaluation

Decision R6, run by `veritrade_evals.summary_eval` on the stored HTML of the 30 dev articles through the serving
adapters (full scope, frozen prompts). Serving caps every summary at **3 sentences / 60 words** at a sentence
boundary, and the checks score the capped text.

- **Deterministic checks** (pass-rate gate >= 0.8): length, sentence count, every number grounded in the gold article
  (format-tolerant; a derived figure counts as unsupported), gold entity named, not truncated.
- **Key-fact coverage**: each Opus key fact and each summary sentence is embedded with `qwen3-embedding:0.6b`; a fact
  is covered at a best cosine **>= 0.60** (primary), with 0.75 as the strict bound. In the hand-labelled spot check
  of 45 facts on 10 summaries ([`spotcheck.dev.json`](../../../evals/results/summary/spotcheck.dev.json)), 0.60
  matched 33 of 45 flags with 5 false "covered" and 0.55 matched 34 with 6, a near tie; 0.75 had no false "covered"
  but missed 20 stated facts (see the build-time findings in `01-decisions.md`).
- **Opus judges**, blind: per article the candidate summaries are anonymised as `S1..Sn` and shuffled
  ([`50-judge.md`](../../../evals/opus_prompts/50-judge.md)); each gets faithfulness 1-5, coverage 1-5, a list of
  hallucinations, and whether it leads with the market-moving fact. **Material error rate** = share at
  faithfulness <= 3.

Dev articles, n = 30 per row ([`summary/summary.md`](../../../evals/results/summary/summary.md)); length and sentence
count pass 100% by construction; CIs resample articles within their sentiment label.

| endpoint | model | pass | numbers | entity | key facts 0.60 / 0.75 | faithful [95% CI] | cover | halluc. flag | material [95% CI] | leads |
|---|---|--:|--:|--:|---|---|--:|--:|---|--:|
| summary | `qwen3.5:2b-q4_K_M` | 90% | 97% | 93% | 52% / 24% | 3.97 [3.67, 4.23] | 4.30 | 67% | 30% [17%, 47%] | 90% |
| summary | `qwen3.5:4b-q4_K_M` | 90% | 93% | 97% | 47% / 23% | 4.40 [4.17, 4.60] | 4.57 | 50% | 10% [0%, 20%] | 100% |
| analyze | `qwen3.5:4b-q4_K_M` | 93% | 97% | 97% | 57% / 29% | 4.47 [4.20, 4.70] | 4.63 | 40% | 10% [0%, 20%] | 100% |
| analyze | `qwen3:4b-instruct-2507-q4_K_M` | 90% | 90% | 100% | 56% / 25% | 4.73 [4.53, 4.93] | 4.43 | 20% | 7% [0%, 17%] | 83% |
| analyze | `ministral-3:3b` | 77% | 80% | 97% | 45% / 19% | 3.77 [3.37, 4.20] | 4.27 | 63% | 40% [23%, 57%] | 80% |

**Hallucination flag vs material error.** The flag rate counts any summary with at least one entry in the judge's
hallucination list, including minor slips that still allow faithfulness 4 (a loosely worded or small unsupported
detail "a trader would shrug off"). The material error rate counts only faithfulness <= 3: a wrong key number, an
unsupported claim that matters, a misattribution, or worse. So `qwen3.5:4b` on `/summary` is flagged on half the
articles but has a material error on 10% [0%, 20%]; the material rate is the one to read as risk. The deterministic
number check agrees in direction: 1-3 unsupported numbers across 30 summaries for the qwen models, 8 for `ministral`.

## 7. Selection

Rules: [`selection/selection.md`](../../../evals/results/selection/selection.md). Accuracy filters come first (dev300
inversion <= 2%, articles dev not clearly over the cap, summary pass rate >= 0.8), then the native warm latency
budgets of R7 (measured in `05-latency.md`). Survivors rank on dev300 F0.5 then articles dev F0.5 (`/signal`,
`/analyze`) or on judge faithfulness then coverage (`/summary`). Budgets are reported, never moved; when nothing meets
one, the Pareto knee is used with the reason stated.

| endpoint | winner | deciding numbers | others |
|---|---|---|---|
| `/signal` | `qwen3.5:0.8b-q8_0` (v2-label, tau 0.28) | only candidate inside the 500 ms lede budget: p50 178 ms, dev300 F0.5 0.730 | `clef-flash` 0.874 at 659 ms; `nimble` 0.843 at 1,061 ms |
| `/summary` | `qwen3.5:4b-q4_K_M` (v2) | faithfulness 4.40 vs 3.97, paired +0.43 [0.07, 0.77]; TTFT p50 1,128 ms, complete 2,199 ms | `qwen3.5:2b` faster (484 / 1,272 ms), more material errors (30%) |
| `/analyze` | `qwen3.5:4b-q4_K_M` (v1) | dev300 F0.5 0.752; TTFT p50 1,292 ms (limit 1,500), complete 2,584 ms | `ministral` 0.685; `qwen3-instruct` 0.647, though its summary faithfulness is best (4.73 vs 4.47, paired gap 0.27 [-0.07, 0.60], not significant) |

Bundled models (R8): `qwen3.5:0.8b-q8_0` + `qwen3.5:4b-q4_K_M`, 4.34 GB against the 10 GB cap.

**The `/signal` trade-off.** `clef-flash:9b-q8_0` has the best dev300 F0.5 (0.874) but misses the 500 ms lede budget
(p50 659 ms), and its 10.58 GB file alone exceeds the 10 GB bundle cap. `nimble:9b-q4_K_M` (0.843, 1,061 ms) misses
the budget too. The selected `qwen3.5:0.8b-q8_0` scores 0.730 at 178 ms: high dev300 precision (0.91 / 0.92), low
recall (it misses about half the directional articles on dev). The budget was not moved to fit the 9B models; both
stay on the allowlist and can be chosen per request with the `model` parameter on native Ollama.

## 8. Holdout

Protocol of `uv run python -m veritrade_evals holdout {plan,run,analyze}` ([`holdout.py`](../../../evals/src/veritrade_evals/holdout.py)):

- Frozen rules at their frozen thresholds and prompts, no re-tuning, natural class priors (unweighted); a prompt that
  drifted since the freeze fails the run. `gap` = holdout F0.5 minus the dev F0.5 of the same rule (articles dev at
  the same scope for articles; dev300 for the tweet and cross-domain sets).
- 22 cells: the three `/signal` finalists on the test articles (lede and full) and `tfns_val`; the three `/analyze`
  finalists on the test articles (full) and `tfns_val`; the two winners also on `fpb_allagree` and
  `finentity_single`; the `qwen3.5:9b` ceiling on the test articles and both cross-domain sets. Then summaries from
  the `/summary` and `/analyze` winners on the 60 test articles go to the blind Opus judges.
- **Run once**: `evals/results/holdout_runs.json` is a guard keyed by the sha256 of
  `frozen.json`, recording the selection sha256, git sha, and start and completion times. An interrupted run resumes
  from its caches; a completed config refuses a second run unless `--force --reason` is passed, which is logged in
  the guard and must be disclosed. It also refuses a provisional selection (one made before the latency suite)
  without `--allow-provisional`; this selection was not provisional.
- `holdout analyze` only re-scores the cached predictions and never calls a model; the tables below come from the
  `evals/results/holdout/holdout.md` it writes.

### Holdout results

The one-time run is in progress; its tables are added here from `evals/results/holdout/holdout.md` when it completes.
<!-- PENDING:holdout replace the in-progress line above with a two-to-four-sentence verdict: do the frozen rules hold on test (F0.5 gap vs dev, inversion cap, coverage), and any clear failure -->

#### `/signal` on the test articles (n = 60, lede and full)
<!-- PENDING:holdout table: qwen3.5:0.8b winner + clef-flash + nimble finalists, lede and full; F0.5 [95% CI], bull P/R, bear P/R, inversion, coverage, dev F0.5, gap -->

#### `/signal` on Twitter FNS validation (n = 2,388)
<!-- PENDING:holdout table: three /signal finalists on tfns_val; F0.5 [95% CI], P/R per class, inversion, coverage, dev300 F0.5, gap -->

#### `/signal` winner, cross-domain (FPB AllAgree n = 2,259, FinEntity single-label n = 851)
<!-- PENDING:holdout table: qwen3.5:0.8b-q8_0 on fpb_allagree and finentity_single; F0.5 [95% CI], P/R per class, inversion, coverage, dev300 F0.5, gap -->

#### `/analyze` on the test articles (n = 60, full)
<!-- PENDING:holdout table: qwen3.5:4b winner + ministral-3:3b + qwen3:4b-instruct-2507 finalists on articles_test full; F0.5 [95% CI], P/R per class, inversion, coverage, dev F0.5, gap -->

#### `/analyze` on Twitter FNS validation (n = 2,388)
<!-- PENDING:holdout table: three /analyze finalists on tfns_val; F0.5 [95% CI], P/R per class, inversion, coverage, dev300 F0.5, gap -->

#### `/analyze` winner, cross-domain (FPB AllAgree n = 2,259, FinEntity single-label n = 851)
<!-- PENDING:holdout table: qwen3.5:4b-q4_K_M /analyze on fpb_allagree and finentity_single; F0.5 [95% CI], P/R per class, inversion, coverage, dev300 F0.5, gap -->

#### Quality ceiling: `qwen3.5:9b` vs the `/analyze` winner
<!-- PENDING:holdout table: qwen3.5:9b-q4_K_M (untuned ceiling) next to qwen3.5:4b-q4_K_M on articles_test full, fpb_allagree, finentity_single; F0.5 [95% CI], inversion, coverage, and one sentence on the headroom -->

#### Summaries on the test articles (n = 60)
<!-- PENDING:holdout table: summary/qwen3.5:4b and analyze/qwen3.5:4b on test articles; pass, numbers, entity, key facts 0.60 / 0.75, Opus judge faithfulness [CI], coverage, halluc. flag, material error rate [CI], leads -->

## 9. Threats to validity

- **One labeller family**: article labels, gold bodies, key facts and summary judges are all Claude Opus. Labellers
  were independent and judges blind, but a shared bias would not show, and no human checked any label.
- **Small n**: with 30 dev articles, the served rules' CIs span about 0.2-0.5 F0.5 (60 test articles in the
  holdout); article results check that rules transfer,
  they do not rank models.
- **Tuned on tweets**: thresholds and prompts were tuned on 300 tweets and only checked on articles; CIs at a tuned
  threshold exclude the threshold choice, and `directional_tau` is one more tuned parameter.
- **Class mix**: the article sets are balanced while real traffic is likely neutral-heavy (64.7% in Twitter FNS
  train), so per-call precision in production is likely below the article figure. dev300 reweights; articles do not.
- **Cleaner than reality**: 10 fetchable sources, a 60-day window, 24 ambiguous articles replaced; for three sources
  the extraction profiles were tuned with this set in view (caveats in `03-extraction.md`).
- **Contamination**: the public datasets may be in the models' pretraining data; the articles are recent, which
  limits but does not rule out overlap.
- **Precision first, by design**: the selected `/signal` model misses about half the directional articles; a miss is
  cheap under Q2, but users will see many neutrals.
- **Indicative parts**: grid and articles timings ran on a shared machine (selection uses the dedicated native-metal
  suite; Docker CPU is in `05-latency.md`), and the error-analysis keyword buckets are a first-match heuristic.
