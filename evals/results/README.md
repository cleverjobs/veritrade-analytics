# Evaluation results

Everything here is numbers, ids, model names and hand-written notes: no article, summary or dataset text (publisher and dataset rights). Inputs and caches stay in `evals/data/private/` (gitignored). Regenerate this index and `benchmarks.json` with `uv run python -m veritrade_evals report`.

Order of the pipeline: grid -> finalists -> freeze -> summary -> latency -> select -> holdout (once) -> report.

## Status at 2026-10-19T12:02:00+00:00

- Selection: /signal `qwen3.5:0.8b-q8_0`, /summary `qwen3.5:4b-q4_K_M`, /analyze `qwen3.5:4b-q4_K_M`.
- Benchmarks rows: signal 37, latency 16, summary 5, extraction 22.
- Holdout runs on record: 1 (`8dfd2f322299` complete).
- Complete.

## Files

| file | what it holds |
|---|---|
| `README.md` | this index (`report`) |
| `articles_dev/manifest.json` | articles dev finalist run manifest |
| `articles_dev/notes.md` | hand-written notes, merged into articles_dev/summary.md |
| `articles_dev/summary.json` | finalists on the 30 dev articles at their dev300 thresholds (`finalists`) |
| `articles_dev/summary.md` | readable articles dev finalist report |
| `benchmarks.json` | BenchmarksV1 payload the frontend Benchmarks page renders (`report`) |
| `extraction/matrix.json` | extraction matrix: parser x extractor quality, speed, install size |
| `extraction/summary.md` | readable extraction matrix |
| `frozen.json` | frozen prompt versions and thresholds per model, with the evidence for each (`freeze`) |
| `grid/analyze__ministral-3_3b.json` | per-model /analyze grid report on dev300: every config, threshold sweep, CIs |
| `grid/analyze__qwen3.5_4b-q4_K_M.json` | per-model /analyze grid report on dev300: every config, threshold sweep, CIs |
| `grid/analyze__qwen3_4b-instruct-2507-q4_K_M.json` | per-model /analyze grid report on dev300: every config, threshold sweep, CIs |
| `grid/error_patterns.json` | error counts per model and keyword bucket on dev300 (counts only) |
| `grid/manifest.json` | dev300 grid manifest: prompt versions, cache keys and model digests per cell |
| `grid/notes.md` | hand-written error-analysis notes, merged into grid/summary.md |
| `grid/signal__clef-flash_9b-q8_0.json` | per-model /signal grid report on dev300: every config, threshold sweep, CIs |
| `grid/signal__nimble_9b-q4_K_M.json` | per-model /signal grid report on dev300: every config, threshold sweep, CIs |
| `grid/signal__qwen3.5_0.8b-q8_0.json` | per-model /signal grid report on dev300: every config, threshold sweep, CIs |
| `grid/summary.md` | dev300 grid report: best config per model, knobs, error patterns, notes |
| `holdout/holdout.json` | one-time holdout metrics with CIs and dev-vs-holdout gaps (`holdout run`) |
| `holdout/holdout.md` | readable holdout report |
| `holdout/manifest.json` | holdout run manifest: cache keys and model digests per cell |
| `holdout_runs.json` | holdout-once guard keyed by the sha256 of frozen.json (`holdout run`) |
| `latency/latency.json` | latency suite aggregates per cell with R7 budget marks (`latency`) |
| `latency/latency.md` | readable latency report |
| `latency/preflight.json` | latency preflight: machine, Ollama version and service env, load, memory |
| `latency/raw/20261018T200825Z.jsonl` | raw native latency samples of one preflight session (ids, numbers) |
| `ollama_probe.json` | Ollama capability and latency probe on synthetic inputs (P2 sanity gate) |
| `selection/selection.json` | per-endpoint winners, every candidate with numbers and verdict, R8 bundle |
| `selection/selection.md` | readable selection report (`select`) |
| `summary/spotcheck.dev.json` | key-fact coverage threshold spot check: ids, cosines and hand labels |
| `summary/summary.json` | summary evaluation per endpoint and model: checks, key facts, judge (`summary`) |
| `summary/summary.md` | readable summary evaluation report |

Notes on `benchmarks.json`: dev300 rows carry the endpoint's default scope (tweets are headline-length); latency rows are warm p50/p95 excluding fetch; `articles.agreement` and `kappa` are the two labellers on the double-labelled pool before adjudication.
