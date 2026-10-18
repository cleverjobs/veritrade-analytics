"""BenchmarksV1 for the frontend: `uv run python -m veritrade_evals report [--strict | --check]`, plus
evals/results/README.md. Built from committed results and validated against a mirror of the frontend
type; a missing input leaves its section empty and is listed, never invented."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veritrade.models_registry import allowlist
from veritrade_evals.holdout import GUARD, HOLDOUT_JSON, git_sha
from veritrade_evals.latency import LATENCY_JSON, PREFLIGHT_JSON
from veritrade_evals.paths import ARTICLES_DIR, REPO_ROOT, RESULTS_DIR
from veritrade_evals.predcache import model_slug
from veritrade_evals.select import (
    ARTICLES_JSON,
    FROZEN_JSON,
    GRID_DIR,
    SELECTION_JSON,
    SUMMARY_JSON,
    article_row,
    dev300_f05,
    frontend_selection,
    grid_point,
)

BENCHMARKS_JSON = RESULTS_DIR / "benchmarks.json"
README = RESULTS_DIR / "README.md"
STATS_JSON = ARTICLES_DIR / "stats.json"
EXTRACTION_JSON = RESULTS_DIR / "extraction" / "matrix.json"
TS_TYPES = REPO_ROOT / "frontend" / "src" / "lib" / "benchmarks.ts"
ENDPOINTS = ("signal", "summary", "analyze")

# Python mirror of BenchmarksV1; test_report checks it against the interfaces in TS_TYPES.
# A kind is a JSON type name, "=value" for a literal, or the name of another interface.
INTERFACES: dict[str, dict[str, tuple[str, ...]]] = {
    "BenchmarkEnvironment": {
        "chip": ("string",),
        "ramGb": ("number",),
        "ollamaVersion": ("string",),
        "backend": ("string",),
        "gitSha": ("string", "null"),
    },
    "SignalBenchmark": {
        "endpoint": ("=signal", "=analyze"),
        "model": ("string",),
        "kind": ("=decision", "=generative"),
        "policy": ("string",),
        "scope": ("=lede", "=full"),
        "fieldOrder": ("string", "null"),
        "promptVersion": ("string",),
        "dataset": ("string",),
        "split": ("string",),
        "n": ("number",),
        "f05Macro": ("number",),
        "precisionBullish": ("number",),
        "precisionBearish": ("number",),
        "recallBullish": ("number",),
        "recallBearish": ("number",),
        "inversionRate": ("number",),
        "coverage": ("number",),
        "ciLow": ("number", "null"),
        "ciHigh": ("number", "null"),
    },
    "LatencyBenchmark": {
        "endpoint": ("string",),
        "model": ("string",),
        "scope": ("=lede", "=full"),
        "streaming": ("boolean",),
        "backend": ("string",),
        "n": ("number",),
        "p50Ms": ("number",),
        "p95Ms": ("number",),
        "ttftP50Ms": ("number", "null"),
        "timeToSignalP50Ms": ("number", "null"),
    },
    "SummaryBenchmark": {
        "endpoint": ("=summary", "=analyze"),
        "model": ("string",),
        "n": ("number",),
        "deterministicPassRate": ("number",),
        "keyFactCoverage": ("number",),
        "judgeFaithfulness": ("number", "null"),
        "judgeCoverage": ("number", "null"),
        "hallucinationRate": ("number", "null"),
    },
    "ExtractionBenchmark": {
        "parser": ("string", "null"),
        "extractor": ("string",),
        "n": ("number",),
        "p50Ms": ("number",),
        "p95Ms": ("number",),
        "precision": ("number",),
        "recall": ("number",),
        "f1": ("number",),
        "leakage": ("number",),
        "tokensToLlm": ("number",),
        "failureRate": ("number",),
        "installMb": ("number", "null"),
    },
    "Selection": {"model": ("string",), "reason": ("string",)},
    "ArticleSetStats": {
        "n": ("number",),
        "perSource": ("Record<string, { bullish: number; bearish: number; neutral: number }>",),
        "agreement": ("number", "null"),
        "kappa": ("number", "null"),
    },
    "BenchmarksV1": {
        "version": ("=1",),
        "sample": ("boolean",),
        "generatedAt": ("string", "null"),
        "environment": ("BenchmarkEnvironment",),
        "signal": ("SignalBenchmark[]",),
        "latency": ("LatencyBenchmark[]",),
        "summary": ("SummaryBenchmark[]",),
        "extraction": ("ExtractionBenchmark[]",),
        "selection": ("Record<SelectionEndpoint, Selection>",),
        "articles": ("ArticleSetStats",),
    },
}
PER_SOURCE_KEYS = ("bullish", "bearish", "neutral")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- validation ------------------------------------------------------------------------------


def _kind_ok(value: Any, kind: str) -> bool:
    if kind.startswith("="):
        literal = kind[1:]
        return value == (int(literal) if literal.isdigit() else literal) and not isinstance(value, bool)
    return {
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(kind, False)


def _check_object(value: Any, name: str, where: str) -> list[str]:
    spec = INTERFACES[name]
    if not isinstance(value, dict):
        return [f"{where}: expected a {name} object"]
    problems = [f"{where}: missing key {k!r}" for k in spec if k not in value]
    problems += [f"{where}: unexpected key {k!r}" for k in value if k not in spec]
    for key, kinds in spec.items():
        if key not in value:
            continue
        field = f"{where}.{key}"
        item = value[key]
        nested = [k for k in kinds if k in INTERFACES]
        if nested:
            problems += _check_object(item, nested[0], field)
        elif kinds[0].endswith("[]"):
            if not isinstance(item, list):
                problems.append(f"{field}: expected a list")
            else:
                for i, row in enumerate(item):
                    problems += _check_object(row, kinds[0][:-2], f"{field}[{i}]")
        elif kinds[0].startswith("Record<SelectionEndpoint"):
            problems += _check_selection(item, field)
        elif kinds[0].startswith("Record<string"):
            problems += _check_per_source(item, field)
        elif not any(_kind_ok(item, k) for k in kinds):
            problems.append(f"{field}: {item!r} is not {' | '.join(kinds)}")
    return problems


def _check_selection(value: Any, where: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != set(ENDPOINTS):
        return [f"{where}: expected exactly {', '.join(ENDPOINTS)}"]
    return [p for e in ENDPOINTS for p in _check_object(value[e], "Selection", f"{where}.{e}")]


def _check_per_source(value: Any, where: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{where}: expected an object"]
    problems = []
    for source, row in value.items():
        if not isinstance(row, dict) or set(row) != set(PER_SOURCE_KEYS):
            problems.append(f"{where}.{source}: expected {', '.join(PER_SOURCE_KEYS)}")
        elif not all(_kind_ok(row[k], "number") for k in PER_SOURCE_KEYS):
            problems.append(f"{where}.{source}: counts must be numbers")
    return problems


def validate(payload: Any) -> list[str]:
    """Problems that would break the BenchmarksV1 contract; empty means the page can render it."""
    return _check_object(payload, "BenchmarksV1", "benchmarks")


_INTERFACE = re.compile(r"export interface (\w+) \{\n(.*?)\n\}", re.DOTALL)
_FIELD = re.compile(r"^\s+(\w+)\??: (.+);$")


def ts_interfaces(text: str) -> dict[str, dict[str, str]]:
    """Field name -> TS type text for every exported interface (the drift check's view of the frontend)."""
    out = {}
    for name, body in _INTERFACE.findall(text):
        fields = {}
        for line in body.splitlines():
            if m := _FIELD.match(line):
                fields[m.group(1)] = m.group(2).strip()
        out[name] = fields
    return out


def ts_kinds(type_text: str) -> tuple[str, ...]:
    """Mirror kinds of a TS field type; unions of literals and primitives only, others pass through."""
    parts = [p.strip() for p in type_text.split("|")]
    if len(parts) > 1 or parts[0] in ("string", "number", "boolean", "null") or parts[0].startswith('"'):
        return tuple(f"={p.strip(chr(34))}" if p.startswith('"') else p for p in parts)
    if parts[0].isdigit():
        return (f"={parts[0]}",)
    return (type_text,)


# --- sections --------------------------------------------------------------------------------


def _short_prompt(version: str | None) -> str:
    return (version or "-").rsplit("/", 1)[-1]


def _signal_row(
    *,
    endpoint: str,
    model: str,
    policy: str,
    scope: str,
    field_order: str | None,
    prompt_version: str | None,
    dataset: str,
    split: str,
    n: int,
    metrics: Mapping[str, Any],
    ci: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pc = metrics["per_class"]
    spec = allowlist().get(model)
    return {
        "endpoint": endpoint,
        "model": model,
        "kind": spec.kind if spec else "generative",
        "policy": policy,
        "scope": scope,
        "fieldOrder": field_order,
        "promptVersion": _short_prompt(prompt_version),
        "dataset": dataset,
        "split": split,
        "n": n,
        "f05Macro": metrics["f05_macro"],
        "precisionBullish": pc["bullish"]["precision"],
        "precisionBearish": pc["bearish"]["precision"],
        "recallBullish": pc["bullish"]["recall"],
        "recallBearish": pc["bearish"]["recall"],
        "inversionRate": metrics["inversion_rate"],
        "coverage": metrics["coverage"],
        "ciLow": ci.get("low") if ci else None,
        "ciHigh": ci.get("high") if ci else None,
    }


def signal_rows(missing: list[str]) -> list[dict[str, Any]]:
    """Frozen rules on dev300 and the articles dev set, then the holdout rows when the holdout has run."""
    return dev_rows(missing) + holdout_rows(missing)


def dev_rows(missing: list[str]) -> list[dict[str, Any]]:
    frozen = read_json(FROZEN_JSON)
    if frozen is None:
        missing.append(_rel(FROZEN_JSON))
        return []
    articles = read_json(ARTICLES_JSON)
    if articles is None:
        missing.append(_rel(ARTICLES_JSON))
    rows: list[dict[str, Any]] = []
    for model, endpoints in frozen["models"].items():
        for endpoint in ("signal", "analyze"):
            entry = endpoints.get(endpoint)
            if entry is None:
                continue
            ev = entry["evidence"]["dev300"]
            policy = ev["policy"] if endpoint == "signal" else "ordinal"
            config = ev["config"]
            field_order = entry.get("field_order") if endpoint == "analyze" else None
            report = read_json(GRID_DIR / f"{endpoint}__{model_slug(model)}.json")
            ci = ev["f05_macro"] if endpoint == "signal" else dev300_f05(report, config, policy)
            point = grid_point(report, config, policy)
            common = {"endpoint": endpoint, "model": model, "policy": policy, "field_order": field_order}
            if point is None:
                missing.append(f"dev300 grid point for {endpoint}/{model}/{config}")
            else:
                rows.append(
                    _signal_row(
                        **common,
                        scope="lede" if endpoint == "signal" else "full",
                        prompt_version=entry.get("prompt_version"),
                        dataset="twitter-fns",
                        split="dev300",
                        n=int(point["metrics"]["n"]),
                        metrics=point["metrics"],
                        ci=ci,
                    )
                )
            for scope in ("lede", "full") if endpoint == "signal" else ("full",):
                row = article_row(articles, model, config, scope, policy)
                if row is None:
                    continue
                rows.append(
                    _signal_row(
                        **common,
                        scope=scope,
                        prompt_version=row.get("prompt_version"),
                        dataset="articles",
                        split="dev",
                        n=int(row["articles"]["n"]),
                        metrics=row["articles"],
                        ci=row["articles"]["cis"]["f05_macro"],
                    )
                )
    return rows


def holdout_rows(missing: list[str]) -> list[dict[str, Any]]:
    holdout = read_json(HOLDOUT_JSON)
    if holdout is None or holdout.get("dryRun"):
        missing.append(f"{_rel(HOLDOUT_JSON)} (the holdout has not run)")
        return []
    rows = []
    for r in holdout["rows"]:
        default_scope = "lede" if r["endpoint"] == "signal" else "full"
        rows.append(
            _signal_row(
                endpoint=r["endpoint"],
                model=r["model"],
                policy=r["policy"],
                scope=r["scope"] or default_scope,
                field_order=r.get("fieldOrder"),
                prompt_version=r.get("promptVersion"),
                dataset=r["dataset"],
                split=r["datasetSplit"],
                n=int(r["n"]),
                metrics=r["metrics"],
                ci=r["cis"].get("f05_macro"),
            )
        )
    return rows


def latency_rows(missing: list[str]) -> list[dict[str, Any]]:
    latency = read_json(LATENCY_JSON)
    if latency is None:
        missing.append(_rel(LATENCY_JSON))
        return []
    if not latency.get("clean"):
        missing.append("a clean latency run (the recorded one followed a failed preflight)")
    if latency.get("missing") or latency.get("partial"):
        missing.append(
            f"latency cells: {len(latency.get('missing', []))} unmeasured, "
            f"{len(latency.get('partial', []))} partial"
        )
    rows = []
    for c in latency["cells"]:
        warm = c["warm"]
        if not warm["n"] or not warm.get("totalMs"):
            continue
        rows.append(
            {
                "endpoint": c["endpoint"],
                "model": c["model"],
                "scope": c["scope"],
                "streaming": c["streaming"],
                "backend": c["backend"],
                "n": warm["n"],
                "p50Ms": warm["totalMs"]["p50"],
                "p95Ms": warm["totalMs"]["p95"],
                "ttftP50Ms": (warm.get("ttftMs") or {}).get("p50") if c["streaming"] else None,
                "timeToSignalP50Ms": (warm.get("signalMs") or {}).get("p50"),
            }
        )
    return rows


def summary_rows(missing: list[str]) -> list[dict[str, Any]]:
    summary = read_json(SUMMARY_JSON)
    dev = ((summary or {}).get("splits") or {}).get("dev")
    if dev is None:
        missing.append(f"{_rel(SUMMARY_JSON)} (dev split)")
        return []
    if not dev.get("judged"):
        missing.append("Opus judge scores on the dev summaries")
    keys = INTERFACES["SummaryBenchmark"]
    return [{k: r.get(k) for k in keys} for r in dev["rows"]]


def extraction_rows(missing: list[str]) -> list[dict[str, Any]]:
    matrix = read_json(EXTRACTION_JSON)
    if matrix is None:
        missing.append(_rel(EXTRACTION_JSON))
        return []
    keys = INTERFACES["ExtractionBenchmark"]
    return [{k: r.get(k) for k in keys} for r in matrix["rows"]]


def selection_block(missing: list[str]) -> dict[str, dict[str, str]]:
    selection = read_json(SELECTION_JSON)
    if selection is None:
        missing.append(_rel(SELECTION_JSON))
        return {e: {"model": "-", "reason": "Not selected yet: run `select`."} for e in ENDPOINTS}
    if selection.get("provisional"):
        missing.append("a final (non-provisional) selection")
    return frontend_selection(selection)


def article_stats(missing: list[str]) -> dict[str, Any]:
    stats = read_json(STATS_JSON)
    if stats is None:
        missing.append(_rel(STATS_JSON))
        return {"n": 0, "perSource": {}, "agreement": None, "kappa": None}
    pool = stats.get("pool") or {}
    return {
        "n": stats["n"],
        "perSource": {s: {k: c.get(k, 0) for k in PER_SOURCE_KEYS} for s, c in stats["per_source"].items()},
        # Agreement of the two independent labellers on the double-labelled pool, before adjudication.
        "agreement": pool.get("raw_agreement"),
        "kappa": pool.get("cohen_kappa"),
    }


def environment(missing: list[str]) -> dict[str, Any]:
    """From the latency preflight; without one, the extraction bench host and the grid's Ollama, said so."""
    pf = (read_json(PREFLIGHT_JSON) or {}).get("environment")
    if pf:
        return {
            "chip": pf.get("chip") or "unknown",
            "ramGb": pf.get("ramGb") or 0,
            "ollamaVersion": pf.get("ollamaVersion") or "unknown",
            "backend": pf.get("backend") or "native-metal",
            "gitSha": git_sha(),
        }
    missing.append(f"{_rel(PREFLIGHT_JSON)} (environment taken from the extraction bench and grid manifest)")
    host = (read_json(EXTRACTION_JSON) or {}).get("environment") or {}
    grid = read_json(GRID_DIR / "manifest.json") or {}
    return {
        "chip": host.get("chip") or "unknown",
        "ramGb": host.get("ramGb") or 0,
        "ollamaVersion": grid.get("ollama_version") or "unknown",
        "backend": grid.get("backend") or "native-metal",
        "gitSha": git_sha(),
    }


def build_benchmarks() -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    payload = {
        "version": 1,
        "sample": False,
        "generatedAt": _now(),
        "environment": environment(missing),
        "signal": signal_rows(missing),
        "latency": latency_rows(missing),
        "summary": summary_rows(missing),
        "extraction": extraction_rows(missing),
        "selection": selection_block(missing),
        "articles": article_stats(missing),
    }
    return payload, missing


# --- results README --------------------------------------------------------------------------


# First match wins; patterns are posix paths relative to evals/results.
DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("README.md", "this index (`report`)"),
    ("benchmarks.json", "BenchmarksV1 payload the frontend Benchmarks page renders (`report`)"),
    ("frozen.json", "frozen prompt versions and thresholds per model, with the evidence for each (`freeze`)"),
    ("holdout_runs.json", "holdout-once guard keyed by the sha256 of frozen.json (`holdout run`)"),
    ("ollama_probe.json", "Ollama capability and latency probe on synthetic inputs (P2 sanity gate)"),
    ("grid/manifest.json", "dev300 grid manifest: prompt versions, cache keys and model digests per cell"),
    ("grid/summary.md", "dev300 grid report: best config per model, knobs, error patterns, notes"),
    ("grid/notes.md", "hand-written error-analysis notes, merged into grid/summary.md"),
    ("grid/error_patterns.json", "error counts per model and keyword bucket on dev300 (counts only)"),
    ("grid/signal__*.json", "per-model /signal grid report on dev300: every config, threshold sweep, CIs"),
    ("grid/analyze__*.json", "per-model /analyze grid report on dev300: every config, threshold sweep, CIs"),
    ("articles_dev/manifest.json", "articles dev finalist run manifest"),
    (
        "articles_dev/summary.json",
        "finalists on the 30 dev articles at their dev300 thresholds (`finalists`)",
    ),
    ("articles_dev/summary.md", "readable articles dev finalist report"),
    ("articles_dev/notes.md", "hand-written notes, merged into articles_dev/summary.md"),
    (
        "summary/summary.json",
        "summary evaluation per endpoint and model: checks, key facts, judge (`summary`)",
    ),
    ("summary/summary.md", "readable summary evaluation report"),
    ("summary/spotcheck.*.json", "key-fact coverage threshold spot check: ids, cosines and hand labels"),
    ("extraction/matrix.json", "extraction matrix: parser x extractor quality, speed, install size"),
    ("extraction/summary.md", "readable extraction matrix"),
    ("latency/preflight.json", "latency preflight: machine, Ollama version and service env, load, memory"),
    ("latency/latency.json", "latency suite aggregates per cell with R7 budget marks (`latency`)"),
    ("latency/latency.md", "readable latency report"),
    ("latency/raw/docker-cpu-*.jsonl", "raw docker-cpu latency samples, one line per request (ids, numbers)"),
    ("latency/raw/*.jsonl", "raw native latency samples of one preflight session (ids, numbers)"),
    ("selection/selection.json", "per-endpoint winners, every candidate with numbers and verdict, R8 bundle"),
    ("selection/selection.md", "readable selection report (`select`)"),
    ("holdout/manifest.json", "holdout run manifest: cache keys and model digests per cell"),
    ("holdout/holdout.json", "one-time holdout metrics with CIs and dev-vs-holdout gaps (`holdout run`)"),
    ("holdout/holdout.md", "readable holdout report"),
)


def describe(rel_path: str) -> str | None:
    return next((text for pattern, text in DESCRIPTIONS if fnmatch.fnmatch(rel_path, pattern)), None)


def results_files(root: Path = RESULTS_DIR) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
    )


def render_readme(files: Sequence[str], missing: Sequence[str], payload: Mapping[str, Any]) -> str:
    guard = (read_json(GUARD) or {}).get("runs", {})
    lines = [
        "# Evaluation results",
        "",
        "Everything here is numbers, ids, model names and hand-written notes: no article, summary or dataset "
        "text (publisher and dataset rights). Inputs and caches stay in `evals/data/private/` (gitignored). "
        "Regenerate this index and `benchmarks.json` with `uv run python -m veritrade_evals report`.",
        "",
        "Order of the pipeline: grid -> finalists -> freeze -> summary -> latency -> select -> holdout "
        "(once) -> report.",
        "",
        f"## Status at {payload['generatedAt']}",
        "",
        f"- Selection: {', '.join(f'/{e} `{s["model"]}`' for e, s in payload['selection'].items())}.",
        f"- Benchmarks rows: signal {len(payload['signal'])}, latency {len(payload['latency'])}, summary "
        f"{len(payload['summary'])}, extraction {len(payload['extraction'])}.",
        f"- Holdout runs on record: {len(guard)}"
        + "".join(f" (`{sha[:12]}` {e['status']})" for sha, e in guard.items())
        + ".",
        *(
            ["- Missing or incomplete inputs:", *(f"  - {m}" for m in missing)]
            if missing
            else ["- Complete."]
        ),
        "",
        "## Files",
        "",
        "| file | what it holds |",
        "|---|---|",
    ]
    for rel_path in files:
        text = describe(rel_path) or "(undocumented: add it to report.DESCRIPTIONS)"
        lines.append(f"| `{rel_path}` | {text} |")
    lines += [
        "",
        "Notes on `benchmarks.json`: dev300 rows carry the endpoint's default scope (tweets are "
        "headline-length); latency rows are warm p50/p95 excluding fetch; `articles.agreement` and "
        "`kappa` are the two labellers on the double-labelled pool before adjudication.",
        "",
    ]
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    if args.check:
        problems = validate(read_json(BENCHMARKS_JSON))
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{_rel(BENCHMARKS_JSON)}: {'invalid' if problems else 'valid BenchmarksV1'}")
        return 1 if problems else 0
    payload, missing = build_benchmarks()
    problems = validate(payload)
    if problems:
        for problem in problems[:20]:
            print(f"shape: {problem}", file=sys.stderr)
        print("benchmarks.json NOT written: the payload breaks the BenchmarksV1 contract", file=sys.stderr)
        return 1
    BENCHMARKS_JSON.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    files = results_files()
    README.write_text(render_readme(files, missing, payload), encoding="utf-8")
    print(
        f"wrote {_rel(BENCHMARKS_JSON)} (valid BenchmarksV1) and {_rel(README)} ({len(files)} files indexed)"
    )
    for item in missing:
        print(f"missing: {item}", file=sys.stderr)
    undocumented = [f for f in files if describe(f) is None]
    for item in undocumented:
        print(f"undocumented results file: {item}", file=sys.stderr)
    return 1 if args.strict and (missing or undocumented) else 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `report` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "report", help="benchmarks.json (BenchmarksV1) and the results README"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", help="exit 1 while any input is missing")
    mode.add_argument("--check", action="store_true", help="only validate the existing benchmarks.json")
    return parser


def run(args: argparse.Namespace) -> int:
    return cmd_report(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.report", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["report", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
