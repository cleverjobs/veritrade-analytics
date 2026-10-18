"""Grid finalists on the articles dev set: `uv run python -m veritrade_evals finalists {list,run,analyze}`.
Inputs are the serving extraction of the stored HTML cut to the serving scope, and every threshold comes
from dev300, so this checks the frozen operating points on real articles rather than tuning them."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from veritrade.config import get_settings
from veritrade.errors import VtError
from veritrade.extract import registry as extraction
from veritrade.extract.scope import build_input
from veritrade.extract.types import Article, ExtractorName, ParserName, Scope
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Sentiment
from veritrade_evals import grid
from veritrade_evals.analysis import (
    POLICIES,
    TAU_BASE,
    Candidate,
    CellInput,
    DevSet,
    EvalPolicy,
    Served,
    best_candidate,
    bucket_of,
    candidate_cis,
    candidates,
    chosen_point,
    counts,
    error_patterns,
    evaluate_policy,
    grid_only,
    is_tau,
    latency,
    metrics_dict,
    paired_f05,
    point_row,
    rounded,
    threshold_text,
)
from veritrade_evals.articles import ArticleExample, ArticlePaths, ArticleSplit, load_articles, read_fetched
from veritrade_evals.metrics import INVERSION_CAP, SweepPoint, select_operating_point
from veritrade_evals.paths import PREDS_DIR, RESULTS_DIR
from veritrade_evals.predcache import canonical_json
from veritrade_evals.runner import RunSpec, make_resident, model_digest, run_cell
from veritrade_evals.splits import load_split

ARTICLE_SPLIT: ArticleSplit = "dev"
CACHE_SPLIT = "articles_dev"
OUT_DIR = RESULTS_DIR / "articles_dev"
MANIFEST = OUT_DIR / "manifest.json"
# The serving request defaults; test_finalists pins them to the API contract.
PARSER: ParserName = "selectolax"
EXTRACTOR: ExtractorName = "cascade"
# A collapse: the whole coverage CI sits under this share of the dev300 coverage.
COLLAPSE_SHARE = 0.5
DEFAULT_CONCURRENCY = 1


@dataclass(frozen=True, slots=True)
class Finalist:
    model: str
    config: str
    scope: Scope

    @property
    def key(self) -> str:
        return f"{self.model}/{self.config}/{self.scope}"


SIGNAL_FINALISTS = {"clef-flash:9b-q8_0": "v4", "nimble:9b-q4_K_M": "v1", "qwen3.5:0.8b-q8_0": "v2-label"}
ANALYZE_FINALISTS = {
    "qwen3.5:4b-q4_K_M": ("v1", "v5", "v1-summary-first"),
    "ministral-3:3b": ("v1", "v5", "v1-summary-first"),
    "qwen3:4b-instruct-2507-q4_K_M": ("v1", "v5"),
}
SIGNAL_SCOPES: tuple[Scope, ...] = ("lede", "full")
FINALISTS: tuple[Finalist, ...] = (
    *(Finalist(m, c, s) for m, c in SIGNAL_FINALISTS.items() for s in SIGNAL_SCOPES),
    *(Finalist(m, c, "full") for m, configs in ANALYZE_FINALISTS.items() for c in configs),
)


def finalist_cells(finalists: Sequence[Finalist] = FINALISTS) -> list[tuple[Finalist, RunSpec]]:
    cells = {(c.model, c.name): c for c in grid.grid_cells({f.model for f in finalists})}
    return [(f, cells[(f.model, f.config)]) for f in finalists]


# --- inputs ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArticleInput:
    id: str
    text: str
    label: Sentiment


@dataclass(frozen=True, slots=True)
class Extracted:
    example: ArticleExample
    article: Article | None
    error: str | None = None


def extract_all(examples: Sequence[ArticleExample], paths: ArticlePaths) -> list[Extracted]:
    """The serving extractor on the stored HTML at the final URL, as a live request would see it."""
    out = []
    for example in examples:
        fetched = read_fetched(paths, example.id)
        url = fetched.final_url if fetched and fetched.final_url else example.url
        html = example.html_path.read_text(encoding="utf-8")
        try:
            article = extraction.extract(html, url, parser=PARSER, extractor=EXTRACTOR)
        except VtError as exc:
            out.append(Extracted(example, None, exc.code.value))
            continue
        out.append(Extracted(example, article))
    return out


def inputs_for(
    extracted: Sequence[Extracted], scope: Scope, max_input_tokens: int
) -> tuple[list[ArticleInput], dict[str, Any]]:
    """Model inputs plus their identity for the cache key, so an extraction change re-runs the cell."""
    items = [
        ArticleInput(e.example.id, build_input(e.article, scope, max_input_tokens), e.example.label)
        for e in extracted
        if e.article is not None
    ]
    sha = hashlib.sha256(canonical_json([[i.id, i.text] for i in items]).encode("utf-8")).hexdigest()
    identity = {
        "articles": ARTICLE_SPLIT,
        "parser": PARSER,
        "extractor": EXTRACTOR,
        "scope": scope,
        "sha": sha,
    }
    return items, identity


# --- run -------------------------------------------------------------------------------------


def read_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if not path.exists():
        return {"split": ARTICLE_SPLIT, "cells": {}}
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


async def run_finalists(
    client: OllamaClient,
    cells: Sequence[tuple[Finalist, RunSpec]],
    extracted: Sequence[Extracted],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    preds_root: Path = PREDS_DIR,
    manifest_path: Path = MANIFEST,
    log: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Model by model on a shared Ollama (nothing is unloaded); the manifest is rewritten after every cell."""
    manifest = read_manifest(manifest_path)
    manifest["split"] = ARTICLE_SPLIT
    manifest["ollama_version"] = await client.version()
    manifest["backend"] = get_settings().ollama_backend_label
    for model, group in itertools.groupby(cells, key=lambda fc: fc[1].model):
        digest = await model_digest(client, model)
        prepared = [
            (f, cell, *inputs_for(extracted, f.scope, cell.spec.max_input_tokens)) for f, cell in group
        ]
        if any(
            not {i.id for i in items} <= cell.cache(CACHE_SPLIT, digest, preds_root, identity).done_ids()
            for _, cell, items, identity in prepared
        ):
            print(f"== {model}: warming up", file=log, flush=True)
            await make_resident(client, prepared[0][1], shared=True, log=log)
        for finalist, cell, items, identity in prepared:
            stats = await run_cell(
                client,
                cell,
                items,
                split=CACHE_SPLIT,
                model_digest=digest,
                concurrency=concurrency,
                root=preds_root,
                inputs=identity,
                log=log,
            )
            records = cell.cache(CACHE_SPLIT, digest, preds_root, identity).records()
            manifest["cells"][finalist.key] = {
                "endpoint": cell.endpoint,
                "model": cell.model,
                "config": cell.name,
                "scope": finalist.scope,
                "variant": cell.variant,
                "prompt_version": cell.prompt.version,
                "prompt_sha": cell.prompt_sha(),
                "run_config": json.loads(canonical_json(cell.run_config(CACHE_SPLIT, identity))),
                "model_digest": digest,
                "cache_key": stats.key,
                "inputs": len(items),
                "extraction_failures": len(extracted) - len(items),
                "done": len(records),
                "errors": sum(1 for r in records.values() if r.get("error") is not None),
                "concurrency": concurrency,
                "shared_ollama": True,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            grid.write_manifest(manifest, manifest_path)
    return manifest


# --- analyze ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dev300Point:
    """A config's dev300 operating point under one policy: the threshold the articles are scored at."""

    candidate: Candidate
    point: SweepPoint
    cis: Mapping[str, Any]
    best: bool


def dev300_points(
    configs: Collection[tuple[str, str]], n_boot: int, log: TextIO
) -> dict[tuple[str, str, str], Dev300Point]:
    """(model, config, policy) -> dev300 operating point of each wanted config, from the grid caches."""
    dev = grid.dev_set(load_split(grid.GRID_SPLIT))
    models = sorted({m for m, _ in configs})
    inputs = grid.cell_inputs(grid.grid_cells(models), grid.read_manifest(), dev, PREDS_DIR, log)
    out: dict[tuple[str, str, str], Dev300Point] = {}
    for model, group in itertools.groupby(inputs, key=lambda c: c.model):
        cands = candidates(list(group), dev)
        best, _ = best_candidate(grid_only(cands))
        for c in cands:
            if (model, c.cell.name) not in configs:
                continue
            point = chosen_point(c, c.evaluation.operating is not None)
            out[(model, c.cell.name, c.evaluation.policy)] = Dev300Point(
                c, point, candidate_cis(c, point, dev, n_boot), c is best
            )
    return out


def article_dev(extracted: Sequence[Extracted]) -> DevSet:
    return DevSet(
        ids=tuple(e.example.id for e in extracted),
        gold=tuple(e.example.label for e in extracted),
        weights=None,
        buckets=tuple(bucket_of(e.article.title) if e.article else "extraction failed" for e in extracted),
    )


def article_cells(
    cells: Sequence[tuple[Finalist, RunSpec]],
    extracted: Sequence[Extracted],
    manifest: Mapping[str, Any],
    dev: DevSet,
    preds_root: Path,
    log: TextIO,
) -> list[tuple[Finalist, CellInput]]:
    out = []
    for finalist, cell in cells:
        entry = manifest.get("cells", {}).get(finalist.key)
        if entry is None:
            print(f"skip {finalist.key}: not run yet", file=log)
            continue
        _, identity = inputs_for(extracted, finalist.scope, cell.spec.max_input_tokens)
        current = json.loads(canonical_json(cell.run_config(CACHE_SPLIT, identity)))
        if entry["prompt_sha"] != cell.prompt_sha() or entry["run_config"] != current:
            print(f"skip {finalist.key}: prompt, config or extraction changed since the run", file=log)
            continue
        records = cell.cache(CACHE_SPLIT, entry["model_digest"], preds_root, identity).records()
        aligned = tuple(records.get(i) for i in dev.ids)
        out.append(
            (
                finalist,
                CellInput(
                    endpoint=cell.endpoint,
                    model=cell.model,
                    name=f"{cell.name} {finalist.scope}",
                    variant=cell.variant,
                    prompt_version=cell.prompt.version,
                    records=aligned,
                    dims=dict(cell.dims),
                    ordinal_t=cell.spec.ordinal_t,
                    meta={
                        "cache_key": entry["cache_key"],
                        "extraction_failures": entry["extraction_failures"],
                    },
                ),
            )
        )
    return out


def _counts(point: SweepPoint) -> dict[str, int]:
    """Raw counts behind the rates; at n=30 a single inversion is 5% of the calls."""
    conf = point.metrics.confusion
    return {
        "directional_calls": sum(row[0] + row[1] for row in conf),
        "inversions": conf[0][1] + conf[1][0],
        "abstained": sum(row[3] for row in conf),
    }


@dataclass(frozen=True, slots=True)
class Scored:
    finalist: Finalist
    cell: CellInput
    policy: EvalPolicy
    candidate: Candidate
    point: SweepPoint
    dev300: Dev300Point
    served: bool


def score_cells(
    cells: Sequence[tuple[Finalist, CellInput]], dev: DevSet, d300: Mapping[tuple[str, str, str], Dev300Point]
) -> list[Scored]:
    out = []
    for finalist, cell in cells:
        for i, policy in enumerate(POLICIES[cell.variant]):
            ref = d300.get((finalist.model, finalist.config, policy))
            if ref is None:
                continue
            evaluation = evaluate_policy(policy, cell.records, dev.gold, None, ordinal_t=cell.ordinal_t)
            point = next(p for p in evaluation.sweep if p.threshold == ref.point.threshold)
            # Signal finalists serve the dev300 best rule; /analyze configs have one rule each.
            served = ref.best or cell.endpoint == "analyze"
            out.append(Scored(finalist, cell, policy, Candidate(cell, evaluation, i), point, ref, served))
    return out


def clear_failure(s: Scored, cis: Mapping[str, Any]) -> list[str]:
    """Only failures the n=30 CI is sure of: the whole inversion CI above the cap, or a coverage collapse."""
    reasons = []
    if cis["inversion_rate"]["low"] > INVERSION_CAP:
        reasons.append(f"inversion CI low {cis['inversion_rate']['low']:.3f} > cap {INVERSION_CAP}")
    if cis["coverage"]["high"] < COLLAPSE_SHARE * s.dev300.point.coverage:
        dev300 = s.dev300.point.coverage
        reasons.append(
            f"coverage CI high {cis['coverage']['high']:.3f} < {COLLAPSE_SHARE} x dev300 {dev300:.3f}"
        )
    return reasons


def row_report(s: Scored, dev: DevSet, n_boot: int) -> dict[str, Any]:
    cis = candidate_cis(s.candidate, s.point, dev, n_boot)
    failure = clear_failure(s, cis) if s.served else []
    report: dict[str, Any] = {
        "key": s.finalist.key,
        "endpoint": s.cell.endpoint,
        "model": s.finalist.model,
        "config": s.finalist.config,
        "scope": s.finalist.scope,
        "variant": s.cell.variant,
        "policy": s.policy,
        "prompt_version": s.cell.prompt_version,
        "threshold": s.point.threshold,
        "served_rule": s.served,
        "articles": {**metrics_dict(s.point.metrics), **_counts(s.point), "cis": cis},
        "dev300": {
            "f05_macro": s.dev300.point.f05_macro,
            "coverage": s.dev300.point.coverage,
            "inversion_rate": s.dev300.point.inversion_rate,
            "cis": s.dev300.cis,
        },
        "counts": counts(s.cell.records),
        "extraction_failures": s.cell.meta.get("extraction_failures", 0),
        "latency": latency(s.cell.records),
        "clear_failure": failure,
    }
    if failure:
        # A threshold that makes no directional call meets the cap trivially, so it is no re-pick.
        op = select_operating_point([p for p in s.candidate.evaluation.sweep if p.metrics.signal_rate > 0])
        report["repick"] = None if op is None else point_row(op)
    return report


def _paired(a: Scored, b: Scored, dev: DevSet, n_boot: int, what: str) -> dict[str, Any]:
    boot = paired_f05((a.candidate, a.point), (b.candidate, b.point), dev, n_boot)
    gap = boot["gap"]
    return {
        "comparison": what,
        "a": a.finalist.key + f" ({a.policy})",
        "b": b.finalist.key + f" ({b.policy})",
        "a_f05": boot["a"].point,
        "b_f05": boot["b"].point,
        "gap": {"point": gap.point, "low": gap.low, "high": gap.high},
        "verdict": "a" if gap.low > 0 else "b" if gap.high < 0 else "tie",
    }


def comparisons(scored: Sequence[Scored], dev: DevSet, n_boot: int) -> list[dict[str, Any]]:
    by = {(s.finalist.model, s.finalist.config, s.finalist.scope, s.policy): s for s in scored}
    out = []
    for s in scored:
        model, config, scope, policy = s.finalist.model, s.finalist.config, s.finalist.scope, s.policy
        pairs: list[tuple[Scored | None, str]] = []
        if s.cell.endpoint == "signal" and scope == "lede" and s.served:
            pairs.append((by.get((model, config, "full", policy)), "scope: lede - full"))
        if is_tau(policy):
            pairs.append((by.get((model, config, scope, TAU_BASE[policy])), "rule: tau - plain"))
        if s.cell.endpoint == "analyze" and config != "v1":
            order = s.cell.variant == "summary-first"
            what = "field order: summary-first - sentiment-first" if order else f"prompt: {config} - v1"
            pairs.append((by.get((model, "v1", scope, policy)), what))
        out += [_paired(s, other, dev, n_boot, what) for other, what in pairs if other is not None]
    return out


def misses(scored: Sequence[Scored], dev: DevSet, extracted: Sequence[Extracted]) -> dict[str, Any]:
    """How the served rules miss, by title keyword bucket and by source (counts only)."""
    served = []
    for s in scored:
        if not s.served or (s.cell.endpoint == "signal" and s.finalist.scope != "lede"):
            continue
        if s.cell.endpoint == "analyze" and s.cell.variant != "sentiment-first":
            continue
        pred, abstained = s.candidate.evaluation.served(s.point.threshold)
        failed = tuple(r is None or r.get("error") is not None for r in s.cell.records)
        served.append(Served(s.finalist.key, tuple(pred), tuple(abstained), failed))
    buckets = sorted(set(dev.buckets))
    sources = tuple(e.example.source for e in extracted)
    by_source = DevSet(dev.ids, dev.gold, None, buckets=sources)
    return {
        "by_title_bucket": error_patterns(served, dev, buckets),
        "by_source": error_patterns(served, by_source, sorted(set(sources)))["buckets"],
    }


def analyze_finalists(
    *,
    n_boot: int = 1000,
    preds_root: Path = PREDS_DIR,
    manifest_path: Path = MANIFEST,
    out_dir: Path = OUT_DIR,
    paths: ArticlePaths | None = None,
    log: TextIO = sys.stderr,
) -> list[Path]:
    paths = paths or ArticlePaths()
    extracted = extract_all(load_articles(ARTICLE_SPLIT, paths=paths), paths)
    dev = article_dev(extracted)
    cells = finalist_cells()
    inputs = article_cells(cells, extracted, read_manifest(manifest_path), dev, preds_root, log)
    if not inputs:
        print("nothing to analyze: run the finalists first", file=log)
        return []
    d300 = dev300_points({(f.model, f.config) for f, _ in inputs}, n_boot, log)
    scored = score_cells(inputs, dev, d300)
    gold = Counter(dev.gold)
    report = rounded(
        {
            "split": ARTICLE_SPLIT,
            "n": len(dev.ids),
            "gold": {k: gold.get(k, 0) for k in ("bullish", "bearish", "neutral")},
            "weights": None,
            "extraction": {
                "parser": PARSER,
                "extractor": EXTRACTOR,
                "failures": sum(e.article is None for e in extracted),
                "failure_codes": dict(Counter(e.error for e in extracted if e.error)),
            },
            "inversion_cap": INVERSION_CAP,
            "collapse_share": COLLAPSE_SHARE,
            "n_boot": n_boot,
            "generated_by": "uv run python -m veritrade_evals finalists analyze",
            "rows": [row_report(s, dev, n_boot) for s in scored],
            "comparisons": comparisons(scored, dev, n_boot),
            "misses": misses(scored, dev, extracted),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    md_path = out_dir / "summary.md"
    manifest = read_manifest(manifest_path)
    notes_path = out_dir / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else None
    md_path.write_text(render(report, manifest, notes), encoding="utf-8")
    return [json_path, md_path]


# --- markdown --------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _ci(ci: Mapping[str, float] | None, digits: int = 3) -> str:
    if ci is None or ci.get("point") is None:
        return "-"
    return f"{ci['point']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _metric_row(row: Mapping[str, Any]) -> list[str]:
    a = row["articles"]
    pc = a["per_class"]
    return [
        _ci(a["cis"]["f05_macro"]),
        f"{_num(pc['bullish']['precision'])} / {_num(pc['bullish']['recall'])}",
        f"{_num(pc['bearish']['precision'])} / {_num(pc['bearish']['recall'])}",
        f"{a['inversions']}/{a['directional_calls']}",
        _pct(a["coverage"]),
        _ci(row["dev300"]["cis"].get("f05_macro")),
    ]


def _confusion(row: Mapping[str, Any]) -> str:
    counts_ = row["articles"]["confusion"]["counts"]
    return " ; ".join("/".join(str(v) for v in r) for r in counts_)


def _group_n(row: Mapping[str, Any]) -> str:
    gold = row["gold"]
    return f"{row['n']} ({gold['bullish']}/{gold['bearish']}/{gold['neutral']})"


def render(report: Mapping[str, Any], manifest: Mapping[str, Any], notes: str | None = None) -> str:
    gold = report["gold"]
    ext = report["extraction"]
    rows = report["rows"]
    signal = [r for r in rows if r["endpoint"] == "signal"]
    analyze = [r for r in rows if r["endpoint"] == "analyze"]
    concurrency = sorted({e.get("concurrency") for e in manifest.get("cells", {}).values()} - {None})
    lines = [
        "# Grid finalists on the articles dev set",
        "",
        "Generated by `uv run python -m veritrade_evals finalists analyze` from the local prediction cache; "
        "counts and rates only, no article text.",
        "",
        f"- Data: the articles dev split, n={report['n']} news articles labelled for their primary entity "
        f"(bullish {gold['bullish']}, bearish {gold['bearish']}, neutral {gold['neutral']}). Balanced and "
        "unweighted: unlike dev300, rates are not reweighted to a class prior.",
        f"- Inputs: the serving path. `{ext['extractor']}` extraction with `{ext['parser']}` on the stored "
        "HTML at the fetch's final URL, then the endpoint scope (headline + lede, or the full body) cut to "
        f"each model's `max_input_tokens`. Extraction failures: {ext['failures']} (no call; scored as "
        "abstentions).",
        "- Thresholds: each config's dev300 operating point (max F0.5 with inversions <= "
        f"{report['inversion_cap']:.0%}), never re-tuned here. A rule counts as clearly failing only when "
        "the whole 95% CI of its inversion rate is above the cap, or the whole coverage CI is below "
        f"{report['collapse_share']:.0%} of its dev300 coverage.",
        f"- CIs: 95% stratified bootstrap (B={report['n_boot']}). With 10 articles per class they are wide: "
        "a gap needs roughly 0.2 F0.5 to clear zero, so most comparisons below are ties.",
        f"- Latency: wall ms per call on an Ollama {manifest.get('ollama_version', '?')} "
        f"{manifest.get('backend', '?')} shared with other eval runs (client concurrency "
        f"{', '.join(str(c) for c in concurrency) or '?'}); indicative only.",
        "",
        "## /signal finalists",
        "",
        "Served = the dev300 best rule for the model. P / R are precision / recall; inversions are raw "
        "bullish<->bearish calls over directional calls.",
        "",
    ]
    header = [
        "model", "config", "rule", "threshold (dev300)", "scope", "served", "F0.5 [95% CI]", "bull P / R",
        "bear P / R", "inversions", "coverage", "dev300 F0.5 [95% CI]", "p50 ms",
    ]  # fmt: skip
    lines += _table(
        header,
        [
            [
                f"`{r['model']}`",
                r["config"],
                r["policy"],
                threshold_text(r["policy"], r["threshold"]),
                r["scope"],
                "yes" if r["served_rule"] else "",
                *_metric_row(r)[:5],
                _metric_row(r)[5],
                "-" if r["latency"]["wall_ms_p50"] is None else f"{r['latency']['wall_ms_p50']:.0f}",
            ]
            for r in signal
        ],
    )
    lines += ["", "## /analyze (full article, ordinal |score| >= t)", ""]
    header = [
        "model", "config", "order", "threshold (dev300)", "F0.5 [95% CI]", "bull P / R", "bear P / R",
        "inversions", "coverage", "dev300 F0.5 [95% CI]", "p50 ms",
    ]  # fmt: skip
    lines += _table(
        header,
        [
            [
                f"`{r['model']}`",
                r["config"],
                r["variant"],
                threshold_text(r["policy"], r["threshold"]),
                *_metric_row(r),
                "-" if r["latency"]["wall_ms_p50"] is None else f"{r['latency']['wall_ms_p50']:.0f}",
            ]
            for r in analyze
        ],
    )
    lines += [
        "",
        "/analyze latency is time to the score: sentiment-first stops the stream there, summary-first "
        "has to generate the whole summary first.",
        "",
        "## Paired comparisons (same 30 articles, paired bootstrap on the F0.5 gap)",
        "",
    ]
    lines += _table(
        ["comparison", "a", "b", "F0.5 a", "F0.5 b", "gap a - b [95% CI]", "verdict"],
        [
            [
                c["comparison"],
                c["a"],
                c["b"],
                f"{c['a_f05']:.3f}",
                f"{c['b_f05']:.3f}",
                f"{c['gap']['point']:+.3f} [{c['gap']['low']:+.3f}, {c['gap']['high']:+.3f}]",
                {"a": "a wins", "b": "b wins", "tie": "tie"}[c["verdict"]],
            ]
            for c in report["comparisons"]
        ],
    )
    lines += [
        "",
        "## Confusion matrices (raw counts)",
        "",
        "Rows gold bullish ; bearish ; neutral, each as predicted bullish/bearish/neutral/abstained.",
        "",
    ]
    lines += _table(
        ["finalist", "rule", "confusion"],
        [[r["key"], r["policy"], _confusion(r)] for r in rows],
    )
    failures = [r for r in rows if r["clear_failure"]]
    lines += ["", "## Clear-failure check on the served rules", ""]
    if not failures:
        lines.append("No served rule clearly fails on the articles, so every dev300 operating point stands.")
    for r in failures:
        repick = r.get("repick")
        lines.append(
            f"- {r['key']} ({r['policy']}): {'; '.join(r['clear_failure'])}. Articles re-pick: "
            + ("none meets the cap" if repick is None else threshold_text(r["policy"], repick["threshold"]))
        )
    lines += ["", "## Misses of the served rules (served scope, raw counts)", ""]
    patterns = report["misses"]["by_title_bucket"]
    models = list(patterns["per_model"])
    lines += _table(
        ["finalist", "directional calls", "wrong", "missed bull", "missed bear", "failed"],
        [
            [
                m,
                *(
                    str(t[k])
                    for k in (
                        "directional_calls",
                        "wrong_calls",
                        "bullish_missed",
                        "bearish_missed",
                        "failed",
                    )
                ),
            ]
            for m, t in patterns["per_model"].items()
        ],
    )
    for title, buckets in (
        ("By headline keyword bucket (first match wins; heuristic):", patterns["buckets"]),
        ("By source:", report["misses"]["by_source"]),
    ):
        lines += [
            "",
            title,
            "",
            *_table(
                ["group", "n (bull/bear/neu)", *(f"{m} wrong/calls, missed" for m in models)],
                [
                    [
                        name,
                        _group_n(row),
                        *(
                            f"{v['wrong_calls']}/{v['directional_calls']}, {v['missed_directional']}"
                            for v in (row["models"][m] for m in models)
                        ),
                    ]
                    for name, row in buckets.items()
                ],
            ),
        ]
    if notes:
        lines += ["", notes.strip()]
    return "\n".join(lines) + "\n"


# --- CLI -------------------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    for finalist, cell in finalist_cells():
        print(f"{cell.endpoint:8s} {finalist.key:50s} {cell.prompt.version}")
    print(f"{len(FINALISTS)} cells on the articles {ARTICLE_SPLIT} split")
    return 0


async def _run(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    extracted = extract_all(load_articles(ARTICLE_SPLIT, paths=paths), paths)
    cells = finalist_cells()
    if args.model:
        cells = [fc for fc in cells if fc[1].model in args.model]
    settings = get_settings()
    client = OllamaClient(
        args.base_url or settings.ollama_base_url,
        timeout=settings.inference_timeout_s,
        keep_alive=settings.ollama_keep_alive,
    )
    async with client:
        manifest = await run_finalists(
            client,
            cells,
            extracted,
            concurrency=args.concurrency,
            preds_root=args.preds_root,
            manifest_path=args.manifest,
        )
    done = sum(e["done"] for e in manifest["cells"].values())
    errors = sum(e["errors"] for e in manifest["cells"].values())
    print(f"finalists: {len(manifest['cells'])} cells, {done} predictions, {errors} errors", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(_run(args))


def cmd_analyze(args: argparse.Namespace) -> int:
    paths = analyze_finalists(
        n_boot=args.n_boot, preds_root=args.preds_root, manifest_path=args.manifest, out_dir=args.out_dir
    )
    for path in paths:
        print(path)
    return 0 if paths else 1


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `finalists` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "finalists", help="grid finalists on the articles dev set (serving extraction, dev300 thresholds)"
    )
    actions = parser.add_subparsers(dest="finalists_action", required=True)
    actions.add_parser("list", help="print the finalist cells").set_defaults(finalists_func=cmd_list)

    running = actions.add_parser("run", help="run (or resume) the finalists on a shared Ollama")
    running.add_argument("--model", action="append", help="restrict to this model (repeatable)")
    running.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    running.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    running.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    running.add_argument("--manifest", type=Path, default=MANIFEST)
    running.set_defaults(finalists_func=cmd_run)

    analyzing = actions.add_parser(
        "analyze", help="score cached predictions and write evals/results/articles_dev"
    )
    analyzing.add_argument("--n-boot", type=int, default=1000)
    analyzing.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    analyzing.add_argument("--manifest", type=Path, default=MANIFEST)
    analyzing.add_argument("--out-dir", type=Path, default=OUT_DIR)
    analyzing.set_defaults(finalists_func=cmd_analyze)
    return parser


def run(args: argparse.Namespace) -> int:
    code: int = args.finalists_func(args)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.finalists", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["finalists", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
