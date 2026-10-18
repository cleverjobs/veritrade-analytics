"""Per-endpoint winners from the frozen configs: `uv run python -m veritrade_evals select [--write]`.
`--write` puts the winners in models.toml [defaults] and the size-capped bundle (R8) in
deploy/bundled.env."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from veritrade.config import get_settings
from veritrade.errors import VtError
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Endpoint
from veritrade.models_registry import Registry, parse_models_toml
from veritrade_evals.latency import LATENCY_JSON, PREFLIGHT_JSON, Cell
from veritrade_evals.metrics import INVERSION_CAP
from veritrade_evals.metrics import Candidate as KneePoint
from veritrade_evals.metrics import knee as pareto_knee
from veritrade_evals.paths import REPO_ROOT, RESULTS_DIR
from veritrade_evals.predcache import model_slug

SELECTION_DIR = RESULTS_DIR / "selection"
SELECTION_JSON = SELECTION_DIR / "selection.json"
SELECTION_MD = SELECTION_DIR / "selection.md"
FROZEN_JSON = RESULTS_DIR / "frozen.json"
GRID_DIR = RESULTS_DIR / "grid"
ARTICLES_JSON = RESULTS_DIR / "articles_dev" / "summary.json"
SUMMARY_JSON = RESULTS_DIR / "summary" / "summary.json"
MODELS_TOML = REPO_ROOT / "backend" / "src" / "veritrade" / "data" / "models.toml"
BUNDLED_ENV = REPO_ROOT / "deploy" / "bundled.env"

ENDPOINTS: tuple[Endpoint, ...] = get_args(Endpoint)
# R8 "about 10 GB of model files", decimal: clef-flash alone (10.58 GB) must not fit.
SIZE_CAP_BYTES = 10_000_000_000
PASS_GATE = 0.8
NATIVE = "native-metal"

RULES: dict[str, str] = {
    "signal": "Candidates: the frozen /signal configs at the default lede scope. Filters: dev300 inversion "
    "rate <= 2% at the frozen operating point (n=300); the articles dev set must not fail the cap, meaning "
    "the 95% CI of its inversion rate reaches down to 2% (the freeze's clear-failure rule: at n=30 one "
    "inversion is about 5% of the calls); warm native p50 excluding fetch <= 500 ms at lede. Rank: dev300 "
    "F0.5 macro (primary, n=300), then articles dev F0.5, then lower latency.",
    "analyze": "Candidates: the frozen /analyze configs (full scope, ordinal |score| >= t, sentiment-first). "
    "Filters: the same inversion rules as /signal; streamed TTFT p50 <= 1.5 s and complete p50 <= 4 s "
    "(warm, native, excluding fetch). Rank: dev300 F0.5 macro, then articles dev F0.5, then lower latency.",
    "summary": "Candidates: the frozen /summary models. Filter: deterministic pass rate >= 0.8 on the dev "
    "articles; streamed TTFT p50 <= 1.5 s and complete p50 <= 4 s. Rank: Opus judge faithfulness, then judge "
    "coverage, then key-fact coverage (key-fact coverage leads while the judge is pending).",
    "winner": "The best-ranked candidate inside every filter. When no accuracy-eligible candidate meets its "
    "latency budget, the Pareto knee (metrics.knee) of quality against budget utilisation (the worst "
    "measured metric divided by its limit) among the measured candidates, with the reason stated.",
    "bundle": "R8: the per-endpoint winners when their model files total <= 10 GB (decimal), else the "
    "combination of accuracy-eligible candidates that fits with the fewest rank steps away from the winners "
    "(ties: signal, then summary, then analyze keeps its better pick; then the smaller total).",
}


class SelectError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _f(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(ci: Mapping[str, Any] | None) -> str:
    if not ci or ci.get("point") is None:
        return "-"
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


# --- inputs ----------------------------------------------------------------------------------


@dataclass(slots=True)
class Inputs:
    frozen: dict[str, Any]
    grid: dict[tuple[str, str], dict[str, Any]]
    articles: dict[str, Any] | None
    summary: dict[str, Any] | None
    latency: dict[str, Any] | None
    sizes: dict[str, int] | None
    sizes_source: str
    missing: list[str] = field(default_factory=list)


async def ollama_sizes(base_url: str) -> dict[str, int]:
    async with OllamaClient(base_url, timeout=10.0) as client:
        return {str(e["name"]): int(e["size"]) for e in await client.tags() if isinstance(e.get("size"), int)}


def model_sizes(base_url: str | None) -> tuple[dict[str, int] | None, str]:
    """Installed sizes from Ollama's tags, else the ones the latency preflight recorded."""
    try:
        return asyncio.run(ollama_sizes(base_url or get_settings().ollama_base_url)), "ollama /api/tags"
    except VtError:
        pass
    recorded = (read_json(PREFLIGHT_JSON) or {}).get("environment", {}).get("modelSizes")
    if recorded:
        return {str(k): int(v) for k, v in recorded.items()}, _rel(PREFLIGHT_JSON)
    return None, "unavailable"


def load_inputs(
    *,
    frozen: Path = FROZEN_JSON,
    grid_dir: Path = GRID_DIR,
    articles: Path = ARTICLES_JSON,
    summary: Path = SUMMARY_JSON,
    latency: Path = LATENCY_JSON,
    sizes: tuple[dict[str, int] | None, str] = (None, "not requested"),
) -> Inputs:
    frozen_payload = read_json(frozen)
    if frozen_payload is None:
        raise SelectError(f"{_rel(frozen)} is missing; run `freeze` first")
    grid: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    for model, endpoints in frozen_payload["models"].items():
        for endpoint in ("signal", "analyze"):
            if endpoint in endpoints:
                path = grid_dir / f"{endpoint}__{model_slug(model)}.json"
                if (report := read_json(path)) is not None:
                    grid[(endpoint, model)] = report
                else:
                    missing.append(_rel(path))
    loaded = {p: read_json(p) for p in (articles, summary, latency)}
    missing += [_rel(p) for p, v in loaded.items() if v is None]
    return Inputs(
        frozen=frozen_payload,
        grid=grid,
        articles=loaded[articles],
        summary=loaded[summary],
        latency=loaded[latency],
        sizes=sizes[0],
        sizes_source=sizes[1],
        missing=missing,
    )


def grid_point(report: Mapping[str, Any] | None, config: str, policy: str) -> dict[str, Any] | None:
    """The frozen config's dev300 operating point under its served policy."""
    if report is None:
        return None
    for entry in report.get("configs", []):
        if entry.get("name") == config:
            pol = entry.get("policies", {}).get(policy) or {}
            point = pol.get("operating_point") or pol.get("default")
            return dict(point) if point else None
    return None


def dev300_f05(report: Mapping[str, Any] | None, config: str, policy: str) -> dict[str, Any] | None:
    """F0.5 [95% CI] of a config's dev300 operating point: the grid's best cell or a follow-up cell."""
    if report is None:
        return None
    best = report.get("best") or {}
    if best.get("config") == config and best.get("policy") == policy:
        return dict(best["cis"]["f05_macro"])
    for follow_up in report.get("follow_ups", []):
        if follow_up.get("config") == config and follow_up.get("policy") == policy:
            return dict(follow_up["paired_bootstrap"]["follow_up"])
    point = grid_point(report, config, policy)
    return None if point is None else {"point": point["metrics"]["f05_macro"], "low": None, "high": None}


def article_row(articles: Mapping[str, Any] | None, model: str, config: str, scope: str, policy: str) -> Any:
    if articles is None:
        return None
    key = f"{model}/{config}/{scope}"
    return next((r for r in articles.get("rows", []) if r["key"] == key and r["policy"] == policy), None)


def latency_cell(latency: Mapping[str, Any] | None, cell: Cell) -> dict[str, Any] | None:
    if latency is None:
        return None
    return next(
        (c for c in latency.get("cells", []) if c["key"] == cell.key and c["backend"] == NATIVE), None
    )


# --- candidates ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool | None
    detail: str


@dataclass(slots=True)
class Cand:
    endpoint: Endpoint
    model: str
    config: str | None
    rule: str
    quality: float | None
    quality_label: str
    secondary: tuple[float, ...]
    checks: list[Check]
    latency: dict[str, Any] | None
    numbers: dict[str, Any] = field(default_factory=dict)
    size_bytes: int | None = None
    verdict: str = ""

    @property
    def eligible(self) -> bool:
        return self.quality is not None and all(c.passed is True for c in self.checks)

    @property
    def budget(self) -> str:
        return str((self.latency or {}).get("budget", {}).get("status", "unmeasured"))

    @property
    def utilisation(self) -> float | None:
        value = (self.latency or {}).get("budget", {}).get("utilisation")
        return float(value) if isinstance(value, int | float) else None

    def rank_key(self) -> tuple[float, ...]:
        latency = self.utilisation if self.utilisation is not None else float("inf")
        return (-(self.quality or 0.0), *(-x for x in self.secondary), latency)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.passed is not True]


def _inversion_checks(dev300_inv: float | None, row: Mapping[str, Any] | None) -> list[Check]:
    dev = Check(
        "dev300 inversion <= 2%",
        None if dev300_inv is None else dev300_inv <= INVERSION_CAP + 1e-12,
        "no dev300 operating point" if dev300_inv is None else f"{_pct(dev300_inv)} (n=300)",
    )
    if row is None:
        return [dev, Check("articles dev inversion cap", None, "no articles dev row")]
    art = row["articles"]
    low = art["cis"]["inversion_rate"]["low"]
    point = art["inversion_rate"]
    detail = f"{art['inversions']}/{art['directional_calls']} calls ({_pct(point)}), 95% CI low {_pct(low)}"
    if point > INVERSION_CAP and low <= INVERSION_CAP:
        detail += "; the point estimate alone would fail the cap"
    passed = low <= INVERSION_CAP + 1e-12 and not row.get("clear_failure")
    return [dev, Check("articles dev inversion cap", passed, detail)]


def _articles_numbers(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    art = row["articles"]
    return {
        "articlesF05": art["cis"]["f05_macro"],
        "articlesInversions": f"{art['inversions']}/{art['directional_calls']}",
        "articlesCoverage": art["coverage"],
        "articlesInferenceP50Ms": (row.get("latency") or {}).get("wall_ms_p50"),
    }


def signal_cands(inputs: Inputs) -> list[Cand]:
    out = []
    for model, endpoints in inputs.frozen["models"].items():
        entry = endpoints.get("signal")
        if entry is None:
            continue
        ev = entry["evidence"]["dev300"]
        config, policy = ev["config"], ev["policy"]
        point = grid_point(inputs.grid.get(("signal", model)), config, policy)
        lede = article_row(inputs.articles, model, config, "lede", policy)
        full = article_row(inputs.articles, model, config, "full", policy)
        dev_inv = ev["inversion_rate"]["point"] if "inversion_rate" in ev else None
        articles_f05 = lede["articles"]["f05_macro"] if lede else None
        out.append(
            Cand(
                endpoint="signal",
                model=model,
                config=config,
                rule=f"{policy} @ {ev['threshold']}",
                quality=ev["f05_macro"]["point"],
                quality_label="dev300 F0.5",
                secondary=(articles_f05 if articles_f05 is not None else -1.0,),
                checks=_inversion_checks(dev_inv, lede),
                latency=latency_cell(inputs.latency, Cell("signal", model, "lede", False)),
                numbers={
                    "dev300F05": ev["f05_macro"],
                    "dev300Inversion": dev_inv,
                    "dev300Coverage": ev.get("coverage", {}).get("point"),
                    "dev300Point": point,
                    **_articles_numbers(lede),
                    "articlesFullF05": full["articles"]["cis"]["f05_macro"] if full else None,
                    "promptVersion": entry.get("prompt_version"),
                },
            )
        )
    return out


def analyze_cands(inputs: Inputs) -> list[Cand]:
    out = []
    for model, endpoints in inputs.frozen["models"].items():
        entry = endpoints.get("analyze")
        if entry is None:
            continue
        ev = entry["evidence"]["dev300"]
        config = (
            ev["config"] if entry.get("field_order") != "summary-first" else f"{ev['config']}-summary-first"
        )
        report = inputs.grid.get(("analyze", model))
        point = grid_point(report, config, "ordinal")
        f05 = dev300_f05(report, config, "ordinal")
        row = article_row(inputs.articles, model, config, "full", "ordinal")
        dev_inv = point["metrics"]["inversion_rate"] if point else None
        articles_f05 = row["articles"]["f05_macro"] if row else None
        stream = latency_cell(inputs.latency, Cell("analyze", model, "full", True))
        out.append(
            Cand(
                endpoint="analyze",
                model=model,
                config=config,
                rule=f"ordinal |s| >= {entry['ordinal_t']}, {entry.get('field_order')}",
                quality=None if f05 is None else f05["point"],
                quality_label="dev300 F0.5",
                secondary=(articles_f05 if articles_f05 is not None else -1.0,),
                checks=_inversion_checks(dev_inv, row),
                latency=stream,
                numbers={
                    "dev300F05": f05,
                    "dev300Inversion": dev_inv,
                    "dev300Point": point,
                    **_articles_numbers(row),
                    "jsonLatency": latency_cell(inputs.latency, Cell("analyze", model, "full", False)),
                    "summaryQuality": _summary_row(inputs.summary, "analyze", model),
                    "promptVersion": entry.get("prompt_version"),
                },
            )
        )
    return out


def _summary_row(summary: Mapping[str, Any] | None, endpoint: str, model: str) -> dict[str, Any] | None:
    rows = ((summary or {}).get("splits", {}).get("dev") or {}).get("rows", [])
    return next((r for r in rows if r["endpoint"] == endpoint and r["model"] == model), None)


def summary_cands(inputs: Inputs) -> tuple[list[Cand], bool]:
    """Second value: whether the Opus judge scored every candidate (else key facts lead, provisionally)."""
    models = [m for m, eps in inputs.frozen["models"].items() if "summary" in eps]
    rows = {m: _summary_row(inputs.summary, "summary", m) for m in models}
    gate = float(((inputs.summary or {}).get("config") or {}).get("passGate", PASS_GATE))
    judged = bool(rows) and all(
        r is not None and r.get("judgeFaithfulness") is not None for r in rows.values()
    )
    out = []
    for model in models:
        row = rows[model]
        quality: float | None
        secondary: tuple[float, ...]
        if row is None:
            check = Check(f"deterministic pass >= {gate}", None, "no summary eval row")
            quality, secondary = None, ()
        else:
            rate = row["deterministicPassRate"]
            check = Check(
                f"deterministic pass >= {gate}", rate >= gate, f"{_pct(rate)} of {row['n']} articles"
            )
            if judged:
                quality = row["judgeFaithfulness"]
                secondary = (row.get("judgeCoverage") or 0.0, row["keyFactCoverage"])
            else:
                quality, secondary = row["keyFactCoverage"], (rate,)
        out.append(
            Cand(
                endpoint="summary",
                model=model,
                config=inputs.frozen["models"][model]["summary"].get("prompt"),
                rule="full scope, served 3-sentence / 60-word cap",
                quality=quality,
                quality_label="judge faithfulness" if judged else "key-fact coverage (judge pending)",
                secondary=secondary,
                checks=[check],
                latency=latency_cell(inputs.latency, Cell("summary", model, "full", True)),
                numbers={
                    "summary": row,
                    "jsonLatency": latency_cell(inputs.latency, Cell("summary", model, "full", False)),
                    "promptVersion": inputs.frozen["models"][model]["summary"].get("prompt_version"),
                },
            )
        )
    return out, judged


# --- choice ----------------------------------------------------------------------------------


@dataclass(slots=True)
class Choice:
    endpoint: Endpoint
    winner: Cand | None
    how: str
    reason: str
    ranked: list[Cand]
    provisional: bool


def _latency_text(c: Cand) -> str:
    checks = (c.latency or {}).get("budget", {}).get("checks") or []
    if not checks:
        return "latency not measured"
    return ", ".join(f"{k['metric']} {k['value']:,.0f} ms (limit {k['limit']:,.0f})" for k in checks)


def _quality_text(c: Cand) -> str:
    return f"{c.quality_label} {_f(c.quality)}"


def choose(endpoint: Endpoint, cands: Sequence[Cand], *, provisional: bool) -> Choice:
    ranked = sorted((c for c in cands if c.eligible), key=lambda c: (c.rank_key(), c.model))
    for c in cands:
        if not c.eligible:
            c.verdict = "excluded: " + "; ".join(f"{k.name} ({k.detail})" for k in c.failed())
    if not ranked:
        return Choice(endpoint, None, "none", "No candidate passes the accuracy filters.", [], True)
    inside = [c for c in ranked if c.budget == "met"]
    measured = [c for c in ranked if c.budget in ("met", "missed") and c.utilisation is not None]
    if inside:
        winner, how = inside[0], "inside-budget"
        reason = (
            f"Best {_quality_text(winner)} among the candidates inside every filter; {_latency_text(winner)}."
        )
    elif measured:
        points = [KneePoint(c.model, c.quality or 0.0, c.utilisation or 0.0) for c in measured]
        name = pareto_knee(points).name
        winner, how = next(c for c in measured if c.model == name), "knee"
        reason = (
            "No accuracy-eligible candidate meets its latency budget, so this is the Pareto knee of "
            f"{winner.quality_label} against budget utilisation: {_quality_text(winner)} at "
            f"{winner.utilisation:.2f}x the budget ({_latency_text(winner)})."
        )
    else:
        winner, how = ranked[0], "quality-only"
        reason = f"Best {_quality_text(winner)}; latency is not measured yet, so the budget is unchecked."
    for c in ranked:
        if c is winner:
            c.verdict = f"won ({how})"
        elif c.budget == "missed" and how == "inside-budget":
            c.verdict = f"misses the budget: {_latency_text(c)}"
        elif how == "knee":
            c.verdict = f"not the knee ({_quality_text(c)}, {c.utilisation or 0:.2f}x the budget)"
        elif c.budget == "unmeasured" and how == "inside-budget":
            c.verdict = "latency not measured, so it cannot be confirmed inside the budget"
        else:
            c.verdict = f"ranks below the winner ({_quality_text(c)} vs {_f(winner.quality)})"
    return Choice(endpoint, winner, how, reason, ranked, provisional or how == "quality-only")


# --- bundle (R8) -----------------------------------------------------------------------------


def bundle_prefs(choice: Choice) -> list[str]:
    """The winner, then the other accuracy-eligible candidates in rank order."""
    if choice.winner is None:
        return []
    return list(dict.fromkeys([choice.winner.model, *(c.model for c in choice.ranked)]))


def bundle_plan(
    prefs: Mapping[Endpoint, Sequence[str]], sizes: Mapping[str, int] | None, cap: int = SIZE_CAP_BYTES
) -> dict[str, Any]:
    """Per-endpoint preference lists (winner first) -> the fitting combination closest to the winners."""
    if sizes is None:
        return {"status": "sizes unknown", "fits": False}
    order = [e for e in ENDPOINTS if prefs.get(e)]
    options = [list(enumerate(prefs[e])) for e in order]
    best: tuple[tuple[int, tuple[int, ...], int], dict[Endpoint, str]] | None = None
    unknown = sorted({m for e in order for m in prefs[e]} - set(sizes))
    for combo in itertools.product(*options):
        models = {m for _, m in combo}
        if models & set(unknown):
            continue
        total = sum(sizes[m] for m in models)
        if total > cap:
            continue
        ranks = tuple(i for i, _ in combo)
        key = (sum(ranks), ranks, total)
        if best is None or key < best[0]:
            best = (key, {e: m for e, (_, m) in zip(order, combo, strict=True)})
    winners = {e: prefs[e][0] for e in order}
    winners_total = sum(sizes.get(m, 0) for m in set(winners.values()))
    if best is None:
        return {"status": "nothing fits", "fits": False, "winnersBytes": winners_total, "capBytes": cap}
    picks = best[1]
    bundled = sorted(set(picks.values()))
    return {
        "status": "winners" if picks == winners else "swapped",
        "fits": True,
        "defaults": picks,
        "models": bundled,
        "totalBytes": sum(sizes[m] for m in bundled),
        "winnersBytes": winners_total,
        "capBytes": cap,
        "swaps": {e: {"winner": winners[e], "bundled": picks[e]} for e in order if picks[e] != winners[e]},
        "unknownSizes": unknown,
    }


# --- writing ---------------------------------------------------------------------------------


def write_defaults(text: str, defaults: Mapping[Endpoint, str]) -> str:
    """Rewrites only the [defaults] lines; the result must parse and every default must serve its endpoint."""
    lines = text.split("\n")
    start = lines.index("[defaults]")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    block = lines[start + 1 : end]
    for endpoint, model in defaults.items():
        line = f"{endpoint} = {json.dumps(model)}"
        keys = [ln.split("=", 1)[0].strip() if "=" in ln else None for ln in block]
        if endpoint in keys:
            block[keys.index(endpoint)] = line
        else:
            last = max((i for i, k in enumerate(keys) if k), default=-1)
            block.insert(last + 1, line)
    out = "\n".join([*lines[: start + 1], *block, *lines[end:]])
    specs, parsed = parse_models_toml(out)
    Registry(specs, parsed)
    for endpoint, model in defaults.items():
        if parsed.get(endpoint) != model or not specs[model].serves(endpoint):
            raise SelectError(f"[defaults] {endpoint} = {model} did not stick or is not allowed")
    return out


def write_env(text: str, values: Mapping[str, str]) -> str:
    lines = text.rstrip("\n").split("\n")
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in values:
            lines[i] = f"{key}={values[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in values.items() if k not in seen]
    return "\n".join(lines) + "\n"


def bundle_env(plan: Mapping[str, Any]) -> dict[str, str]:
    models = ",".join(plan["models"])
    defaults = plan["defaults"]
    return {
        "VT_PULL_MODELS": models,
        "VT_MODELS": models,
        "VT_DEFAULT_MODEL_SIGNAL": defaults["signal"],
        "VT_DEFAULT_MODEL_SUMMARY": defaults["summary"],
        "VT_DEFAULT_MODEL_ANALYZE": defaults["analyze"],
    }


# --- report ----------------------------------------------------------------------------------


def _notes(inputs: Inputs, judged: bool) -> list[str]:
    notes = [f"missing input: {m}" for m in inputs.missing]
    lat = inputs.latency
    if lat is None:
        notes.append("latency.json is missing: budgets are unchecked and every winner is quality-only")
    else:
        if not lat.get("clean"):
            notes.append("the latency run was not clean (preflight failed): budget marks are indicative")
        if lat.get("missing"):
            notes.append(
                f"latency cells unmeasured: {len(lat['missing'])} ({', '.join(lat['missing'][:4])}...)"
            )
        if lat.get("partial"):
            notes.append(f"latency cells below their sample targets: {len(lat['partial'])}")
    if not judged:
        notes.append("the Opus judge has not scored the summaries: /summary is ranked by key-fact coverage")
    if inputs.sizes is None:
        notes.append("model sizes unavailable (Ollama down, no preflight record): the bundle is not planned")
    return notes


def build_selection(inputs: Inputs, cap: int = SIZE_CAP_BYTES) -> dict[str, Any]:
    summary, judged = summary_cands(inputs)
    pools: dict[Endpoint, list[Cand]] = {
        "signal": signal_cands(inputs),
        "summary": summary,
        "analyze": analyze_cands(inputs),
    }
    lat = inputs.latency or {}
    latency_provisional = (
        not lat or not lat.get("clean") or bool(lat.get("missing")) or bool(lat.get("partial"))
    )
    choices: dict[Endpoint, Choice] = {}
    for endpoint in ENDPOINTS:
        for c in pools[endpoint]:
            c.size_bytes = (inputs.sizes or {}).get(c.model)
        provisional = latency_provisional or (endpoint == "summary" and not judged)
        choices[endpoint] = choose(endpoint, pools[endpoint], provisional=provisional)
    prefs = {e: bundle_prefs(choices[e]) for e in ENDPOINTS if choices[e].winner is not None}
    if len(prefs) == len(ENDPOINTS):
        plan = bundle_plan(prefs, inputs.sizes, cap)
    else:
        plan = {"status": "no winner for every endpoint", "fits": False}
    notes = _notes(inputs, judged)
    return {
        "version": 1,
        "generatedAt": _now(),
        "generatedBy": "uv run python -m veritrade_evals select",
        "provisional": any(ch.provisional for ch in choices.values()),
        "notes": notes,
        "rules": RULES,
        "inversionCap": INVERSION_CAP,
        "passGate": PASS_GATE,
        "latency": {
            "session": lat.get("session"),
            "clean": lat.get("clean"),
            "complete": lat.get("complete"),
            "budgets": lat.get("budgets"),
        },
        "sizesSource": inputs.sizes_source,
        "endpoints": {
            e: {
                "winner": None if ch.winner is None else ch.winner.model,
                "how": ch.how,
                "reason": ch.reason,
                "provisional": ch.provisional,
                "candidates": [_cand_json(c) for c in pools[e]],
            }
            for e, ch in choices.items()
        },
        "bundle": plan,
    }


def _cand_json(c: Cand) -> dict[str, Any]:
    return {
        "model": c.model,
        "config": c.config,
        "rule": c.rule,
        "quality": {"label": c.quality_label, "value": c.quality, "secondary": list(c.secondary)},
        "checks": [{"name": k.name, "passed": k.passed, "detail": k.detail} for k in c.checks],
        "eligible": c.eligible,
        "budget": c.budget,
        "utilisation": c.utilisation,
        "latencyChecks": (c.latency or {}).get("budget", {}).get("checks"),
        "latencyWarmN": (c.latency or {}).get("warm", {}).get("n"),
        "sizeBytes": c.size_bytes,
        "numbers": c.numbers,
        "verdict": c.verdict,
    }


def frontend_selection(selection: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """BenchmarksV1.selection: model and reason per endpoint."""
    out = {}
    for endpoint, entry in selection["endpoints"].items():
        prefix = "Provisional. " if entry["provisional"] else ""
        out[endpoint] = {"model": entry["winner"] or "-", "reason": prefix + entry["reason"]}
    return out


def _gb(size: int | None) -> str:
    return "-" if size is None else f"{size / 1e9:.2f}"


def _lat_cell(checks: Sequence[Mapping[str, Any]] | None) -> str:
    if not checks:
        return "-"
    return " / ".join(f"{k['metric']} {k['value']:,.0f}" for k in checks if k.get("value") is not None) or "-"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _accuracy_rows(cands: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows = []
    for c in cands:
        n, checks = c["numbers"], {k["name"]: k for k in c["checks"]}
        art = checks.get("articles dev inversion cap", {})
        rows.append(
            [
                f"`{c['model']}`",
                f"{c['config']}, {c['rule']}",
                _ci(n.get("dev300F05")),
                _pct(n.get("dev300Inversion")),
                _ci(n.get("articlesF05")),
                str(art.get("detail", "-")),
                _lat_cell(c["latencyChecks"]),
                c["budget"],
                _gb(c["sizeBytes"]),
                c["verdict"],
            ]
        )
    return rows


def _summary_rows(cands: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows = []
    for c in cands:
        s = c["numbers"].get("summary") or {}
        rows.append(
            [
                f"`{c['model']}`",
                _pct(s.get("deterministicPassRate")),
                _pct(s.get("keyFactCoverage")),
                _f(s.get("judgeFaithfulness"), 2),
                _f(s.get("judgeCoverage"), 2),
                _pct(s.get("hallucinationRate")),
                _lat_cell(c["latencyChecks"]),
                c["budget"],
                _gb(c["sizeBytes"]),
                c["verdict"],
            ]
        )
    return rows


ACCURACY_HEADER = (
    "model", "config, rule", "dev300 F0.5 [95% CI]", "dev300 inversion", "articles dev F0.5 [95% CI]",
    "articles dev inversions", "latency (ms)", "budget", "size GB", "verdict",
)  # fmt: skip
SUMMARY_HEADER = (
    "model", "pass", "key facts", "judge faithful", "judge cover", "halluc.", "latency (ms)", "budget",
    "size GB", "verdict",
)  # fmt: skip


def render(selection: Mapping[str, Any]) -> str:
    lines = [
        "# Model selection",
        "",
        f"Generated by `{selection['generatedBy']}` from committed results; numbers only.",
        "",
    ]
    if selection["provisional"]:
        lines += ["> **Provisional.** " + " ".join(f"{n[0].upper()}{n[1:]}." for n in selection["notes"]), ""]
    elif selection["notes"]:
        lines += [*(f"- {n}" for n in selection["notes"]), ""]
    lines += ["## Rules", "", *(f"- **{k}**: {v}" for k, v in selection["rules"].items()), ""]
    for endpoint in ("signal", "summary", "analyze"):
        entry = selection["endpoints"][endpoint]
        winner = f"`{entry['winner']}`" if entry["winner"] else "no winner"
        lines += [f"## /{endpoint}: {winner} ({entry['how']})", "", entry["reason"], ""]
        if endpoint == "summary":
            lines += _table(SUMMARY_HEADER, _summary_rows(entry["candidates"]))
        else:
            lines += _table(ACCURACY_HEADER, _accuracy_rows(entry["candidates"]))
        lines.append("")
    plan = selection["bundle"]
    lines += ["## Bundled models (R8)", ""]
    if plan.get("fits"):
        lines += [
            f"{plan['status']}: {', '.join(f'`{m}`' for m in plan['models'])}, "
            f"{_gb(plan['totalBytes'])} GB of model files (cap {_gb(plan['capBytes'])} GB; the winners "
            f"alone {_gb(plan['winnersBytes'])} GB). "
            f"Sizes from {selection['sizesSource']}.",
            "",
            *(
                f"- /{e}: winner `{s['winner']}` does not fit, bundled `{s['bundled']}` instead"
                for e, s in plan.get("swaps", {}).items()
            ),
            "",
        ]
    else:
        lines += [f"Not planned: {plan.get('status')}.", ""]
    return "\n".join(lines)


def write_outputs(selection: Mapping[str, Any], out_dir: Path = SELECTION_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / SELECTION_JSON.name, out_dir / SELECTION_MD.name
    json_path.write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    md_path.write_text(render(selection), encoding="utf-8")
    return [json_path, md_path]


def apply_write(selection: Mapping[str, Any], models_toml: Path, env_path: Path) -> list[Path]:
    if selection["provisional"]:
        raise SelectError(
            "refusing --write: the selection is provisional (" + "; ".join(selection["notes"]) + ")"
        )
    winners = {e: v["winner"] for e, v in selection["endpoints"].items()}
    if any(w is None for w in winners.values()):
        raise SelectError("refusing --write: an endpoint has no winner")
    plan = selection["bundle"]
    if not plan.get("fits"):
        raise SelectError(f"refusing --write: no bundle fits ({plan.get('status')})")
    models_toml.write_text(write_defaults(models_toml.read_text(encoding="utf-8"), winners), encoding="utf-8")
    env_path.write_text(write_env(env_path.read_text(encoding="utf-8"), bundle_env(plan)), encoding="utf-8")
    return [models_toml, env_path]


# --- CLI -------------------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> int:
    inputs = load_inputs(sizes=model_sizes(args.base_url))
    selection = build_selection(inputs, cap=int(args.size_cap_gb * 1e9))
    for path in write_outputs(selection, args.out_dir):
        print(f"wrote {_rel(path)}")
    for endpoint, entry in selection["endpoints"].items():
        flag = " (provisional)" if entry["provisional"] else ""
        print(f"/{endpoint}: {entry['winner']} [{entry['how']}]{flag}")
    plan = selection["bundle"]
    print(f"bundle: {plan.get('status')} {plan.get('models', '')}")
    for note in selection["notes"]:
        print(f"note: {note}", file=sys.stderr)
    if args.write:
        for path in apply_write(selection, MODELS_TOML, BUNDLED_ENV):
            print(f"updated {_rel(path)}")
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `select` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser("select", help="pick per-endpoint winners")
    parser.add_argument("--write", action="store_true", help="update models.toml [defaults] and bundled.env")
    parser.add_argument("--base-url", default=None, help="Ollama URL for model sizes (default from settings)")
    parser.add_argument("--size-cap-gb", type=float, default=SIZE_CAP_BYTES / 1e9, help="R8 bundle cap")
    parser.add_argument("--out-dir", type=Path, default=SELECTION_DIR)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        return cmd_select(args)
    except SelectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.select", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["select", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
