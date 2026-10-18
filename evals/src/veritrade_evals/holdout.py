"""One-time holdout of the frozen rules: `uv run python -m veritrade_evals holdout {plan,run,analyze}`.
evals/results/holdout_runs.json refuses a second run of the same frozen config; `--dry-run` drives the
same code on a few dev300 rows into a scratch folder and never reads a holdout split or the guard."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from veritrade.config import get_settings
from veritrade.errors import VtError
from veritrade.extract.types import Scope
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Endpoint, Sentiment
from veritrade.models_registry import allowlist
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals import summary_eval as se
from veritrade_evals.analysis import (
    EvalPolicy,
    ci_dict,
    counts,
    evaluate_policy,
    latency,
    metrics_dict,
    rounded,
)
from veritrade_evals.articles import ArticlePaths, load_articles
from veritrade_evals.finalists import extract_all, inputs_for
from veritrade_evals.grid import cell_name, write_manifest
from veritrade_evals.metrics import headline_cis, signal_metrics
from veritrade_evals.paths import PREDS_DIR, REPO_ROOT, RESULTS_DIR, SPLITS_DIR
from veritrade_evals.predcache import canonical_json, model_slug
from veritrade_evals.runner import (
    Item,
    RunAborted,
    RunSpec,
    Variant,
    make_resident,
    model_digest,
    run_cell,
    serving_spec,
)
from veritrade_evals.select import ARTICLES_JSON, GRID_DIR, SELECTION_JSON, dev300_f05
from veritrade_evals.splits import SplitName, load_split

HOLDOUT_DIR = RESULTS_DIR / "holdout"
HOLDOUT_JSON = HOLDOUT_DIR / "holdout.json"
HOLDOUT_MD = HOLDOUT_DIR / "holdout.md"
MANIFEST = HOLDOUT_DIR / "manifest.json"
GUARD = RESULTS_DIR / "holdout_runs.json"
FROZEN_JSON = RESULTS_DIR / "frozen.json"
CEILING = "qwen3.5:9b-q4_K_M"
ARTICLES = "articles_test"
TWEETS: SplitName = "tfns_val"
CROSS: tuple[SplitName, ...] = ("fpb_allagree", "finentity_single")
DRY_SPLIT: SplitName = "dev300"
DRY_LIMIT = 6
N_BOOT = 1000
DEFAULT_CONCURRENCY = 1
# Display names for the benchmarks page: (dataset, split).
DATASETS: dict[str, tuple[str, str]] = {
    ARTICLES: ("articles", "test"),
    "tfns_val": ("twitter-fns", "validation"),
    "fpb_allagree": ("fpb", "allagree"),
    "finentity_single": ("finentity", "single-label"),
    "dev300": ("twitter-fns", "dev300"),
}
# Per-call ms when no measurement exists; FinEntity paragraphs run longer than tweets.
FALLBACK_MS = 1000.0
LENGTH_FACTOR = {"finentity_single": 1.5}
SUMMARY_CALL_MS = 3000.0

Role = Literal["finalist", "ceiling"]


class HoldoutError(RuntimeError):
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- frozen rules ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rule:
    """What one endpoint serves for one model after the freeze, with its threshold."""

    endpoint: Endpoint
    model: str
    run: RunSpec
    policy: EvalPolicy
    threshold: float
    role: Role
    dev: Mapping[str, float]


def _signal_variant(kind: str, schema: str | None) -> Variant:
    if kind == "decision":
        return "decision"
    if schema not in ("label", "ordinal"):
        raise HoldoutError(f"generative /signal schema {schema!r} is not label or ordinal")
    return cast(Variant, schema)


def rules_from_frozen(frozen: Mapping[str, Any], registry: PromptRegistry) -> list[Rule]:
    """Every frozen /signal and /analyze rule; a prompt that drifted since the freeze fails loudly."""
    roster = allowlist()
    rules: list[Rule] = []
    for model, endpoints in frozen["models"].items():
        spec = roster[model]
        for endpoint in ("signal", "analyze"):
            entry = endpoints.get(endpoint)
            if entry is None:
                continue
            ev = entry["evidence"]
            prompt = registry.get(endpoint, model, kind=spec.kind, version=entry["prompt"])
            if prompt.version != entry["prompt_version"]:
                frozen_version = entry["prompt_version"]
                raise HoldoutError(f"{model} /{endpoint}: prompt {prompt.version} != frozen {frozen_version}")
            dev = {f"articles_dev/{a['scope']}": a["f05_macro"]["point"] for a in ev["articles_dev"]}
            if endpoint == "signal":
                variant = _signal_variant(spec.kind, spec.signal_schema)
                policy = cast(EvalPolicy, ev["dev300"]["policy"])
                threshold = float(ev["dev300"]["threshold"])
                dev["dev300"] = ev["dev300"]["f05_macro"]["point"]
            else:
                variant = cast(Variant, entry["field_order"])
                policy, threshold = "ordinal", float(entry["ordinal_t"])
                grid_report = read_json(GRID_DIR / f"analyze__{model_slug(model)}.json")
                f05 = dev300_f05(grid_report, cell_name(entry["prompt"], variant), "ordinal")
                if f05 is not None:
                    dev["dev300"] = f05["point"]
            run = RunSpec(cell_name(entry["prompt"], variant), serving_spec(spec, variant), prompt, variant)
            rules.append(Rule(endpoint, model, run, policy, threshold, "finalist", dev))
    return rules


def ceiling_rule(registry: PromptRegistry, model: str = CEILING) -> Rule:
    spec = allowlist()[model]
    prompt = registry.get("analyze", model, kind=spec.kind)
    variant: Variant = spec.field_order or "sentiment-first"
    run = RunSpec(cell_name(prompt.version_id, variant), serving_spec(spec, variant), prompt, variant)
    return Rule("analyze", model, run, "ordinal", float(spec.ordinal_t), "ceiling", {})


# --- plan ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HCell:
    rule: Rule
    split: str
    scope: Scope | None = None

    @property
    def key(self) -> str:
        scope = f"/{self.scope}" if self.scope else ""
        return f"{self.split}/{self.rule.endpoint}/{self.rule.model}/{self.rule.run.name}{scope}"

    @property
    def model(self) -> str:
        return self.rule.model


def plan_cells(rules: Sequence[Rule], ceiling: Rule, winners: Mapping[str, str]) -> list[HCell]:
    """Finalists on the test articles and tweets; winners and the ceiling on the cross-domain sets."""
    cells: list[HCell] = []
    for rule in rules:
        scopes: tuple[Scope, ...] = ("lede", "full") if rule.endpoint == "signal" else ("full",)
        cells += [HCell(rule, ARTICLES, s) for s in scopes]
        cells.append(HCell(rule, TWEETS))
        if winners.get(rule.endpoint) == rule.model:
            cells += [HCell(rule, split) for split in CROSS]
    cells += [HCell(ceiling, ARTICLES, "full"), *(HCell(ceiling, split) for split in CROSS)]
    order = {m: i for i, m in enumerate(dict.fromkeys(c.model for c in cells))}
    return sorted(cells, key=lambda c: order[c.model])


def summary_pairs(winners: Mapping[str, str]) -> list[tuple[str, str]]:
    return [(e, winners[e]) for e in ("summary", "analyze") if winners.get(e)]


def per_call_ms(cell: HCell) -> float | None:
    """Measured p50 per call of the same rule: the articles dev run for articles, else the dev300 grid."""
    rule = cell.rule
    if cell.split == ARTICLES:
        rows = (read_json(ARTICLES_JSON) or {}).get("rows", [])
        key = f"{rule.model}/{rule.run.name}/{cell.scope}"
        value = next(((r.get("latency") or {}).get("wall_ms_p50") for r in rows if r["key"] == key), None)
    else:
        report = read_json(GRID_DIR / f"{rule.endpoint}__{model_slug(rule.model)}.json") or {}
        entry: dict[str, Any] = next(
            (c for c in report.get("configs", []) if c.get("name") == rule.run.name), {}
        )
        value = (entry.get("latency") or {}).get("wall_ms_p50")
    return float(value) if isinstance(value, int | float) else None


def estimate(
    cells: Sequence[HCell], sizes: Mapping[str, int], summaries: int, ceiling_ref: str = "qwen3.5:4b-q4_K_M"
) -> dict[str, Any]:
    """Dev-run p50 per call times the rows; the unmeasured ceiling borrows twice qwen3.5:4b's time."""
    known = {(c.model, c.split == ARTICLES): per_call_ms(c) for c in cells if c.rule.role == "finalist"}
    rows = []
    by_model: dict[str, float] = {}
    for cell in cells:
        per_call: float | None = None
        if cell.rule.role == "finalist":
            per_call = per_call_ms(cell)
        elif (ref := known.get((ceiling_ref, cell.split == ARTICLES))) is not None:
            per_call = 2 * ref
        per_call = (per_call or FALLBACK_MS) * LENGTH_FACTOR.get(cell.split, 1.0)
        n = sizes.get(cell.split, 0)
        by_model[cell.model] = by_model.get(cell.model, 0.0) + n * per_call / 1000
        rows.append(
            {"key": cell.key, "n": n, "perCallMs": round(per_call), "seconds": round(n * per_call / 1000)}
        )
    summary_s = summaries * sizes.get(ARTICLES, 0) * SUMMARY_CALL_MS / 1000
    # About 15 s per model for unloading the previous one and a cold load.
    total = sum(by_model.values()) + summary_s + 15 * len(by_model)
    return {
        "cells": rows,
        "byModelSeconds": {k: round(v) for k, v in by_model.items()},
        "totalSeconds": total,
    }


# --- inputs ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gold:
    ids: tuple[str, ...]
    labels: tuple[Sentiment, ...]


def balanced_head(items: Sequence[Item], n: int) -> list[Item]:
    """The first rows of each class in split order, about n in all, so a small slice still has every class."""
    per_class = max(1, n // 3)
    seen: Counter[str] = Counter()
    out = []
    for item in items:
        if seen[item.label] < per_class:
            seen[item.label] += 1
            out.append(item)
    return out


class Inputs:
    """Model inputs per (split, scope, token budget), loaded once; only ids and labels leave this class."""

    def __init__(self, *, limit: int = 0, paths: ArticlePaths | None = None) -> None:
        self.limit = limit
        self.paths = paths or ArticlePaths()
        self._splits: dict[str, list[Item]] = {}
        self._extracted: list[Any] | None = None

    def _split(self, name: str) -> list[Item]:
        if name not in self._splits:
            examples: list[Item] = list(load_split(cast(SplitName, name)).examples)
            self._splits[name] = balanced_head(examples, self.limit) if self.limit else examples
        return self._splits[name]

    def items(self, cell: HCell) -> tuple[list[Item], dict[str, Any] | None]:
        if cell.split != ARTICLES:
            return self._split(cell.split), None
        if self._extracted is None:
            self._extracted = extract_all(load_articles("test", paths=self.paths), self.paths)
        scope = cell.scope or "full"
        items, identity = inputs_for(self._extracted, scope, cell.rule.run.spec.max_input_tokens)
        return list(items), {**identity, "articles": "test"}

    def gold(self, split: str) -> Gold:
        if split == ARTICLES:
            examples = load_articles("test", paths=self.paths)
            return Gold(tuple(e.id for e in examples), tuple(e.label for e in examples))
        rows = self._split(split)
        return Gold(tuple(e.id for e in rows), tuple(cast(Sentiment, e.label) for e in rows))


# --- run -------------------------------------------------------------------------------------


def read_manifest(path: Path) -> dict[str, Any]:
    return read_json(path) or {"cells": {}}


async def run_cells(
    client: OllamaClient,
    cells: Sequence[HCell],
    inputs: Inputs,
    *,
    preds_root: Path,
    manifest_path: Path,
    concurrency: int,
    shared: bool,
    log: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Model by model through the eval runner and its prediction cache; resumable cell by cell."""
    manifest = read_manifest(manifest_path)
    manifest["ollama_version"] = await client.version()
    manifest["backend"] = get_settings().ollama_backend_label
    for model, group in itertools.groupby(cells, key=lambda c: c.model):
        digest = await model_digest(client, model)
        prepared = [(cell, *inputs.items(cell)) for cell in group]
        pending = [
            cell
            for cell, items, identity in prepared
            if not {i.id for i in items}
            <= cell.rule.run.cache(cell.split, digest, preds_root, identity).done_ids()
        ]
        if pending:
            print(f"== {model}: {'' if shared else 'unloading others and '}warming up", file=log, flush=True)
            await make_resident(client, pending[0].rule.run, shared=shared, log=log)
        for cell, items, identity in prepared:
            run = cell.rule.run
            stats = await run_cell(
                client,
                run,
                items,
                split=cell.split,
                model_digest=digest,
                concurrency=concurrency,
                root=preds_root,
                inputs=identity,
                log=log,
            )
            records = run.cache(cell.split, digest, preds_root, identity).records()
            manifest["cells"][cell.key] = {
                "split": cell.split,
                "endpoint": cell.rule.endpoint,
                "model": model,
                "config": run.name,
                "scope": cell.scope,
                "role": cell.rule.role,
                "prompt_version": run.prompt.version,
                "run_config": json.loads(canonical_json(run.run_config(cell.split, identity))),
                "model_digest": digest,
                "cache_key": stats.key,
                "inputs": len(items),
                "done": len(records),
                "errors": sum(1 for r in records.values() if r.get("error") is not None),
                "concurrency": concurrency,
                "updated_at": _now(),
            }
            write_manifest(manifest, manifest_path)
    return manifest


async def run_summaries(
    client: OllamaClient, pairs: Sequence[tuple[str, str]], log: TextIO
) -> dict[str, Any]:
    """Test-article summaries through summary_eval: generate, score, then blind packets for the Opus judge."""
    paths = se.SummaryPaths()
    articles = se.load_eval_articles("test", ArticlePaths())
    typed = [se.parse_pair(f"{e}/{m}") for e, m in pairs]
    for pair in typed:
        run = se.GenRun.resolve(pair)
        await se.generate(
            client,
            run,
            articles,
            split="test",
            digest=await model_digest(client, pair.model),
            paths=paths,
            log=log,
        )
    summaries = {pair: se.load_summaries(paths, "test", pair) for pair in typed}
    rows = await se.score_split(client, articles, summaries)
    se.write_scores(paths, "test", rows, {p.key for p in typed})
    written = se.prepare_packets("test", articles, summaries, paths.judge)
    return {
        "pairs": [p.key for p in typed],
        "packets": len(written),
        "packetList": _rel(paths.judge.packet_list("test")),
    }


# --- guard -----------------------------------------------------------------------------------


def read_guard(path: Path = GUARD) -> dict[str, Any]:
    guard = read_json(path)
    return guard if guard is not None else {"runs": {}}


def guard_mode(guard: Mapping[str, Any], frozen_sha: str, *, force: bool) -> str:
    """fresh, resume (an interrupted run continues from its caches) or forced; a finished run refuses."""
    entry = guard["runs"].get(frozen_sha)
    if entry is None:
        return "fresh"
    if entry["status"] != "complete":
        return "resume"
    if not force:
        raise HoldoutError(
            f"the holdout already ran for frozen config {frozen_sha[:12]} "
            f"(completed {entry['completedAt']}); "
            "it runs once per frozen config. Re-freeze for a new config, or pass --force --reason '...' to "
            "record a deliberate second run."
        )
    return "forced"


def guard_start(
    guard: dict[str, Any],
    frozen_sha: str,
    *,
    mode: str,
    selection_sha: str | None,
    cells: int,
    reason: str | None,
    log: TextIO | None = None,
) -> dict[str, Any]:
    entry = guard["runs"].setdefault(
        frozen_sha,
        {"status": "started", "startedAt": _now(), "completedAt": None, "resumed": [], "forced": []},
    )
    entry.update({"selectionSha256": selection_sha, "cells": cells, "gitSha": git_sha()})
    if mode == "resume":
        entry["resumed"].append(_now())
    if mode == "forced":
        banner = "!" * 100
        print(
            f"{banner}\nHOLDOUT FORCED: frozen config {frozen_sha[:12]} already completed on "
            f"{entry['completedAt']}. This second run is logged in {_rel(GUARD)} and must be disclosed.\n"
            f"reason: {reason or '(none given)'}\n{banner}",
            file=log or sys.stderr,
            flush=True,
        )
        entry["forced"].append({"at": _now(), "reason": reason, "previousCompletedAt": entry["completedAt"]})
        entry["status"] = "started"
    return guard


def guard_complete(guard: dict[str, Any], frozen_sha: str) -> dict[str, Any]:
    entry = guard["runs"][frozen_sha]
    entry["status"] = "complete"
    entry["completedAt"] = _now()
    return guard


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


# --- analyze ---------------------------------------------------------------------------------


def score_cell(
    cell: HCell, records: Mapping[str, Mapping[str, Any]], gold: Gold, n_boot: int
) -> dict[str, Any]:
    """The frozen rule at its frozen threshold; natural class priors, so no weights."""
    aligned = tuple(records.get(i) for i in gold.ids)
    rule = cell.rule
    ev = evaluate_policy(rule.policy, aligned, gold.labels, None, ordinal_t=rule.run.spec.ordinal_t)
    pred, abstained = ev.served(rule.threshold)
    metrics = signal_metrics(gold.labels, pred, abstained)
    cis = ci_dict(headline_cis(gold.labels, pred, abstained, n_boot=n_boot))
    dataset, split = DATASETS[cell.split]
    dev_key = f"articles_dev/{cell.scope}" if cell.split == ARTICLES else "dev300"
    dev = rule.dev.get(dev_key)
    return {
        "key": cell.key,
        "split": cell.split,
        "dataset": dataset,
        "datasetSplit": split,
        "endpoint": rule.endpoint,
        "model": rule.model,
        "kind": rule.run.spec.kind,
        "role": rule.role,
        "config": rule.run.name,
        "policy": rule.policy,
        "threshold": rule.threshold,
        "scope": cell.scope,
        "fieldOrder": rule.run.spec.field_order if rule.endpoint == "analyze" else None,
        "promptVersion": rule.run.prompt.version,
        "n": len(gold.ids),
        "gold": dict(Counter(gold.labels)),
        "metrics": metrics_dict(metrics),
        "cis": cis,
        "counts": counts(aligned),
        "latency": latency(aligned),
        "dev": {"reference": dev_key if dev is not None else None, "f05Macro": dev},
        "gap": None if dev is None else metrics.f05_macro - dev,
    }


def analyze_cells(
    cells: Sequence[HCell],
    inputs: Inputs,
    manifest: Mapping[str, Any],
    preds_root: Path,
    n_boot: int,
    log: TextIO,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, missing = [], []
    for cell in cells:
        entry = manifest.get("cells", {}).get(cell.key)
        if entry is None:
            missing.append(cell.key)
            continue
        _, identity = inputs.items(cell)
        current = json.loads(canonical_json(cell.rule.run.run_config(cell.split, identity)))
        if entry["run_config"] != current:
            print(f"skip {cell.key}: config or inputs changed since the run", file=log)
            missing.append(cell.key)
            continue
        records = cell.rule.run.cache(cell.split, entry["model_digest"], preds_root, identity).records()
        rows.append(score_cell(cell, records, inputs.gold(cell.split), n_boot))
    return rows, missing


def summary_block(pairs: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
    """Deterministic checks now, judge columns once the Opus judgements are ingested."""
    if not pairs:
        return None
    paths = se.SummaryPaths()
    scores = se.read_scores(paths, "test")
    if not scores:
        return {"status": "not generated", "rows": []}
    ids = _test_ids()
    judged = se.read_judged(paths.judge, "test")
    rows = [se.aggregate(se.parse_pair(f"{e}/{m}"), scores, ids, judged) for e, m in pairs]
    return {"status": "judged" if judged else "awaiting the Opus judge", "n": len(ids), "rows": rows}


def build_holdout(
    rows: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    *,
    frozen_sha: str,
    selection_sha: str | None,
    guard_entry: Mapping[str, Any] | None,
    summaries: Mapping[str, Any] | None,
    n_boot: int,
    dry_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = rounded(
        {
            "version": 1,
            "generatedAt": _now(),
            "generatedBy": "uv run python -m veritrade_evals holdout",
            "dryRun": dry_run,
            "frozenSha256": frozen_sha,
            "selectionSha256": selection_sha,
            "guard": guard_entry,
            "nBoot": n_boot,
            "rules": "frozen thresholds and prompts, no re-tuning; natural class priors (unweighted); 95% "
            "stratified bootstrap CIs; gap = holdout F0.5 minus the dev F0.5 of the same rule (articles: the "
            "articles dev set at the same scope; tweets and cross-domain sets: dev300)",
            "rows": list(rows),
            "missing": list(missing),
            "summaries": summaries,
        }
    )
    return report


SPLIT_TITLES = {
    ARTICLES: "Test articles (Opus-labelled, balanced)",
    "tfns_val": "Twitter FNS validation (in-domain tweets)",
    "fpb_allagree": "Financial PhraseBank AllAgree (cross-domain sentences)",
    "finentity_single": "FinEntity single-label (cross-domain paragraphs)",
    "dev300": "dev300 slice (dry run only)",
}


def _ci(ci: Mapping[str, Any] | None) -> str:
    if not ci:
        return "-"
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def _row(r: Mapping[str, Any]) -> str:
    m = r["metrics"]
    pc = m["per_class"]
    rule = f"{r['config']}, {r['policy']} @ {r['threshold']:g}"
    dev = r["dev"]["f05Macro"]
    cells = [
        f"/{r['endpoint']}",
        f"`{r['model']}`" + (" (ceiling)" if r["role"] == "ceiling" else ""),
        rule,
        r["scope"] or "-",
        _ci(r["cis"].get("f05_macro")),
        f"{pc['bullish']['precision']:.2f} / {pc['bullish']['recall']:.2f}",
        f"{pc['bearish']['precision']:.2f} / {pc['bearish']['recall']:.2f}",
        f"{100 * m['inversion_rate']:.1f}%",
        f"{100 * m['coverage']:.0f}%",
        f"{r['counts'].get('errors', 0)}",
        "-" if dev is None else f"{dev:.3f}",
        "-" if r["gap"] is None else f"{r['gap']:+.3f}",
    ]
    return "| " + " | ".join(cells) + " |"


ROW_HEADER = (
    "endpoint", "model", "rule", "scope", "F0.5 [95% CI]", "bull P / R", "bear P / R", "inversion",
    "coverage", "errors", "dev F0.5", "gap",
)  # fmt: skip


def render(report: Mapping[str, Any]) -> str:
    title = "# Holdout (dry run)" if report["dryRun"] else "# Holdout (run once)"
    guard = report.get("guard") or {}
    lines = [
        title,
        "",
        f"Generated by `{report['generatedBy']}`; ids and numbers only, no article or dataset text.",
        "",
        f"- Frozen config sha256 `{report['frozenSha256'][:12]}`, "
        f"selection `{(report['selectionSha256'] or '-')[:12]}`; "
        f"guard status {guard.get('status', '-')}, started {guard.get('startedAt', '-')}, completed "
        f"{guard.get('completedAt', '-')}"
        + (f", forced {len(guard['forced'])} time(s)" if guard.get("forced") else "")
        + ".",
        f"- {report['rules']}.",
        "",
    ]
    if report["missing"]:
        lines += [f"> **Missing cells ({len(report['missing'])}):** " + ", ".join(report["missing"][:8]), ""]
    for split, title_text in SPLIT_TITLES.items():
        rows = [r for r in report["rows"] if r["split"] == split]
        if not rows:
            continue
        gold = ", ".join(f"{k} {v}" for k, v in sorted(rows[0]["gold"].items()))
        lines += [f"## {title_text}, n={rows[0]['n']} ({gold})", ""]
        lines += ["| " + " | ".join(ROW_HEADER) + " |", "|" + "|".join("---" for _ in ROW_HEADER) + "|"]
        lines += [_row(r) for r in rows]
        lines.append("")
    summaries = report.get("summaries")
    if summaries:
        lines += [f"## Summaries on the test articles ({summaries['status']})", ""]
        header = ("endpoint", "model", "n", "pass", "key facts", "judge faithful", "judge cover", "halluc.")
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        for r in summaries.get("rows", []):
            judge = [
                "-" if v is None else f"{v:.2f}" for v in (r.get("judgeFaithfulness"), r.get("judgeCoverage"))
            ]
            halluc = r.get("hallucinationRate")
            cells = [
                f"/{r['endpoint']}",
                f"`{r['model']}`",
                str(r["n"]),
                f"{100 * r['deterministicPassRate']:.0f}%",
                f"{100 * r['keyFactCoverage']:.0f}%",
                *judge,
                "-" if halluc is None else f"{100 * halluc:.0f}%",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Setup:
    frozen_sha: str
    selection_sha: str | None
    winners: dict[str, str]
    cells: list[HCell]
    pairs: list[tuple[str, str]]


def setup(args: argparse.Namespace) -> Setup:
    frozen = read_json(FROZEN_JSON)
    if frozen is None:
        raise HoldoutError(f"{_rel(FROZEN_JSON)} is missing; run `freeze` first")
    registry = PromptRegistry()
    rules = rules_from_frozen(frozen, registry)
    selection = read_json(SELECTION_JSON)
    winners: dict[str, str] = {}
    if selection is not None:
        winners = {e: v["winner"] for e, v in selection["endpoints"].items() if v.get("winner")}
    if args.dry_run:
        chosen = [r for r in rules if not args.model or r.model in args.model]
        if args.model and CEILING in args.model:
            chosen.append(ceiling_rule(registry))
        cells = [HCell(rule, DRY_SPLIT) for rule in chosen]
        return Setup(file_sha(FROZEN_JSON), None, winners, cells, [])
    if selection is None or len(winners) < 3:
        raise HoldoutError(f"{_rel(SELECTION_JSON)} has no winner for every endpoint; run `select` first")
    if selection["provisional"] and not args.allow_provisional:
        raise HoldoutError(
            "the selection is provisional (" + "; ".join(selection["notes"]) + "); re-run `select` after the "
            "latency suite, or pass --allow-provisional"
        )
    cells = plan_cells(rules, ceiling_rule(registry), winners)
    return Setup(file_sha(FROZEN_JSON), file_sha(SELECTION_JSON), winners, cells, summary_pairs(winners))


def _split_sizes() -> dict[str, int]:
    sizes = {ARTICLES: len(_test_ids())}
    for name in (TWEETS, *CROSS):
        payload = read_json(SPLITS_DIR / f"{name}.json") or {}
        sizes[name] = int(payload.get("n", 0))
    return sizes


def _test_ids() -> list[str]:
    path = ArticlePaths().rows
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["id"] for r in rows if r.get("split") == "test"]


def cmd_plan(args: argparse.Namespace) -> int:
    args.dry_run = False
    s = setup(args)
    est = estimate(s.cells, _split_sizes(), len(s.pairs))
    for row in est["cells"]:
        print(
            f"{row['key']:75s} n={row['n']:5d} ~{row['perCallMs']:5d} ms/call ~{row['seconds'] / 60:5.1f} min"
        )
    print(f"summaries: {', '.join(f'{e}/{m}' for e, m in s.pairs)} on the test articles, then judge packets")
    print(
        f"{len(s.cells)} cells; estimated {est['totalSeconds'] / 3600:.1f} h at concurrency 1 "
        "(dev-run timings)"
    )
    return 0


async def _run(args: argparse.Namespace, s: Setup, inputs: Inputs, preds_root: Path, manifest: Path) -> None:
    settings = get_settings()
    client = OllamaClient(
        args.base_url or settings.ollama_base_url,
        timeout=settings.inference_timeout_s,
        keep_alive=settings.ollama_keep_alive,
    )
    async with client:
        await run_cells(
            client,
            s.cells,
            inputs,
            preds_root=preds_root,
            manifest_path=manifest,
            concurrency=args.concurrency,
            shared=args.shared,
        )
        if s.pairs and not args.skip_summaries:
            info = await run_summaries(client, s.pairs, sys.stderr)
            print(f"summary packets: {info['packets']} ({info['packetList']})", file=sys.stderr)


def _write(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    write_json(out_dir / HOLDOUT_JSON.name, report)
    (out_dir / HOLDOUT_MD.name).write_text(render(report), encoding="utf-8")
    return [out_dir / HOLDOUT_JSON.name, out_dir / HOLDOUT_MD.name]


def cmd_run(args: argparse.Namespace) -> int:
    if args.model and not args.dry_run:
        raise HoldoutError("--model only narrows a --dry-run; the holdout always runs every cell")
    s = setup(args)
    if args.dry_run:
        if args.out_dir is None:
            raise HoldoutError("--dry-run writes to a scratch folder: pass --out-dir")
        inputs = Inputs(limit=args.limit or DRY_LIMIT)
        preds, manifest = args.out_dir / "preds", args.out_dir / "manifest.json"
        asyncio.run(_run(args, s, inputs, preds, manifest))
        rows, missing = analyze_cells(
            s.cells, inputs, read_manifest(manifest), preds, args.n_boot, sys.stderr
        )
        report = build_holdout(
            rows,
            missing,
            frozen_sha=s.frozen_sha,
            selection_sha=None,
            guard_entry=None,
            summaries=None,
            n_boot=args.n_boot,
            dry_run=True,
        )
        for path in _write(report, args.out_dir):
            print(f"dry run: wrote {path}")
        return 0
    guard = read_guard()
    mode = guard_mode(guard, s.frozen_sha, force=args.force)
    write_json(
        GUARD,
        guard_start(
            guard,
            s.frozen_sha,
            mode=mode,
            selection_sha=s.selection_sha,
            cells=len(s.cells),
            reason=args.reason,
        ),
    )
    print(f"holdout {mode}: {len(s.cells)} cells, summaries {s.pairs}", file=sys.stderr)
    inputs = Inputs()
    asyncio.run(_run(args, s, inputs, PREDS_DIR, MANIFEST))
    write_json(GUARD, guard_complete(read_guard(), s.frozen_sha))
    return _analyze(s, inputs, args.n_boot)


def _analyze(s: Setup, inputs: Inputs, n_boot: int) -> int:
    rows, missing = analyze_cells(s.cells, inputs, read_manifest(MANIFEST), PREDS_DIR, n_boot, sys.stderr)
    report = build_holdout(
        rows,
        missing,
        frozen_sha=s.frozen_sha,
        selection_sha=s.selection_sha,
        guard_entry=read_guard()["runs"].get(s.frozen_sha),
        summaries=summary_block(s.pairs),
        n_boot=n_boot,
        dry_run=False,
    )
    for path in _write(report, HOLDOUT_DIR):
        print(f"wrote {_rel(path)}")
    if missing:
        print(f"{len(missing)} cells missing: {', '.join(missing[:5])}", file=sys.stderr)
    return 1 if missing else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Scores cached predictions only; never calls a model, so it may run at any time after `run`."""
    args.dry_run = False
    s = setup(args)
    if s.frozen_sha not in read_guard()["runs"]:
        raise HoldoutError("no holdout run is recorded for the current frozen config; nothing to analyze")
    return _analyze(s, Inputs(), args.n_boot)


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `holdout` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser("holdout", help="one-shot holdout run")
    actions = parser.add_subparsers(dest="holdout_action", required=True)

    plan = actions.add_parser("plan", help="print the holdout cells and a time estimate (no model calls)")
    plan.add_argument("--allow-provisional", action="store_true")
    plan.set_defaults(holdout_func=cmd_plan)

    running = actions.add_parser("run", help="run the holdout once (resumable; guarded per frozen config)")
    running.add_argument("--dry-run", action="store_true", help="a few dev300 rows into --out-dir; no guard")
    running.add_argument("--out-dir", type=Path, default=None, help="scratch folder for --dry-run")
    running.add_argument(
        "--limit", type=int, default=0, help=f"dev300 rows for --dry-run (default {DRY_LIMIT})"
    )
    running.add_argument("--model", action="append", help="--dry-run only: restrict to this model")
    running.add_argument("--force", action="store_true", help="re-run a completed holdout (logged loudly)")
    running.add_argument("--reason", default=None, help="why --force is needed; recorded in the guard file")
    running.add_argument("--allow-provisional", action="store_true", help="accept a provisional selection")
    running.add_argument("--skip-summaries", action="store_true", help="no test-article summaries")
    running.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    running.add_argument(
        "--shared", action="store_true", help="Ollama serves other runs: never unload models"
    )
    running.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    running.add_argument("--n-boot", type=int, default=N_BOOT)
    running.set_defaults(holdout_func=cmd_run)

    analyzing = actions.add_parser("analyze", help="re-score the cached holdout predictions")
    analyzing.add_argument("--allow-provisional", action="store_true")
    analyzing.add_argument("--n-boot", type=int, default=N_BOOT)
    analyzing.set_defaults(holdout_func=cmd_analyze)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        code: int = args.holdout_func(args)
    except (HoldoutError, RunAborted, se.SummaryEvalError, VtError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not isinstance(exc, HoldoutError):
            print(
                "the run stays open in the guard: re-running `holdout run` resumes from the caches",
                file=sys.stderr,
            )
        return 1
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.holdout", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["holdout", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
