"""Lean prompt x knob grid on dev300: `uv run python -m veritrade_evals.grid {list,run,analyze}`."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from veritrade.config import get_settings
from veritrade.inference.ollama import FIXED_OPTIONS, OllamaClient
from veritrade.models_registry import allowlist
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals.analysis import (
    CellInput,
    DevSet,
    best_candidate,
    best_report,
    bucket_of,
    candidates,
    counts,
    diagnostics,
    directional_mass_report,
    error_patterns,
    knob_comparison,
    latency,
    policy_report,
    render_summary,
    rounded,
    served_calls,
)
from veritrade_evals.datasets import Example
from veritrade_evals.metrics import INVERSION_CAP
from veritrade_evals.paths import PREDS_DIR, RESULTS_DIR
from veritrade_evals.predcache import canonical_json, model_slug
from veritrade_evals.runner import (
    DEFAULT_CONCURRENCY,
    RunSpec,
    Variant,
    make_resident,
    model_digest,
    run_cell,
    serving_spec,
)
from veritrade_evals.splits import HOLDOUT_SPLITS, Split, SplitName, load_split, read_split_file

GRID_SPLIT: SplitName = "dev300"
if GRID_SPLIT in HOLDOUT_SPLITS:
    raise RuntimeError("the grid tunes prompts and thresholds, so it must never read a holdout split")

GRID_DIR = RESULTS_DIR / "grid"
MANIFEST = GRID_DIR / "manifest.json"
NOTES = GRID_DIR / "notes.md"
VERSIONS = ("v1", "v2", "v3", "v4")


@dataclass(frozen=True, slots=True)
class ModelPlan:
    model: str
    variants: tuple[Variant, ...]


# Execution order: the fast baseline first so a broken pipeline shows up in minutes.
PLAN: tuple[ModelPlan, ...] = (
    ModelPlan("qwen3.5:0.8b-q8_0", ("label", "ordinal")),
    ModelPlan("clef-flash:9b-q8_0", ("decision",)),
    ModelPlan("nimble:9b-q4_K_M", ("decision",)),
    ModelPlan("ministral-3:3b", ("sentiment-first",)),
    ModelPlan("qwen3:4b-instruct-2507-q4_K_M", ("sentiment-first",)),
    ModelPlan("qwen3.5:4b-q4_K_M", ("sentiment-first",)),
)
EXCLUDED = {
    "tev1:0.8b-q8_0": "CI plumbing model, not evaluated",
    "qwen3.5:9b-q4_K_M": "holdout-only quality ceiling",
}
# What each prompt version varies; test_grid checks these labels against the prompt files.
DIMS: dict[str, dict[str, dict[str, str]]] = {
    "decision": {
        "v1": {"instructions": "rubric", "criteria": "described"},
        "v2": {"instructions": "rubric", "criteria": "bare"},
        "v3": {"instructions": "plain", "criteria": "described"},
        "v4": {"instructions": "plain", "criteria": "bare"},
    },
    "generative": {
        "v1": {"rubric": "A", "shots": "0"},
        "v2": {"rubric": "A", "shots": "4"},
        "v3": {"rubric": "B", "shots": "0"},
        "v4": {"rubric": "B", "shots": "4"},
    },
}


def cell_name(version: str, variant: Variant) -> str:
    return f"{version}-{variant}" if variant in ("label", "ordinal") else version


def cell_key(cell: RunSpec) -> str:
    return f"{cell.endpoint}/{cell.model}/{cell.name}"


def grid_cells(models: Collection[str] = (), registry: PromptRegistry | None = None) -> list[RunSpec]:
    registry = registry or PromptRegistry()
    roster = allowlist()
    unknown = set(models) - {p.model for p in PLAN}
    if unknown:
        raise ValueError(f"not in the grid: {sorted(unknown)}")
    cells: list[RunSpec] = []
    for plan in PLAN:
        if models and plan.model not in models:
            continue
        base = roster[plan.model]
        for variant in plan.variants:
            spec = serving_spec(base, variant)
            for version in VERSIONS:
                cell = RunSpec(
                    cell_name(version, variant),
                    spec,
                    registry.get(
                        "analyze" if variant == "sentiment-first" else "signal",
                        plan.model,
                        kind=base.kind,
                        version=version,
                    ),
                    variant,
                    DIMS[base.kind][version],
                )
                cells.append(cell)
    return cells


# --- manifest --------------------------------------------------------------------------------


def read_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if not path.exists():
        return {"split": GRID_SPLIT, "cells": {}}
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def write_manifest(manifest: dict[str, Any], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["cells"] = dict(sorted(manifest["cells"].items()))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- run -------------------------------------------------------------------------------------


async def run_grid(
    client: OllamaClient,
    cells: Sequence[RunSpec],
    examples: Sequence[Example],
    *,
    concurrency: int,
    preds_root: Path = PREDS_DIR,
    manifest_path: Path = MANIFEST,
    log: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Model by model with only that model resident; the manifest is rewritten after every cell."""
    manifest = read_manifest(manifest_path)
    manifest["split"] = GRID_SPLIT
    manifest["ollama_version"] = await client.version()
    manifest["backend"] = get_settings().ollama_backend_label
    ids = {e.id for e in examples}
    for model, group in itertools.groupby(cells, key=lambda c: c.model):
        model_cells = list(group)
        digest = await model_digest(client, model)
        if any(not ids <= c.cache(GRID_SPLIT, digest, preds_root).done_ids() for c in model_cells):
            print(f"== {model}: unloading others and warming up", file=log, flush=True)
            await make_resident(client, model_cells[0], log=log)
        for cell in model_cells:
            stats = await run_cell(
                client,
                cell,
                examples,
                split=GRID_SPLIT,
                model_digest=digest,
                concurrency=concurrency,
                root=preds_root,
                log=log,
            )
            records = cell.cache(GRID_SPLIT, digest, preds_root).records()
            manifest["cells"][cell_key(cell)] = {
                "endpoint": cell.endpoint,
                "model": cell.model,
                "config": cell.name,
                "variant": cell.variant,
                "dims": dict(cell.dims),
                "prompt_version": cell.prompt.version,
                "prompt_sha": cell.prompt_sha(),
                "run_config": json.loads(canonical_json(cell.run_config(GRID_SPLIT))),
                "model_digest": digest,
                "cache_key": stats.key,
                "done": len(records),
                "errors": sum(1 for r in records.values() if r.get("error") is not None),
                "concurrency": concurrency,
                "updated_at": _now(),
            }
            write_manifest(manifest, manifest_path)
    return manifest


# --- analyze ---------------------------------------------------------------------------------


def dev_set(split: Split) -> DevSet:
    weights = split.example_weights()
    return DevSet(
        ids=tuple(e.id for e in split.examples),
        gold=tuple(e.label for e in split.examples),
        weights=None if weights is None else tuple(weights),
        buckets=tuple(bucket_of(e.text) for e in split.examples),
    )


def cell_inputs(
    cells: Sequence[RunSpec], manifest: dict[str, Any], dev: DevSet, preds_root: Path, log: TextIO
) -> list[CellInput]:
    """Cached records per cell, aligned to the split; skips cells not run yet or run under other prompts."""
    out: list[CellInput] = []
    for cell in cells:
        entry = manifest.get("cells", {}).get(cell_key(cell))
        if entry is None:
            print(f"skip {cell_key(cell)}: not run yet", file=log)
            continue
        current = json.loads(canonical_json(cell.run_config(GRID_SPLIT)))
        if entry["prompt_sha"] != cell.prompt_sha() or entry["run_config"] != current:
            print(f"skip {cell_key(cell)}: prompt or config changed since the run; re-run it", file=log)
            continue
        records = cell.cache(GRID_SPLIT, entry["model_digest"], preds_root).records()
        aligned = tuple(records.get(i) for i in dev.ids)
        missing = sum(r is None for r in aligned)
        if missing:
            print(f"warning {cell_key(cell)}: {missing} examples missing, scored as abstentions", file=log)
        out.append(
            CellInput(
                endpoint=cell.endpoint,
                model=cell.model,
                name=cell.name,
                variant=cell.variant,
                prompt_version=cell.prompt.version,
                records=aligned,
                dims=dict(cell.dims),
                ordinal_t=cell.spec.ordinal_t,
                meta={"cache_key": entry["cache_key"], "model_digest": entry["model_digest"]},
            )
        )
    return out


def model_report(inputs: Sequence[CellInput], dev: DevSet, class_weights: Any, n_boot: int) -> dict[str, Any]:
    cands = candidates(inputs, dev)
    first = inputs[0]
    configs = []
    for cell in inputs:
        own = [c for c in cands if c.cell is cell]
        configs.append(
            {
                "name": cell.name,
                "variant": cell.variant,
                "prompt_version": cell.prompt_version,
                "dims": dict(cell.dims),
                "cache_key": cell.meta.get("cache_key"),
                "counts": counts(cell.records),
                "diagnostics": diagnostics(cell),
                "latency": latency(cell.records),
                "policies": {c.evaluation.policy: policy_report(c.evaluation) for c in own},
            }
        )
    report: dict[str, Any] = {
        "endpoint": first.endpoint,
        "model": first.model,
        "model_digest": first.meta.get("model_digest"),
        "split": GRID_SPLIT,
        "n": len(dev.ids),
        "class_weights": class_weights,
        "inversion_cap": INVERSION_CAP,
        "generated_by": "uv run python -m veritrade_evals.grid analyze",
        "configs": configs,
        "best": best_report(cands, dev, n_boot),
    }
    knob = knob_comparison(cands, dev, n_boot) if first.variant in ("label", "ordinal") else None
    if knob is not None:
        report["knob_comparison"] = knob
    mass = directional_mass_report(inputs, cands, dev, n_boot)
    if mass is not None:
        report["directional_mass"] = mass
    return report


def analyze_grid(
    *,
    n_boot: int = 1000,
    preds_root: Path = PREDS_DIR,
    manifest_path: Path = MANIFEST,
    out_dir: Path = GRID_DIR,
    log: TextIO = sys.stderr,
) -> list[Path]:
    split = load_split(GRID_SPLIT)
    payload = read_split_file(GRID_SPLIT)
    manifest = read_manifest(manifest_path)
    dev = dev_set(split)
    inputs = cell_inputs(grid_cells(), manifest, dev, preds_root, log)
    groups = [list(g) for _, g in itertools.groupby(inputs, key=lambda c: (c.endpoint, c.model))]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    reports: list[dict[str, Any]] = []
    served = []
    for group in groups:
        report = rounded(model_report(group, dev, payload["class_weights"], n_boot))
        reports.append(report)
        served.append(served_calls(*best_candidate(candidates(group, dev))))
        path = out_dir / f"{report['endpoint']}__{model_slug(report['model'])}.json"
        path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    if not reports:
        print("nothing to analyze: run the grid first", file=log)
        return written
    patterns = error_patterns(served, dev)
    concurrency = sorted({e.get("concurrency") for e in manifest["cells"].values()} - {None})
    header = {
        "n": len(dev.ids),
        "train_prior": payload["train_prior"],
        "n_boot": n_boot,
        "ollama_version": manifest.get("ollama_version", "?"),
        "backend": manifest.get("backend", "?"),
        "concurrency": ", ".join(str(c) for c in concurrency) or "?",
        "decoding": ", ".join(f"{k}={v}" for k, v in FIXED_OPTIONS.items()),
    }
    notes = NOTES.read_text(encoding="utf-8") if out_dir == GRID_DIR and NOTES.exists() else None
    summary = out_dir / "summary.md"
    summary.write_text(render_summary(reports, patterns, header=header, notes=notes), encoding="utf-8")
    patterns_path = out_dir / "error_patterns.json"
    patterns_path.write_text(json.dumps(rounded(patterns), indent=1) + "\n", encoding="utf-8")
    return [*written, summary, patterns_path]


# --- CLI -------------------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    cells = grid_cells(args.model or ())
    for cell in cells:
        dims = " ".join(f"{k}={v}" for k, v in cell.dims.items())
        print(f"{cell_key(cell):55s} {cell.prompt.version:45s} {dims}")
    per_model = {m: sum(c.model == m for c in cells) for m in dict.fromkeys(c.model for c in cells)}
    print(f"{len(cells)} configs: " + ", ".join(f"{m}={n}" for m, n in per_model.items()))
    print("excluded: " + "; ".join(f"{m} ({why})" for m, why in EXCLUDED.items()))
    return 0


async def _run(args: argparse.Namespace) -> int:
    cells = grid_cells(args.model or ())
    examples = list(load_split(GRID_SPLIT).examples)
    if args.limit:
        examples = examples[: args.limit]
    settings = get_settings()
    client = OllamaClient(
        args.base_url or settings.ollama_base_url,
        timeout=settings.inference_timeout_s,
        keep_alive=settings.ollama_keep_alive,
    )
    async with client:
        manifest = await run_grid(
            client,
            cells,
            examples,
            concurrency=args.concurrency,
            preds_root=args.preds_root,
            manifest_path=args.manifest,
        )
    done = sum(e["done"] for e in manifest["cells"].values())
    errors = sum(e["errors"] for e in manifest["cells"].values())
    print(
        f"grid: {len(manifest['cells'])} cells in the manifest, {done} predictions, {errors} errors",
        file=sys.stderr,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(_run(args))


def cmd_analyze(args: argparse.Namespace) -> int:
    paths = analyze_grid(
        n_boot=args.n_boot, preds_root=args.preds_root, manifest_path=args.manifest, out_dir=args.out_dir
    )
    for path in paths:
        print(path)
    return 0 if paths else 1


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `grid` on a parent CLI; dispatch with `run(args)`."""
    grid: argparse.ArgumentParser = subparsers.add_parser("grid", help="lean prompt x knob grid on dev300")
    actions = grid.add_subparsers(dest="grid_action", required=True)

    listing = actions.add_parser("list", help="print the grid configs")
    listing.add_argument("--model", action="append", help="restrict to this model (repeatable)")
    listing.set_defaults(grid_func=cmd_list)

    running = actions.add_parser("run", help="run (or resume) the grid against Ollama")
    running.add_argument("--model", action="append", help="restrict to this model (repeatable)")
    running.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    running.add_argument("--limit", type=int, default=0, help="first N dev300 examples only (smoke runs)")
    running.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    running.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    running.add_argument("--manifest", type=Path, default=MANIFEST)
    running.set_defaults(grid_func=cmd_run)

    analyzing = actions.add_parser("analyze", help="score cached predictions and write evals/results/grid")
    analyzing.add_argument("--n-boot", type=int, default=1000)
    analyzing.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    analyzing.add_argument("--manifest", type=Path, default=MANIFEST)
    analyzing.add_argument("--out-dir", type=Path, default=GRID_DIR)
    analyzing.set_defaults(grid_func=cmd_analyze)
    return grid


def run(args: argparse.Namespace) -> int:
    code: int = args.grid_func(args)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.grid", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["grid", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
