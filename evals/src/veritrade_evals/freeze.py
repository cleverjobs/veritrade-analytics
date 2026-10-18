"""Freeze prompt versions and thresholds into models.toml: `uv run python -m veritrade_evals freeze`.
Reads committed results only (the dev300 grid reports and the articles dev summary), records the frozen set
in evals/results/frozen.json and leaves [defaults] to `select --write`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veritrade.inference.types import Endpoint
from veritrade.models_registry import ModelSpec, parse_models_toml
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals.analysis import TAU_BASE, is_tau
from veritrade_evals.finalists import OUT_DIR as ARTICLES_DIR
from veritrade_evals.grid import GRID_DIR, PLAN
from veritrade_evals.paths import REPO_ROOT, RESULTS_DIR
from veritrade_evals.predcache import model_slug

MODELS_TOML = REPO_ROOT / "backend" / "src" / "veritrade" / "data" / "models.toml"
FROZEN_JSON = RESULTS_DIR / "frozen.json"
# The /analyze revision every analyze model serves, once its accuracy check passes.
ANALYZE_VERSION = "v5"
FIELD_ORDER_CHECK = "v1-summary-first"
# Summary-endpoint models that get the tightened summary prompt without an accuracy gate.
SUMMARY_VERSION = "v2"
SUMMARY_MODELS = ("qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M")
# Keys the freeze owns, in the order a models.toml table lists them.
KEY_ORDER = (
    "kind", "endpoints", "num_ctx", "max_input_tokens", "num_predict", "signal_schema", "policy", "min_p",
    "ordinal_t", "directional_tau", "field_order", "prompt", "evaluated",
)  # fmt: skip
# The spec keys each endpoint's serving decision reads.
ENDPOINT_KEYS: dict[Endpoint, tuple[str, ...]] = {
    "signal": ("signal_schema", "policy", "min_p", "directional_tau", "ordinal_t"),
    "analyze": ("field_order", "ordinal_t"),
    "summary": (),
}


class FreezeError(RuntimeError):
    pass


@dataclass(slots=True)
class Frozen:
    """What one model serves after the freeze, plus the evidence behind each choice."""

    model: str
    values: dict[str, Any] = field(default_factory=dict)
    prompt: dict[Endpoint, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FreezeError(f"{path.relative_to(REPO_ROOT)} is missing; run its analysis first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def grid_report(endpoint: Endpoint, model: str, grid_dir: Path = GRID_DIR) -> dict[str, Any]:
    return _read(grid_dir / f"{endpoint}__{model_slug(model)}.json")


def version_id(prompt_version: str, endpoint: Endpoint, model: str, registry: PromptRegistry) -> str:
    """`v4` from `signal/m/v4@hash`, after checking the prompt file still hashes the same."""
    version = prompt_version.split("/")[-1].split("@")[0]
    if registry.get(endpoint, model, version=version).version != prompt_version:
        raise FreezeError(f"{prompt_version} changed since it was scored; re-run the grid and the finalists")
    return version


def _article_rows(articles: Mapping[str, Any], model: str, config: str) -> list[dict[str, Any]]:
    return [r for r in articles["rows"] if r["model"] == model and r["config"] == config]


def _brief(row: Mapping[str, Any]) -> dict[str, Any]:
    a = row["articles"]
    return {
        "scope": row["scope"],
        "policy": row["policy"],
        "threshold": row["threshold"],
        "f05_macro": a["cis"]["f05_macro"],
        "inversions": f"{a['inversions']}/{a['directional_calls']}",
        "coverage": a["coverage"],
        "clear_failure": row["clear_failure"],
    }


def _comparison(articles: Mapping[str, Any], what: str, a_key: str) -> dict[str, Any] | None:
    return next(
        (c for c in articles["comparisons"] if c["comparison"] == what and c["a"].startswith(a_key)), None
    )


def freeze_signal(model: str, articles: Mapping[str, Any], registry: PromptRegistry) -> Frozen:
    best = grid_report("signal", model)["best"]
    policy, threshold, variant = best["policy"], float(best["threshold"]), best["variant"]
    rows = [r for r in _article_rows(articles, model, best["config"]) if r["policy"] == policy]
    served = next((r for r in rows if r["scope"] == "lede"), None)
    evidence: dict[str, Any] = {
        "dev300": {"config": best["config"], "policy": policy, "threshold": threshold, **best["cis"]},
        "articles_dev": [_brief(r) for r in rows],
    }
    if served is not None and served["clear_failure"]:
        repick = served.get("repick")
        if repick is None:
            raise FreezeError(
                f"{model}: the dev300 rule clearly fails on the articles and no re-pick meets the cap"
            )
        threshold = float(repick["threshold"])
        evidence["repicked_on_articles"] = {"threshold": threshold, "because": served["clear_failure"]}
    base = TAU_BASE[policy] if is_tau(policy) else policy
    values: dict[str, Any] = {
        "signal_schema": {"label": "label", "ordinal": "ordinal"}.get(variant),
        "policy": base,
        "min_p": None,
        "directional_tau": None,
    }
    if base == "ordinal":
        values["ordinal_t"] = int(threshold)
    elif is_tau(policy):
        values |= {"min_p": 0.0, "directional_tau": threshold}
    else:
        values["min_p"] = threshold
    version = version_id(best["prompt_version"], "signal", model, registry)
    return Frozen(model, values, {"signal": version}, {"signal": evidence})


def _gate(follow_up: Mapping[str, Any], articles_cmp: Mapping[str, Any] | None) -> str | None:
    """Why a follow-up may not be frozen: significantly worse on dev300 or on the articles."""
    if follow_up["verdict"] == "against":
        gap = follow_up["paired_bootstrap"]["gap"]
        return f"dev300 F0.5 gap {gap['point']:+.3f} [{gap['low']:+.3f}, {gap['high']:+.3f}]"
    if articles_cmp is not None and articles_cmp["verdict"] == "b":
        gap = articles_cmp["gap"]
        return f"articles F0.5 gap {gap['point']:+.3f} [{gap['low']:+.3f}, {gap['high']:+.3f}]"
    return None


def freeze_analyze(model: str, articles: Mapping[str, Any], registry: PromptRegistry) -> Frozen:
    """The revision unless it is clearly less accurate than the grid best, which then stays served."""
    report = grid_report("analyze", model)
    follow_ups = {f["config"]: f for f in report.get("follow_ups", ())}
    revision = follow_ups.get(ANALYZE_VERSION)
    if revision is None:
        raise FreezeError(f"{model}: no dev300 result for analyze {ANALYZE_VERSION}; run the grid follow-ups")
    articles_cmp = _comparison(articles, f"prompt: {ANALYZE_VERSION} - v1", f"{model}/{ANALYZE_VERSION}/")
    why = _gate(revision, articles_cmp)
    if why is None and any(r["clear_failure"] for r in _article_rows(articles, model, ANALYZE_VERSION)):
        why = "clear failure on the articles"
    chosen = revision if why is None else report["best"]
    order = "sentiment-first"
    check = follow_ups.get(FIELD_ORDER_CHECK)
    order_cmp = _comparison(articles, "field order: summary-first - sentiment-first", f"{model}/")
    # Summary-first gives up the early score, so it has to win on dev300 and not lose on the articles.
    if (
        check is not None
        and check["verdict"] == "follow-up"
        and (order_cmp is None or order_cmp["verdict"] != "b")
    ):
        order = "summary-first"
    evidence = {
        "dev300": {
            "config": chosen["config"],
            "threshold": chosen["threshold"],
            "revision_f05": revision["paired_bootstrap"]["follow_up"],
            "revision_gap_vs_v1": revision["paired_bootstrap"]["gap"],
        },
        "articles_dev": [_brief(r) for r in _article_rows(articles, model, chosen["config"])],
        "articles_revision_gap_vs_v1": None if articles_cmp is None else articles_cmp["gap"],
        "revision_rejected": why,
        "field_order": {
            "dev300_gap_summary_first": None if check is None else check["paired_bootstrap"]["gap"],
            "articles_gap_summary_first": None if order_cmp is None else order_cmp["gap"],
            "chosen": order,
        },
    }
    version = version_id(chosen["prompt_version"], "analyze", model, registry)
    values = {"policy": "ordinal", "ordinal_t": int(chosen["threshold"]), "field_order": order}
    return Frozen(model, values, {"analyze": version}, {"analyze": evidence})


def frozen_set(articles: Mapping[str, Any], registry: PromptRegistry | None = None) -> dict[str, Frozen]:
    registry = registry or PromptRegistry()
    out: dict[str, Frozen] = {}
    for plan in PLAN:
        endpoint: Endpoint = (
            "analyze" if plan.variants[0] in ("sentiment-first", "summary-first") else "signal"
        )
        freeze = freeze_analyze if endpoint == "analyze" else freeze_signal
        out[plan.model] = freeze(plan.model, articles, registry)
    for model in SUMMARY_MODELS:
        frozen = out.setdefault(model, Frozen(model))
        registry.get("summary", model, version=SUMMARY_VERSION)
        frozen.prompt["summary"] = SUMMARY_VERSION
        frozen.evidence["summary"] = "tightened summary rule; quality is scored by the summary eval"
    return out


# --- models.toml -----------------------------------------------------------------------------


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{k} = {toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {value!r} to models.toml")


_KEY = re.compile(r"^([A-Za-z_]+)\s*=")


def set_keys(text: str, model: str, values: Mapping[str, Any]) -> str:
    """Rewrites only the given keys inside one model's table (None removes a key); comments survive."""
    lines = text.split("\n")
    header = f'[models."{model}"]'
    if header not in lines:
        raise FreezeError(f"{model} has no table in models.toml")
    start = lines.index(header) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("[")), len(lines))
    while end > start and not lines[end - 1].strip():
        end -= 1
    block = lines[start:end]
    for key, value in values.items():
        keys = [m.group(1) if (m := _KEY.match(line)) else None for line in block]
        if key in keys:
            i = keys.index(key)
            if value is None:
                del block[i]
            else:
                block[i] = f"{key} = {toml_value(value)}"
        elif value is not None:
            rank = KEY_ORDER.index(key)
            after = [i for i, k in enumerate(keys) if k in KEY_ORDER and KEY_ORDER.index(k) < rank]
            block.insert(after[-1] + 1 if after else len(block), f"{key} = {toml_value(value)}")
    return "\n".join([*lines[:start], *block, *lines[end:]])


def _defaults_block(text: str) -> str:
    lines = text.split("\n")
    start = lines.index("[defaults]")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("["))
    return "\n".join(lines[start:end])


def apply(text: str, frozen: Mapping[str, Frozen]) -> str:
    out = text
    current, _ = parse_models_toml(text)
    for model, f in frozen.items():
        prompt = {**current[model].prompt, **f.prompt}
        out = set_keys(out, model, {**f.values, "prompt": prompt or None})
    specs, _ = parse_models_toml(out)
    if _defaults_block(out) != _defaults_block(text):
        raise FreezeError("the freeze must not change [defaults]")
    for model, f in frozen.items():
        _check(specs[model], f)
    return out


def _check(spec: ModelSpec, frozen: Frozen) -> None:
    for key, value in frozen.values.items():
        if getattr(spec, key) != value:
            raise FreezeError(f"{spec.name}: {key} reads back as {getattr(spec, key)!r}, not {value!r}")
    for endpoint, version in frozen.prompt.items():
        if spec.prompt.get(endpoint) != version:
            raise FreezeError(f"{spec.name}: prompt for /{endpoint} did not stick")


def frozen_json(frozen: Mapping[str, Frozen], text: str, registry: PromptRegistry) -> dict[str, Any]:
    specs, defaults = parse_models_toml(text)
    models: dict[str, Any] = {}
    for model, f in frozen.items():
        spec = specs[model]
        models[model] = {
            endpoint: {
                "prompt": spec.prompt.get(endpoint),
                "prompt_version": registry.get(endpoint, model, version=spec.prompt.get(endpoint)).version,
                **{k: getattr(spec, k) for k in ENDPOINT_KEYS[endpoint]},
                "evidence": f.evidence.get(endpoint),
            }
            for endpoint in spec.endpoints
            if endpoint in f.prompt
        }
    return {
        "generated_by": "uv run python -m veritrade_evals freeze",
        "sources": {
            "dev300": str(GRID_DIR.relative_to(REPO_ROOT)),
            "articles_dev": str((ARTICLES_DIR / "summary.json").relative_to(REPO_ROOT)),
        },
        "rules": {
            "signal": "dev300 best rule (max F0.5 with inversions <= 2%); re-picked on the articles only "
            "when the whole inversion CI is over the cap or coverage collapses",
            "analyze": f"{ANALYZE_VERSION} unless clearly less accurate than v1 (paired 95% CI of the F0.5 "
            "gap below 0) on dev300 or the articles, else v1 stays; t from the chosen config's dev300 "
            "operating point; summary-first only if it wins on dev300 and does not lose on the articles",
            "summary": f"{SUMMARY_VERSION} for {', '.join(SUMMARY_MODELS)}",
        },
        "defaults_untouched": defaults,
        "models": models,
    }


# --- CLI -------------------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    articles = _read(args.articles)
    registry = PromptRegistry()
    frozen = frozen_set(articles, registry)
    before = MODELS_TOML.read_text(encoding="utf-8")
    after = apply(before, frozen)
    for model, f in frozen.items():
        print(f"{model}: {', '.join(f'{k}={v}' for k, v in {**f.values, 'prompt': f.prompt}.items())}")
    if args.dry_run:
        print("dry run: models.toml unchanged" if after == before else "dry run: models.toml would change")
        return 0
    MODELS_TOML.write_text(after, encoding="utf-8")
    payload = frozen_json(frozen, after, registry)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {MODELS_TOML.relative_to(REPO_ROOT)} and {args.out.relative_to(REPO_ROOT)}")
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `freeze` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "freeze", help="write frozen prompt versions and thresholds into models.toml"
    )
    parser.add_argument("--articles", type=Path, default=ARTICLES_DIR / "summary.json")
    parser.add_argument("--out", type=Path, default=FROZEN_JSON)
    parser.add_argument("--dry-run", action="store_true", help="print the frozen set without writing")
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        return cmd_freeze(args)
    except FreezeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.freeze", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["freeze", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
