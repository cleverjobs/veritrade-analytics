from __future__ import annotations

from typing import Any

import pytest
import respx

from veritrade_evals import select as sel
from veritrade_evals.select import Cand, Check

MODELS_TOML = """# header comment
[defaults]
signal = "small:1b"
summary = "small:1b"
analyze = "small:1b"

[models."small:1b"]
kind = "generative"
endpoints = ["signal", "summary", "analyze"]
num_ctx = 4096
max_input_tokens = 1000
num_predict = { signal = 16, summary = 160, analyze = 200 }
signal_schema = "label"
policy = "label-logprob"
min_p = 0.5

[models."big:9b"]
kind = "decision"
endpoints = ["signal"]
max_input_tokens = 2000
policy = "probability"
min_p = 0.8
"""


def latency(status: str, utilisation: float | None, value: float = 400.0) -> dict[str, Any]:
    checks = [{"metric": "p50 excl. fetch", "value": value, "limit": 500.0, "met": status == "met"}]
    return {"budget": {"status": status, "utilisation": utilisation, "checks": checks}, "warm": {"n": 30}}


def cand(model: str, quality: float, lat: dict[str, Any] | None, *, ok: bool = True) -> Cand:
    return Cand(
        endpoint="signal",
        model=model,
        config="v1",
        rule="probability @ 0.8",
        quality=quality,
        quality_label="dev300 F0.5",
        secondary=(),
        checks=[Check("dev300 inversion <= 2%", ok, "synthetic")],
        latency=lat,
    )


# --- winner rules ----------------------------------------------------------------------------


def test_winner_is_the_best_candidate_inside_every_filter() -> None:
    cands = [
        cand("best-but-slow", 0.90, latency("missed", 2.0, 1000.0)),
        cand("good-and-fast", 0.80, latency("met", 0.6, 300.0)),
        cand("worse-and-fast", 0.70, latency("met", 0.3, 150.0)),
        cand("inverts", 0.95, latency("met", 0.2, 100.0), ok=False),
    ]
    choice = sel.choose("signal", cands, provisional=False)
    assert choice.winner is not None
    assert choice.winner.model == "good-and-fast"
    assert choice.how == "inside-budget"
    assert not choice.provisional
    verdicts = {c.model: c.verdict for c in cands}
    assert verdicts["best-but-slow"].startswith("misses the budget")
    assert verdicts["worse-and-fast"].startswith("ranks below")
    assert verdicts["inverts"].startswith("excluded: dev300 inversion")


def test_no_candidate_inside_the_budget_falls_back_to_the_pareto_knee() -> None:
    cands = [
        cand("fast-weak", 0.60, latency("missed", 1.1)),
        cand("knee", 0.85, latency("missed", 1.3)),
        cand("slow-best", 0.87, latency("missed", 3.0)),
    ]
    choice = sel.choose("signal", cands, provisional=False)
    assert choice.how == "knee"
    assert choice.winner is not None
    assert choice.winner.model == "knee"
    assert "Pareto knee" in choice.reason
    assert "1.30x the budget" in choice.reason
    assert cands[2].verdict.startswith("not the knee")


def test_unmeasured_latency_gives_a_provisional_quality_only_pick() -> None:
    cands = [cand("a", 0.7, None), cand("b", 0.8, None)]
    choice = sel.choose("signal", cands, provisional=False)
    assert choice.how == "quality-only"
    assert choice.provisional
    assert choice.winner is not None
    assert choice.winner.model == "b"


def test_no_eligible_candidate_means_no_winner() -> None:
    choice = sel.choose("signal", [cand("a", 0.9, latency("met", 0.5), ok=False)], provisional=False)
    assert choice.winner is None
    assert choice.how == "none"


def test_articles_cap_uses_the_ci_so_one_inversion_in_twenty_calls_is_not_a_failure() -> None:
    def row(inversions: int, calls: int, point: float, low: float) -> dict[str, Any]:
        cis = {"inversion_rate": {"point": point, "low": low, "high": point}}
        articles = {"inversions": inversions, "directional_calls": calls, "inversion_rate": point, "cis": cis}
        return {"articles": articles, "clear_failure": []}

    dev, articles = sel._inversion_checks(0.0, row(1, 20, 0.05, 0.0))
    assert dev.passed
    assert articles.passed
    assert "point estimate alone would fail" in articles.detail
    assert sel._inversion_checks(0.0, row(6, 20, 0.30, 0.10))[1].passed is False
    assert sel._inversion_checks(0.03, None)[0].passed is False


# --- R8 bundle -------------------------------------------------------------------------------

SIZES = {
    "clef": 10_580_000_000,
    "nimble": 5_630_000_000,
    "q08": 1_020_000_000,
    "q2": 1_950_000_000,
    "q4": 3_320_000_000,
}


def test_bundle_keeps_the_winners_when_they_fit_and_counts_a_shared_model_once() -> None:
    plan = sel.bundle_plan({"signal": ["q08"], "summary": ["q4"], "analyze": ["q4"]}, SIZES)
    assert plan["status"] == "winners"
    assert plan["models"] == ["q08", "q4"]
    assert plan["totalBytes"] == SIZES["q08"] + SIZES["q4"]


def test_bundle_swaps_the_fewest_rank_steps_when_the_winners_do_not_fit() -> None:
    prefs = {"signal": ["clef", "nimble", "q08"], "summary": ["q4", "q2"], "analyze": ["q4"]}
    plan = sel.bundle_plan(prefs, SIZES)
    assert plan["status"] == "swapped"
    assert plan["defaults"]["signal"] == "nimble"
    assert plan["swaps"] == {"signal": {"winner": "clef", "bundled": "nimble"}}
    assert plan["totalBytes"] <= sel.SIZE_CAP_BYTES


def test_bundle_tie_on_rank_steps_keeps_the_earlier_endpoint_and_reports_nothing_fitting() -> None:
    prefs = {"signal": ["nimble", "q08"], "summary": ["q4", "q2"], "analyze": ["clef", "q2"]}
    plan = sel.bundle_plan(prefs, SIZES, cap=9_000_000_000)
    assert plan["defaults"] == {"signal": "nimble", "summary": "q2", "analyze": "q2"}
    assert sel.bundle_plan({"signal": ["clef"]}, SIZES)["status"] == "nothing fits"
    assert sel.bundle_plan({"signal": ["q08"]}, None)["status"] == "sizes unknown"


# --- writing ---------------------------------------------------------------------------------


def test_write_defaults_touches_only_the_defaults_table() -> None:
    out = sel.write_defaults(MODELS_TOML, {"signal": "big:9b", "summary": "small:1b", "analyze": "small:1b"})
    assert 'signal = "big:9b"' in out
    assert out.replace('signal = "big:9b"', 'signal = "small:1b"') == MODELS_TOML
    with pytest.raises(sel.SelectError):
        sel.write_defaults(MODELS_TOML, {"summary": "big:9b"})


def test_write_env_rewrites_keys_and_keeps_comments() -> None:
    env = "# keep me\nCOMPOSE_PROFILES=bundled\nVT_MODELS=old\n# VT_PULL_MODELS=commented\n"
    out = sel.write_env(env, {"VT_MODELS": "a,b", "VT_PULL_MODELS": "a,b"})
    assert out.splitlines() == [
        "# keep me",
        "COMPOSE_PROFILES=bundled",
        "VT_MODELS=a,b",
        "# VT_PULL_MODELS=commented",
        "VT_PULL_MODELS=a,b",
    ]


def test_write_refuses_a_provisional_selection(tmp_path: Any) -> None:
    selection = {"provisional": True, "notes": ["latency.json is missing"], "endpoints": {}, "bundle": {}}
    with pytest.raises(sel.SelectError, match="provisional"):
        sel.apply_write(selection, tmp_path / "models.toml", tmp_path / "bundled.env")


def test_model_sizes_come_from_ollama_tags() -> None:
    with respx.mock() as router:
        router.get("http://ollama.test:11434/api/tags").respond(
            200, json={"models": [{"name": "q08", "size": 7}, {"name": "odd"}]}
        )
        sizes, source = sel.model_sizes("http://ollama.test:11434")
    assert sizes == {"q08": 7}
    assert source == "ollama /api/tags"


# --- end to end on synthetic results ---------------------------------------------------------


def _grid(config: str, policy: str, inversion: float, f05: float) -> dict[str, Any]:
    point = {"threshold": 0.8, "metrics": {"n": 300, "inversion_rate": inversion, "f05_macro": f05}}
    cis = {"f05_macro": {"point": f05, "low": f05 - 0.05, "high": f05 + 0.05}}
    return {
        "configs": [{"name": config, "policies": {policy: {"operating_point": point}}}],
        "best": {"config": config, "policy": policy, "cis": cis},
    }


def _article_row(model: str, config: str, scope: str, policy: str, inversions: int) -> dict[str, Any]:
    rate = inversions / 20
    cis = {
        "f05_macro": {"point": 0.8, "low": 0.7, "high": 0.9},
        "inversion_rate": {"point": rate, "low": 0.0},
    }
    return {
        "key": f"{model}/{config}/{scope}",
        "policy": policy,
        "articles": {
            "f05_macro": 0.8,
            "inversion_rate": rate,
            "inversions": inversions,
            "directional_calls": 20,
            "coverage": 1.0,
            "cis": cis,
        },
        "clear_failure": [],
        "latency": {"wall_ms_p50": 500.0},
    }


def test_build_selection_on_synthetic_results() -> None:
    def signal_entry(f05: float) -> dict[str, Any]:
        ev = {
            "config": "v1",
            "policy": "probability",
            "threshold": 0.8,
            "f05_macro": {"point": f05, "low": f05 - 0.05, "high": f05 + 0.05},
            "inversion_rate": {"point": 0.0, "low": 0.0, "high": 0.0},
        }
        return {"signal": {"prompt_version": "p", "evidence": {"dev300": ev}}}

    frozen = {
        "models": {
            "fast": signal_entry(0.7),
            "slow": signal_entry(0.9),
            "an": {
                "analyze": {
                    "prompt_version": "p",
                    "ordinal_t": 1,
                    "field_order": "sentiment-first",
                    "evidence": {"dev300": {"config": "v1"}},
                },
                "summary": {"prompt": "v2", "prompt_version": "p"},
            },
        }
    }
    grid = {
        ("signal", "fast"): _grid("v1", "probability", 0.0, 0.7),
        ("signal", "slow"): _grid("v1", "probability", 0.0, 0.9),
        ("analyze", "an"): _grid("v1", "ordinal", 0.01, 0.75),
    }
    articles = {
        "rows": [
            _article_row("fast", "v1", "lede", "probability", 0),
            _article_row("slow", "v1", "lede", "probability", 1),
            _article_row("an", "v1", "full", "ordinal", 0),
        ]
    }
    summary_row = {"endpoint": "summary", "model": "an", "n": 30, "deterministicPassRate": 0.9}
    summary = {
        "config": {"passGate": 0.8},
        "splits": {"dev": {"rows": [summary_row | {"keyFactCoverage": 0.5, "judgeFaithfulness": 4.2}]}},
    }

    def cell(key: str, status: str, utilisation: float) -> dict[str, Any]:
        return {"key": key, "backend": "native-metal", **latency(status, utilisation)}

    latency_json = {
        "clean": True,
        "missing": [],
        "partial": [],
        "cells": [
            cell("signal/fast/lede/json", "met", 0.4),
            cell("signal/slow/lede/json", "missed", 2.6),
            cell("summary/an/full/stream", "met", 0.7),
            cell("analyze/an/full/stream", "met", 0.9),
        ],
    }
    inputs = sel.Inputs(
        frozen, grid, articles, summary, latency_json, {"fast": 1, "slow": 2, "an": 3}, "test"
    )
    selection = sel.build_selection(inputs)
    winners = {e: v["winner"] for e, v in selection["endpoints"].items()}
    assert winners == {"signal": "fast", "summary": "an", "analyze": "an"}
    assert not selection["provisional"]
    assert selection["bundle"]["models"] == ["an", "fast"]
    slow = next(c for c in selection["endpoints"]["signal"]["candidates"] if c["model"] == "slow")
    assert slow["verdict"].startswith("misses the budget")
    shown = sel.frontend_selection(selection)
    assert shown["signal"]["model"] == "fast"
    assert "inside every filter" in shown["signal"]["reason"]
    assert "## /signal: `fast` (inside-budget)" in sel.render(selection)
