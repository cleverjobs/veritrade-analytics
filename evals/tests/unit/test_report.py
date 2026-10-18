from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from veritrade_evals import report as rep

SAMPLE = rep.REPO_ROOT / "frontend" / "src" / "data" / "benchmarks.sample.json"


def sample() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return payload


# --- the BenchmarksV1 mirror -----------------------------------------------------------------


def test_mirror_matches_the_frontend_interfaces() -> None:
    ts = rep.ts_interfaces(rep.TS_TYPES.read_text(encoding="utf-8"))
    assert set(ts) == set(rep.INTERFACES)
    for name, fields in ts.items():
        assert set(fields) == set(rep.INTERFACES[name]), name
        for key, type_text in fields.items():
            assert set(rep.ts_kinds(type_text)) == set(rep.INTERFACES[name][key]), f"{name}.{key}"


def test_the_committed_frontend_sample_validates() -> None:
    assert rep.validate(sample()) == []


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda p: p.pop("articles"), "missing key 'articles'"),
        (lambda p: p.update(extra=1), "unexpected key 'extra'"),
        (lambda p: p.update(version=2), "benchmarks.version"),
        (lambda p: p["signal"][0].update(scope="headline"), "benchmarks.signal[0].scope"),
        (lambda p: p["latency"][0].update(streaming="yes"), "benchmarks.latency[0].streaming"),
        (lambda p: p["selection"].pop("summary"), "benchmarks.selection: expected exactly"),
        (lambda p: p["articles"]["perSource"]["bbc"].pop("neutral"), "benchmarks.articles.perSource.bbc"),
        (lambda p: p["environment"].update(ramGb=True), "benchmarks.environment.ramGb"),
    ],
)
def test_validation_catches_shape_breaks(mutate: Any, problem: str) -> None:
    payload = sample()
    mutate(payload)
    assert any(problem in p for p in rep.validate(payload)), rep.validate(payload)


# --- assembly --------------------------------------------------------------------------------


@pytest.fixture
def empty_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every input path points into an empty folder."""
    for name in (
        "FROZEN_JSON", "ARTICLES_JSON", "SUMMARY_JSON", "SELECTION_JSON", "LATENCY_JSON", "PREFLIGHT_JSON",
        "HOLDOUT_JSON", "EXTRACTION_JSON", "STATS_JSON", "GUARD",
    ):  # fmt: skip
        monkeypatch.setattr(rep, name, tmp_path / f"{name.lower()}.json")
    monkeypatch.setattr(rep, "GRID_DIR", tmp_path / "grid")
    monkeypatch.setattr(rep, "git_sha", lambda: None)
    return tmp_path


def test_missing_inputs_give_empty_sections_and_are_listed_never_invented(empty_results: Path) -> None:
    payload, missing = rep.build_benchmarks()
    assert rep.validate(payload) == []
    assert payload["sample"] is False
    assert payload["signal"] == payload["latency"] == payload["summary"] == payload["extraction"] == []
    assert {e["model"] for e in payload["selection"].values()} == {"-"}
    assert any("frozen_json" in m for m in missing)
    assert any("select" in m for m in missing)
    assert any("holdout" in m for m in missing)


def test_sections_come_from_their_result_files(empty_results: Path) -> None:
    (empty_results / "latency_json.json").write_text(
        json.dumps(
            {
                "clean": True,
                "missing": [],
                "partial": [],
                "cells": [
                    {
                        "endpoint": "analyze",
                        "model": "m",
                        "scope": "full",
                        "streaming": True,
                        "backend": "native-metal",
                        "warm": {
                            "n": 30,
                            "totalMs": {"p50": 3000.0, "p95": 3900.0},
                            "ttftMs": {"p50": 900.0, "p95": 1200.0},
                            "signalMs": {"p50": 700.0, "p95": 800.0},
                        },
                    },
                    {"endpoint": "signal", "model": "m", "scope": "lede", "streaming": False,
                     "backend": "native-metal", "warm": {"n": 0, "totalMs": None}},
                ],
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    (empty_results / "stats_json.json").write_text(
        json.dumps(
            {
                "n": 3,
                "per_source": {"bbc": {"bullish": 1, "bearish": 1, "neutral": 1}},
                "pool": {"raw_agreement": 0.96, "cohen_kappa": 0.94},
            }
        ),
        encoding="utf-8",
    )
    payload, missing = rep.build_benchmarks()
    assert rep.validate(payload) == []
    assert payload["latency"] == [
        {
            "endpoint": "analyze",
            "model": "m",
            "scope": "full",
            "streaming": True,
            "backend": "native-metal",
            "n": 30,
            "p50Ms": 3000.0,
            "p95Ms": 3900.0,
            "ttftP50Ms": 900.0,
            "timeToSignalP50Ms": 700.0,
        }
    ]
    assert payload["articles"] == {
        "n": 3,
        "perSource": {"bbc": {"bullish": 1, "bearish": 1, "neutral": 1}},
        "agreement": 0.96,
        "kappa": 0.94,
    }
    assert not any("latency_json" in m for m in missing)


def test_the_committed_results_assemble_into_a_valid_payload() -> None:
    payload, _ = rep.build_benchmarks()
    assert rep.validate(payload) == []
    assert payload["signal"]
    assert payload["extraction"]
    assert payload["summary"]


def test_readme_indexes_every_file_and_flags_undocumented_ones(tmp_path: Path) -> None:
    for rel in ("frozen.json", "grid/signal__m.json", "latency/raw/20261018T000000Z.jsonl", "mystery.csv"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    files = rep.results_files(tmp_path)
    assert files == [
        "frozen.json",
        "grid/signal__m.json",
        "latency/raw/20261018T000000Z.jsonl",
        "mystery.csv",
    ]
    payload = {
        "generatedAt": "t",
        "selection": {e: {"model": "-", "reason": ""} for e in rep.ENDPOINTS},
        "signal": [],
        "latency": [],
        "summary": [],
        "extraction": [],
    }
    readme = rep.render_readme(files, ["latency.json"], payload)
    assert "| `grid/signal__m.json` | per-model /signal grid report" in readme
    assert "| `mystery.csv` | (undocumented" in readme
    assert "  - latency.json" in readme
    docker = rep.describe("latency/raw/docker-cpu-20261018T000000Z.jsonl") or ""
    assert docker.startswith("raw docker-cpu")


@pytest.mark.parametrize(
    ("argv", "module_name"),
    [
        (["latency", "report"], "latency"),
        (["select"], "select"),
        (["holdout", "plan"], "holdout"),
        (["report"], "report"),
    ],
)
def test_p5_commands_are_wired_into_the_cli(argv: list[str], module_name: str) -> None:
    import importlib

    from veritrade_evals import __main__ as cli

    module = importlib.import_module(f"veritrade_evals.{module_name}")
    assert cli.build_parser().parse_args(argv).func is module.run
    assert argv[0] not in cli.PLANNED
