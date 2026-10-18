from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Sentiment
from veritrade_evals import grid
from veritrade_evals.datasets import Example
from veritrade_evals.splits import HOLDOUT_SPLITS, Split

BASE_URL = "http://ollama.test"
DECISION = "nimble:9b-q4_K_M"
COMBINED = "ministral-3:3b"
TEXTS: list[tuple[str, Sentiment]] = [
    ("Zentor Mills beats estimates and raises guidance", "bullish"),
    ("Quillon Air upgraded to buy at Harwick", "bullish"),
    ("Orvane Bank posts a surprise loss", "bearish"),
    ("Pellow Foods shares sink after recall", "bearish"),
    ("Marlo Steel sets its annual meeting date", "neutral"),
    ("Tessaly Power names a new finance chief", "neutral"),
]
EXAMPLES = [Example(f"x:{i}", text, label, "tfns", "train") for i, (text, label) in enumerate(TEXTS)]
WEIGHTS: dict[Sentiment, float] = {"bullish": 0.6, "bearish": 0.45, "neutral": 1.95}


def test_grid_enumerates_the_lean_plan() -> None:
    cells = grid.grid_cells(follow_ups=False)
    per_model = Counter(c.model for c in cells)
    assert per_model == {
        "qwen3.5:0.8b-q8_0": 8,
        "clef-flash:9b-q8_0": 4,
        "nimble:9b-q4_K_M": 4,
        "ministral-3:3b": 4,
        "qwen3:4b-instruct-2507-q4_K_M": 4,
        "qwen3.5:4b-q4_K_M": 4,
    }
    assert max(per_model.values()) <= 8
    assert not set(per_model) & set(grid.EXCLUDED)
    assert len({grid.cell_key(c) for c in cells}) == len(cells) == 28
    variants = {m: {c.variant for c in cells if c.model == m} for m in per_model}
    assert variants["qwen3.5:0.8b-q8_0"] == {"label", "ordinal"}
    assert {c.endpoint for c in cells if c.variant == "sentiment-first"} == {"analyze"}
    # Models stay contiguous so each one is loaded once.
    order = [c.model for c in cells]
    assert order == sorted(order, key=list(dict.fromkeys(order)).index)


def test_follow_ups_follow_their_grid_config() -> None:
    cells = grid.grid_cells()
    follow_ups = [c for c in cells if grid.follow_up_of(c.model, c.name) is not None]
    assert len({grid.cell_key(c) for c in cells}) == len(cells) == 28 + len(grid.FOLLOW_UPS)
    assert {(c.model, c.name) for c in follow_ups} == {
        (m, n) for m in grid.ANALYZE_GRID_MODELS for n in ("v5", "v1-summary-first")
    } - {("qwen3:4b-instruct-2507-q4_K_M", "v1-summary-first")}
    by_key = {(c.model, c.name): c for c in cells}
    for cell in follow_ups:
        f = grid.follow_up_of(cell.model, cell.name)
        assert f is not None
        base = by_key[(cell.model, f.against)]
        assert cell.endpoint == base.endpoint == "analyze"
        assert cell.dims["shots"] == str(len(cell.prompt.few_shot)) == "0"
        # Same rubric as the grid config: only the summary instruction or the field order changes.
        assert cell.prompt.system.split("summary:")[0] == base.prompt.system.split("summary:")[0]
    order = [c.model for c in cells]
    assert order == sorted(order, key=list(dict.fromkeys(order)).index)
    first = by_key[("ministral-3:3b", "v1-summary-first")]
    assert (first.variant, first.spec.field_order, first.prompt.version_id) == (
        "summary-first",
        "summary-first",
        "v1",
    )


def test_dims_describe_the_prompt_files() -> None:
    cells = grid.grid_cells(follow_ups=False)
    for model in {c.model for c in cells}:
        own = {c.prompt.version_id: c for c in cells if c.model == model}
        v1 = own["v1"].prompt
        for version, cell in own.items():
            p = cell.prompt
            if cell.variant == "decision":
                bare = all(description is None for _, description in p.criteria)
                assert cell.dims["criteria"] == ("bare" if bare else "described"), (model, version)
                assert (cell.dims["instructions"] == "rubric") == (p.instructions == v1.instructions)
            else:
                assert cell.dims["shots"] == str(len(p.few_shot)), (model, version)
                assert (cell.dims["rubric"] == "A") == (p.system == v1.system), (model, version)


def test_grid_reads_dev300_only() -> None:
    assert grid.GRID_SPLIT == "dev300"
    assert grid.GRID_SPLIT not in HOLDOUT_SPLITS


def test_unknown_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in the grid"):
        grid.grid_cells(["tev1:0.8b-q8_0"])
    assert {c.model for c in grid.grid_cells([COMBINED])} == {COMBINED}


def test_cli_list_and_parent_parser_hooks(capsys: pytest.CaptureFixture[str]) -> None:
    assert grid.main(["list", "--model", DECISION]) == 0
    out = capsys.readouterr().out
    assert "4 configs: nimble:9b-q4_K_M=4" in out
    assert "excluded: tev1" in out

    parent = argparse.ArgumentParser()
    grid.add_parser(parent.add_subparsers(dest="command", required=True))
    args = parent.parse_args(["grid", "analyze", "--n-boot", "10"])
    assert args.grid_func is grid.cmd_analyze
    assert args.n_boot == 10
    with pytest.raises(SystemExit):
        parent.parse_args(["grid", "run", "--bogus"])


# --- end to end on synthetic data ------------------------------------------------------------


def _score(text: str) -> int:
    return (
        2 if "beats" in text or "upgraded" in text else -2 if "loss" in text else -1 if "sink" in text else 0
    )


def _systemone(request: httpx.Request) -> httpx.Response:
    state = json.loads(request.content)["state"]
    score = _score(state)
    label: Sentiment = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    probs = {s: (0.9 if s == label else 0.05) for s in ("neutral", "bullish", "bearish")}
    answer = {"type": "choice", "choice": label, "probabilities": probs, "confidence": 0.8}
    return httpx.Response(200, json={"model": DECISION, "answers": {"sentiment": answer}, "usage": {}})


def _chat(request: httpx.Request) -> httpx.Response:
    text = json.loads(request.content)["messages"][-1]["content"]
    pieces = ['{"score": ', str(_score(text)), ', "entity": "Acme", ', '"summary": "Acme news."}']
    chunks = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
    chunks.append({"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 9})
    body = "".join(json.dumps(c) + "\n" for c in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        router.get("/api/version").respond(json={"version": "0.40.6"})
        router.get("/api/ps").respond(json={"models": []})
        tags = [{"name": m, "model": m, "digest": d * 64} for m, d in ((DECISION, "a"), (COMBINED, "b"))]
        router.get("/api/tags").respond(json={"models": tags})
        router.post("/v1/systemone", name="decide").mock(side_effect=_systemone)
        router.post("/api/chat", name="chat").mock(side_effect=_chat)
        yield router


@pytest.fixture
async def client() -> AsyncIterator[OllamaClient]:
    ollama = OllamaClient(BASE_URL, timeout=2.0, keep_alive="5m")
    yield ollama
    await ollama.aclose()


@pytest.fixture
def synthetic_split(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = {"bullish": 0.2, "bearish": 0.15, "neutral": 0.65}
    monkeypatch.setattr(grid, "load_split", lambda name: Split("dev300", tuple(EXAMPLES), WEIGHTS))
    monkeypatch.setattr(
        grid, "read_split_file", lambda name: {"class_weights": WEIGHTS, "train_prior": prior}
    )


async def test_run_then_analyze_end_to_end(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path, synthetic_split: None
) -> None:
    cells = grid.grid_cells([DECISION, COMBINED])
    manifest_path, preds, out = tmp_path / "manifest.json", tmp_path / "preds", tmp_path / "out"
    log = io.StringIO()
    manifest = await grid.run_grid(
        client, cells, EXAMPLES, concurrency=2, preds_root=preds, manifest_path=manifest_path, log=log
    )
    assert len(manifest["cells"]) == 10
    assert {e["done"] for e in manifest["cells"].values()} == {6}
    assert manifest["ollama_version"] == "0.40.6"
    decide = mock_ollama.routes["decide"]
    calls_after_first_run = decide.call_count
    assert calls_after_first_run == 4 * 6 + 1  # 4 configs x 6 examples + 1 warm-up

    chat = mock_ollama.routes["chat"]
    chat_calls = chat.call_count
    await grid.run_grid(
        client, cells, EXAMPLES, concurrency=2, preds_root=preds, manifest_path=manifest_path, log=log
    )
    # Fully cached: no warm-up, no calls.
    assert (decide.call_count, chat.call_count) == (calls_after_first_run, chat_calls)

    more = [*EXAMPLES, Example("x:6", "Halden Ore wins a mining permit", "bullish", "tfns", "train")]
    await grid.run_grid(
        client, cells, more, concurrency=2, preds_root=preds, manifest_path=manifest_path, log=log
    )
    assert decide.call_count == calls_after_first_run + 1 + 4  # warm-up + the new example per config

    paths = grid.analyze_grid(n_boot=50, preds_root=preds, manifest_path=manifest_path, out_dir=out, log=log)
    names = sorted(p.name for p in paths)
    assert names == [
        "analyze__ministral-3_3b.json",
        "error_patterns.json",
        "signal__nimble_9b-q4_K_M.json",
        "summary.md",
    ]
    report = json.loads((out / "signal__nimble_9b-q4_K_M.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in report["configs"]] == ["v1", "v2", "v3", "v4"]
    assert report["best"]["config"] == "v1"
    assert report["best"]["metrics"]["f05_macro"] == 1.0
    assert report["best"]["cis"]["f05_macro"]["low"] == 1.0
    assert report["configs"][0]["policies"]["probability"]["sweep"][0]["threshold"] == 0.0
    combined = json.loads((out / "analyze__ministral-3_3b.json").read_text(encoding="utf-8"))
    # One gold-bearish tweet scores -1, so |s| >= 2 abstains on it; t=1 keeps it and wins.
    assert combined["best"]["policy"] == "ordinal"
    assert combined["best"]["threshold"] == 1.0
    assert combined["configs"][0]["diagnostics"]["score_histogram"] == {
        "-2": 1,
        "-1": 1,
        "0": 2,
        "1": 0,
        "2": 2,
    }
    assert [f["config"] for f in combined["follow_ups"]] == ["v5", "v1-summary-first"]
    assert {f["against"]["config"] for f in combined["follow_ups"]} == {"v1"}
    assert combined["best"]["config"] == "v1"
    assert "directional_tau" in report
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Best config per model" in summary
    assert "## Follow-ups after the grid" in summary
    assert "## directional_tau vs the plain rule" in summary
    assert "`nimble:9b-q4_K_M`" in summary
    for path in [*paths, manifest_path]:
        written = path.read_text(encoding="utf-8")
        assert all(text not in written for text, _ in TEXTS), path.name
    assert "skip signal/qwen3.5:0.8b-q8_0/v1-label: not run yet" in log.getvalue()


async def test_analyze_skips_cells_whose_prompt_changed(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path, synthetic_split: None
) -> None:
    cells = grid.grid_cells([DECISION])
    manifest_path = tmp_path / "manifest.json"
    await grid.run_grid(
        client, cells, EXAMPLES, concurrency=1, preds_root=tmp_path, manifest_path=manifest_path
    )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"][f"signal/{DECISION}/v2"]["prompt_sha"] = "0" * 64
    grid.write_manifest(manifest, manifest_path)
    log = io.StringIO()
    grid.analyze_grid(
        n_boot=10, preds_root=tmp_path, manifest_path=manifest_path, out_dir=tmp_path / "o", log=log
    )
    assert f"skip signal/{DECISION}/v2: prompt or config changed" in log.getvalue()
    report = json.loads((tmp_path / "o" / "signal__nimble_9b-q4_K_M.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in report["configs"]] == ["v1", "v3", "v4"]
