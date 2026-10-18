from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from veritrade.inference.types import Sentiment
from veritrade_evals import latency as lat
from veritrade_evals.articles import ArticleExample
from veritrade_evals.latency import Cell

OLLAMA = "http://ollama.test:11434"
SIGNAL = Cell("signal", "clef-flash:9b-q8_0", "lede", False)
SUMMARY = Cell("summary", "qwen3.5:2b-q4_K_M", "full", True)
ANALYZE = Cell("analyze", "ministral-3:3b", "full", True)


def timings(total: float, *, fetch: float = 1.0, ttft: float | None = None) -> dict[str, Any]:
    return {
        "total_ms": total,
        "fetch_ms": fetch,
        "parse_ms": 2.0,
        "extract_ms": 1.0,
        "inference": {"wall_ms": total - 4, "ttft_ms": ttft, "prompt_eval_ms": 100.0, "eval_ms": 50.0},
        "cache": {"fetch": "bypass", "result": "bypass"},
    }


def sse(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def sample(cell: Cell, phase: lat.Phase, article: str, total: float, **extra: Any) -> lat.Sample:
    return lat.make_sample(
        cell,
        session="s",
        backend="native-metal",
        phase=phase,
        article=article,
        ok=True,
        total_ms=total,
        **extra,
    )


# --- parsing ---------------------------------------------------------------------------------


def test_json_signal_reply_takes_server_timings_and_signal_time_is_the_total() -> None:
    body = json.dumps({"meta": {"input_chars": 640, "title": "never kept"}, "timings": timings(320.0)})
    parsed = lat.parse_reply(SIGNAL, 200, body)
    assert parsed["ok"]
    assert parsed["total_ms"] == 320.0
    assert parsed["signal_ms"] == 320.0
    assert parsed["input_chars"] == 640
    assert "title" not in lat.make_sample(SIGNAL, **parsed)


def test_analyze_stream_reply_reads_ttft_signal_time_and_event_order() -> None:
    result = {"timings": timings(2400.0, ttft=900.0), "truncated": False}
    body = sse(
        ("meta", {}),
        ("sentiment", {"elapsed_ms": 610.0}),
        ("summary_delta", {"text": "x"}),
        ("summary_delta", {"text": "y"}),
        ("done", {"result": result}),
    )
    parsed = lat.parse_reply(ANALYZE, 200, body)
    assert parsed["ttft_ms"] == 900.0
    assert parsed["signal_ms"] == 610.0
    assert parsed["deltas"] == 2
    assert parsed["order_ok"] is True


def test_stream_order_violations_and_errors_are_recorded() -> None:
    late = sse(
        ("meta", {}), ("done", {"result": {"timings": timings(10.0)}}), ("sentiment", {"elapsed_ms": 5})
    )
    assert lat.parse_reply(ANALYZE, 200, late)["order_ok"] is False
    failed = sse(("meta", {}), ("error", {"error": {"code": "inference_timeout"}}))
    assert lat.parse_reply(SUMMARY, 200, failed) == {"ok": False, "error": "inference_timeout"}
    assert lat.parse_reply(SUMMARY, 503, json.dumps({"error": {"code": "llm_unreachable"}}))["error"] == (
        "llm_unreachable"
    )
    assert lat.parse_reply(SUMMARY, 200, sse(("meta", {})))["error"] == "no_done_event"


# --- aggregation and budgets -----------------------------------------------------------------


def test_aggregate_excludes_fetch_and_marks_the_signal_lede_budget() -> None:
    warm = [sample(SIGNAL, "warm", f"a{i}", 400.0 + 10 * i, fetch_ms=50.0) for i in range(5)]
    agg = lat.aggregate(warm)
    assert agg["n"] == 5
    assert agg["totalMs"]["p50"] == 370.0
    assert agg["totalInclFetchMs"]["p50"] == 420.0
    budget = lat.budget_status(SIGNAL, agg)
    assert budget["status"] == "met"
    assert budget["utilisation"] == pytest.approx(0.74)
    slow = lat.aggregate([sample(SIGNAL, "warm", "a", 700.0)])
    assert lat.budget_status(SIGNAL, slow)["status"] == "missed"
    assert lat.budget_status(Cell("signal", "m", "full", False), slow)["status"] == "n/a"
    assert lat.budget_status(SIGNAL, lat.aggregate([]))["status"] == "unmeasured"


def test_stream_budget_needs_both_ttft_and_complete_p50() -> None:
    fast = lat.aggregate([sample(SUMMARY, "warm", "a", 3000.0, ttft_ms=1200.0)])
    assert lat.budget_status(SUMMARY, fast)["status"] == "met"
    late_first_token = lat.aggregate([sample(SUMMARY, "warm", "a", 3000.0, ttft_ms=1800.0)])
    budget = lat.budget_status(SUMMARY, late_first_token)
    assert budget["status"] == "missed"
    assert budget["utilisation"] == pytest.approx(1.2)
    no_ttft = lat.aggregate([sample(SUMMARY, "warm", "a", 3000.0)])
    assert lat.budget_status(SUMMARY, no_ttft)["status"] == "unmeasured"
    json_cell = Cell("summary", SUMMARY.model, "full", False)
    assert [c["metric"] for c in lat.budget_status(json_cell, fast)["checks"]] == ["complete p50"]


def test_report_flags_unclean_missing_and_partial_cells() -> None:
    samples = [sample(SIGNAL, "warm", f"a{i}", 300.0) for i in range(lat.WARM_N)]
    samples += [sample(SIGNAL, "cold", f"a{i}", 2000.0) for i in range(lat.COLD_N)]
    samples.append(sample(SUMMARY, "warm", "a0", 2000.0, ttft_ms=800.0))
    report = lat.build_report({"session": "s", "ok": False}, samples, [])
    cells = {c["key"]: c for c in report["cells"]}
    assert not report["clean"]
    assert not report["complete"]
    assert not cells[SIGNAL.key]["partial"]
    assert cells[SIGNAL.key]["budget"]["status"] == "met"
    assert SUMMARY.key in report["partial"]
    assert ANALYZE.key in report["missing"]
    assert "Not a clean run" in lat.render(report)


def test_raw_store_resumes_and_keeps_only_known_fields(tmp_path: Path) -> None:
    store = lat.RawStore(tmp_path / "s.jsonl")
    store.append({**sample(SIGNAL, "warm", "a1", 300.0), "summary": "never written"})
    again = lat.RawStore(tmp_path / "s.jsonl")
    assert again.has(SIGNAL, "warm", "a1")
    assert not again.has(SIGNAL, "cold", "a1")
    assert set(again.samples()[0]) == set(lat.SAMPLE_FIELDS)


def test_docker_cells_follow_the_bundled_defaults(tmp_path: Path) -> None:
    env = tmp_path / "bundled.env"
    env.write_text(
        "# comment\nVT_DEFAULT_MODEL_SIGNAL=qwen3.5:0.8b-q8_0\nVT_DEFAULT_MODEL_SUMMARY=qwen3.5:2b-q4_K_M\n"
        "VT_DEFAULT_MODEL_ANALYZE=ministral-3:3b\n",
        encoding="utf-8",
    )
    keys = [c.key for c in lat.docker_cells(None, lat.bundled_defaults(env))]
    assert keys == [
        "signal/qwen3.5:0.8b-q8_0/lede/json",
        "summary/qwen3.5:2b-q4_K_M/full/stream",
        "summary/qwen3.5:2b-q4_K_M/full/json",
        "analyze/ministral-3:3b/full/stream",
        "analyze/ministral-3:3b/full/json",
    ]
    with pytest.raises(lat.LatencyError):
        lat.select_cells(lat.matrix(), ["not-a-model"], None)


def test_matrix_models_are_allowlisted_for_their_endpoints() -> None:
    from veritrade.models_registry import allowlist

    roster = allowlist()
    assert all(roster[c.model].serves(c.endpoint) for c in lat.matrix())
    assert len(lat.matrix()) == 16


# --- the suite on a mocked Ollama ------------------------------------------------------------


def _done(**extra: Any) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "load_duration": 5_000_000,
        "prompt_eval_count": 512,
        "prompt_eval_duration": 120_000_000,
        "eval_count": 12,
        "eval_duration": 60_000_000,
    } | extra


def _chat(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    fields = list(body["format"]["properties"])
    content = json.dumps(
        {f: {"summary": "Acme Corp beat estimates.", "score": 1}.get(f, "Acme") for f in fields}
    )
    if not body["stream"]:
        return httpx.Response(200, json=_done(message={"role": "assistant", "content": content}))
    pieces = [content[i : i + 5] for i in range(0, len(content), 5)]
    lines = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces] + [_done()]
    return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines).encode())


def _systemone(request: httpx.Request) -> httpx.Response:
    probs: dict[Sentiment, float] = {"neutral": 0.05, "bullish": 0.9, "bearish": 0.05}
    answer = {"type": "choice", "choice": "bullish", "probabilities": probs, "confidence": 0.9}
    return httpx.Response(200, json={"answers": {"sentiment": answer}, "usage": {"input_tokens": 120}})


@pytest.fixture
def ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        tags = [{"name": SIGNAL.model, "size": 1}, {"name": SUMMARY.model, "size": 2}]
        router.get(f"{OLLAMA}/api/tags").respond(200, json={"models": tags})
        router.get(f"{OLLAMA}/api/ps").respond(200, json={"models": []})
        router.post(f"{OLLAMA}/api/generate", name="unload").respond(200, json={"done": True})
        router.post(f"{OLLAMA}/api/chat").mock(side_effect=_chat)
        router.post(f"{OLLAMA}/v1/systemone").mock(side_effect=_systemone)
        yield router


def _article(i: int) -> ArticleExample:
    return ArticleExample(
        id=f"art-{i}",
        source="synthetic",
        url=f"https://news.example/{i}",
        label="neutral",
        score=0.0,
        entity="Ostrava Logistics",
        ticker=None,
        key_facts=(),
        gold_text="",
        html_path=Path("unused.html"),
        split="dev",
    )


async def test_native_suite_measures_cold_and_warm_through_the_app(
    ollama: respx.MockRouter, tmp_path: Path
) -> None:
    articles = [_article(i) for i in range(2)]
    pages = {a.url: (a.url, lat.WARMUP_HTML) for a in articles}
    added = await lat.run_native(
        [SIGNAL, SUMMARY],
        articles,
        pages,
        session="test",
        settings=lat.serving_settings(OLLAMA),
        cold_n=1,
        shared=False,
        raw_dir=tmp_path,
    )
    samples = lat.RawStore(tmp_path / "test.jsonl").samples()
    assert added == len(samples) == 6
    assert all(s["ok"] for s in samples), [s["error"] for s in samples]
    stream = [s for s in samples if s["cell"] == SUMMARY.key and s["phase"] == "warm"]
    assert all(s["ttft_ms"] is not None and s["order_ok"] for s in stream)
    # Cold samples unload first; every warm phase reloads through the unmeasured warm-up page.
    assert ollama.routes["unload"].call_count == 4
    assert {s["article"] for s in samples} == {"art-0", "art-1"}
    again = await lat.run_native(
        [SIGNAL, SUMMARY],
        articles,
        pages,
        session="test",
        settings=lat.serving_settings(OLLAMA),
        cold_n=1,
        shared=False,
        raw_dir=tmp_path,
    )
    assert again == 0


def test_docker_rows_come_from_their_own_samples_and_carry_no_native_budget() -> None:
    cell = Cell("summary", "qwen3.5:0.8b-q8_0", "full", True)
    docker = [
        lat.make_sample(
            cell, session="d", backend="docker-cpu", phase="warm", article=f"a{i}", ok=True, total_ms=9000.0
        )
        for i in range(3)
    ]
    report = lat.build_report({"session": "s", "ok": True}, [], docker, docker_session="docker-cpu-s")
    rows = [c for c in report["cells"] if c["backend"] == "docker-cpu"]
    assert [r["key"] for r in rows] == [cell.key]
    assert rows[0]["warm"]["n"] == 3
    assert rows[0]["budget"]["status"] == "n/a"
