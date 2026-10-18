from __future__ import annotations

import asyncio

import httpx
import respx

from veritrade.api.contracts import (
    AnalyzeDoneEvent,
    AnalyzeResponse,
    CacheInfo,
    MetaEvent,
    SentimentEvent,
    SignalResponse,
    SummaryResponse,
)

from .fakes import SUMMARY, FakeOllama, Page, sse_events


async def test_split_mode_shares_one_fetch(
    api: httpx.AsyncClient, router: respx.MockRouter, ollama: FakeOllama, page: Page
) -> None:
    async def slow_page(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, html=page.html)

    route = router.routes[page.source]
    route.side_effect = slow_page
    signal, stream = await asyncio.gather(
        api.post("/api/signal", json={"url": page.url}),
        api.post("/api/summary/stream", json={"url": page.url}),
    )
    assert route.call_count == 1
    signal_fetch = SignalResponse.model_validate(signal.json()).timings.cache.fetch
    stream_fetch = MetaEvent.model_validate(sse_events(stream.text)[0][1]).cache.fetch
    assert sorted([signal_fetch, stream_fetch]) == ["joined", "miss"]


async def test_second_endpoint_reuses_the_cached_page(
    api: httpx.AsyncClient, router: respx.MockRouter, page: Page
) -> None:
    await api.post("/api/signal", json={"url": page.url})
    response = await api.post("/api/summary", json={"url": page.url + "?utm_source=newsletter"})
    timings = SummaryResponse.model_validate(response.json()).timings
    assert timings.cache == CacheInfo(fetch="hit", result="miss")
    assert timings.consent_ms is None
    assert router.routes[page.source].call_count == 1


async def test_identical_request_is_served_from_the_result_cache(
    api: httpx.AsyncClient, router: respx.MockRouter, ollama: FakeOllama, page: Page
) -> None:
    first = SignalResponse.model_validate((await api.post("/api/signal", json={"url": page.url})).json())
    calls = ollama.inference_calls
    second = SignalResponse.model_validate((await api.post("/api/signal", json={"url": page.url})).json())
    assert ollama.inference_calls == calls
    assert router.routes[page.source].call_count == 1
    assert second.signal == first.signal
    assert second.meta == first.meta.model_copy(update={"cached": True})
    assert second.timings.cache == CacheInfo(fetch="hit", result="hit")
    assert second.timings.inference is None
    assert second.timings.total_ms < first.timings.total_ms


async def test_result_key_covers_model_scope_and_extractor(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    await api.post("/api/signal", json={"url": page.url})
    for variant in ({"scope": "full"}, {"model": "fake-decision:1b"}, {"extractor": "trafilatura"}):
        calls = ollama.inference_calls
        response = await api.post("/api/signal", json={"url": page.url, **variant})
        assert SignalResponse.model_validate(response.json()).timings.cache.result == "miss", variant
        assert ollama.inference_calls == calls + 1


async def test_no_cache_bypasses_both_caches(
    api: httpx.AsyncClient, router: respx.MockRouter, ollama: FakeOllama, page: Page
) -> None:
    await api.post("/api/signal", json={"url": page.url})
    calls = ollama.inference_calls
    response = await api.post("/api/signal", json={"url": page.url, "no_cache": True})
    body = SignalResponse.model_validate(response.json())
    assert body.timings.cache == CacheInfo(fetch="bypass", result="bypass")
    assert body.meta.cached is False
    assert ollama.inference_calls == calls + 1
    assert router.routes[page.source].call_count == 2


async def test_stream_replays_a_cached_result(api: httpx.AsyncClient, ollama: FakeOllama, page: Page) -> None:
    stored = AnalyzeResponse.model_validate((await api.post("/api/analyze", json={"url": page.url})).json())
    calls = ollama.inference_calls
    response = await api.post("/api/analyze/stream", json={"url": page.url})
    events = sse_events(response.text)
    assert [name for name, _ in events] == ["meta", "sentiment", "summary_delta", "done"]
    assert ollama.inference_calls == calls
    meta = MetaEvent.model_validate(events[0][1])
    assert meta.meta.cached is True
    assert meta.cache == CacheInfo(fetch="hit", result="hit")
    assert SentimentEvent.model_validate(events[1][1]).signal == stored.signal
    assert events[2][1] == {"text": SUMMARY}
    done = AnalyzeDoneEvent.model_validate(events[3][1]).result
    assert (done.summary, done.entity, done.meta.cached) == (stored.summary, stored.entity, True)


async def test_a_completed_stream_fills_the_result_cache(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    await api.post("/api/summary/stream", json={"url": page.url})
    calls = ollama.inference_calls
    response = await api.post("/api/summary", json={"url": page.url})
    assert SummaryResponse.model_validate(response.json()).meta.cached is True
    assert ollama.inference_calls == calls
