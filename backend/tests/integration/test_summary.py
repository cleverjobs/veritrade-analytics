from __future__ import annotations

import json
import logging

import httpx
import pytest

from veritrade.api.contracts import MetaEvent, SummaryDeltaEvent, SummaryDoneEvent, SummaryResponse

from .fakes import SUMMARY, FakeOllama, Page, sse_events

SECOND_SENTENCE = "Shares rose 4% in early trading."


async def test_summary_json(api: httpx.AsyncClient, ollama: FakeOllama, page: Page) -> None:
    response = await api.post("/api/summary", json={"url": page.url})
    assert response.status_code == 200, response.text
    body = SummaryResponse.model_validate(response.json())
    assert (body.summary, body.truncated) == (SUMMARY, False)
    assert (body.meta.endpoint, body.meta.scope, body.meta.model, body.meta.field_order) == (
        "summary",
        "full",
        "fake-label:1b",
        None,
    )
    assert body.timings.inference is not None
    assert body.timings.inference.ttft_ms is None
    assert body.timings.inference.eval_tokens == 40
    sent = json.loads(ollama.chat.calls.last.request.content)
    assert list(sent["format"]["properties"]) == ["summary"]
    assert page.body_markers[-1] in sent["messages"][-1]["content"]


async def test_summary_stream_event_order(api: httpx.AsyncClient, page: Page) -> None:
    response = await api.post("/api/summary/stream", json={"url": page.url})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"summary_delta"}
    assert len(names) > 3, "the summary should arrive in several deltas"

    meta = MetaEvent.model_validate(events[0][1])
    assert (meta.meta.endpoint, meta.cache.fetch, meta.cache.result) == ("summary", "miss", "miss")
    text = "".join(SummaryDeltaEvent.model_validate(data).text for _, data in events[1:-1])
    done = SummaryDoneEvent.model_validate(events[-1][1]).result
    assert text == done.summary == SUMMARY
    assert done.meta == meta.meta
    inference = done.timings.inference
    assert inference is not None
    assert inference.ttft_ms is not None
    assert meta.fetch_ms <= inference.ttft_ms <= done.timings.total_ms


async def test_summary_cut_at_the_output_limit_keeps_its_complete_sentences(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page, caplog: pytest.LogCaptureFixture
) -> None:
    ollama.summary = f"{SUMMARY} {SECOND_SENTENCE}"
    ollama.cut_after = "Shares ro"
    with caplog.at_level(logging.WARNING, logger="veritrade.inference.adapters"):
        response = await api.post("/api/summary", json={"url": page.url})
    assert response.status_code == 200, response.text
    body = SummaryResponse.model_validate(response.json())
    assert (body.summary, body.truncated) == (SUMMARY, True)
    assert ollama.chat.call_count == 1, "a cut output is salvaged, not regenerated"
    warnings = [r.getMessage() for r in caplog.records if r.name == "veritrade.inference.adapters"]
    assert [w.split(";")[0] for w in warnings] == [
        "summary output from fake-label:1b hit num_predict after 40 eval tokens"
    ]


async def test_summary_stream_cut_at_the_output_limit_flags_the_done_event(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.summary = f"{SUMMARY} {SECOND_SENTENCE}"
    ollama.cut_after = "Shares ro"
    response = await api.post("/api/summary/stream", json={"url": page.url})
    assert response.status_code == 200, response.text
    events = sse_events(response.text)
    assert [name for name, _ in events][-1] == "done"
    text = "".join(SummaryDeltaEvent.model_validate(d).text for n, d in events if n == "summary_delta")
    done = SummaryDoneEvent.model_validate(events[-1][1]).result
    assert text == f"{SUMMARY} Shares ro"
    assert (done.summary, done.truncated) == (SUMMARY, True)

    # The salvaged result is cached with its flag.
    replay = SummaryResponse.model_validate((await api.post("/api/summary", json={"url": page.url})).json())
    assert (replay.meta.cached, replay.summary, replay.truncated) == (True, SUMMARY, True)
