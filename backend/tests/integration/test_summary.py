from __future__ import annotations

import json

import httpx

from veritrade.api.contracts import MetaEvent, SummaryDeltaEvent, SummaryDoneEvent, SummaryResponse

from .fakes import SUMMARY, FakeOllama, Page, sse_events


async def test_summary_json(api: httpx.AsyncClient, ollama: FakeOllama, page: Page) -> None:
    response = await api.post("/api/summary", json={"url": page.url})
    assert response.status_code == 200, response.text
    body = SummaryResponse.model_validate(response.json())
    assert body.summary == SUMMARY
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
