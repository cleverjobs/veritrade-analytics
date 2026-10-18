from __future__ import annotations

import json

import httpx
import pytest

from veritrade.api.contracts import AnalyzeDoneEvent, AnalyzeResponse, SentimentEvent, SummaryDeltaEvent

from .fakes import ENTITY, SUMMARY, FakeOllama, Page, sse_events


@pytest.mark.parametrize(
    ("model", "field_order", "properties"),
    [
        ("fake-label:1b", "sentiment-first", ["score", "entity", "summary"]),
        ("fake-ordinal:1b", "summary-first", ["summary", "entity", "score"]),
    ],
)
async def test_analyze_json(
    api: httpx.AsyncClient,
    ollama: FakeOllama,
    page: Page,
    model: str,
    field_order: str,
    properties: list[str],
) -> None:
    ollama.score = 2
    response = await api.post("/api/analyze", json={"url": page.url, "model": model})
    assert response.status_code == 200, response.text
    body = AnalyzeResponse.model_validate(response.json())
    assert (body.signal.sentiment, body.signal.score, body.signal.policy) == ("bullish", 2, "ordinal")
    assert (body.entity, body.summary) == (ENTITY, SUMMARY)
    assert (body.meta.endpoint, body.meta.scope, body.meta.field_order) == ("analyze", "full", field_order)
    sent = json.loads(ollama.chat.calls.last.request.content)
    assert list(sent["format"]["properties"]) == properties
    assert sent["options"]["num_predict"] == 200


async def test_sentiment_first_stream_sends_the_signal_before_any_text(
    api: httpx.AsyncClient, page: Page
) -> None:
    response = await api.post("/api/analyze/stream", json={"url": page.url, "model": "fake-label:1b"})
    assert response.status_code == 200, response.text
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[:2] == ["meta", "sentiment"]
    assert set(names[2:-1]) == {"summary_delta"}
    assert names[-1] == "done"

    sentiment = SentimentEvent.model_validate(events[1][1])
    done = AnalyzeDoneEvent.model_validate(events[-1][1]).result
    assert sentiment.signal == done.signal
    assert (sentiment.signal.sentiment, sentiment.signal.score) == ("bullish", 1)
    # The entity is generated after the score, so it only arrives with `done`.
    assert (sentiment.entity, done.entity) == (None, ENTITY)
    assert (
        "".join(SummaryDeltaEvent.model_validate(d).text for n, d in events if n == "summary_delta")
        == SUMMARY
    )
    assert done.timings.inference is not None
    assert done.timings.inference.ttft_ms is not None
    assert sentiment.elapsed_ms <= done.timings.inference.ttft_ms <= done.timings.total_ms


async def test_summary_first_stream_sends_the_signal_after_the_text(
    api: httpx.AsyncClient, page: Page
) -> None:
    response = await api.post("/api/analyze/stream", json={"url": page.url, "model": "fake-ordinal:1b"})
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-2:] == ["sentiment", "done"]
    assert set(names[1:-2]) == {"summary_delta"}

    sentiment = SentimentEvent.model_validate(events[-2][1])
    done = AnalyzeDoneEvent.model_validate(events[-1][1]).result
    assert sentiment.entity == ENTITY
    assert sentiment.signal == done.signal
    assert done.meta.field_order == "summary-first"
    assert done.timings.inference is not None
    assert done.timings.inference.ttft_ms is not None
    assert done.timings.inference.ttft_ms <= sentiment.elapsed_ms
