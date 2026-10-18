from __future__ import annotations

import json

import httpx
import pytest

from veritrade.api.contracts import (
    AnalyzeDoneEvent,
    AnalyzeResponse,
    ErrorResponse,
    SentimentEvent,
    SummaryDeltaEvent,
)

from .fakes import ENTITY, SUMMARY, FakeOllama, Page, sse_events

SECOND_SENTENCE = "Shares rose 4% in early trading."


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
    assert (body.entity, body.summary, body.truncated) == (ENTITY, SUMMARY, False)
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


async def test_sentiment_first_cut_at_the_output_limit_keeps_the_signal(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.summary = f"{SUMMARY} {SECOND_SENTENCE}"
    ollama.cut_after = "Shares ro"
    response = await api.post("/api/analyze", json={"url": page.url, "model": "fake-label:1b"})
    assert response.status_code == 200, response.text
    body = AnalyzeResponse.model_validate(response.json())
    assert (body.signal.sentiment, body.signal.score) == ("bullish", 1)
    assert (body.entity, body.summary, body.truncated) == (ENTITY, SUMMARY, True)
    assert ollama.chat.call_count == 1


async def test_summary_first_cut_before_the_score_is_a_schema_violation(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.cut_after = "Acme Corp raised"
    response = await api.post("/api/analyze", json={"url": page.url, "model": "fake-ordinal:1b"})
    assert response.status_code == 502, response.text
    assert ErrorResponse.model_validate(response.json()).error.code == "schema_violation"
    assert ollama.chat.call_count == 1


async def test_analyze_stream_cut_at_the_output_limit_flags_the_done_event(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.summary = f"{SUMMARY} {SECOND_SENTENCE}"
    ollama.cut_after = "Shares ro"
    response = await api.post("/api/analyze/stream", json={"url": page.url, "model": "fake-label:1b"})
    assert response.status_code == 200, response.text
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[:2] == ["meta", "sentiment"]
    assert names[-1] == "done"
    text = "".join(SummaryDeltaEvent.model_validate(d).text for n, d in events if n == "summary_delta")
    done = AnalyzeDoneEvent.model_validate(events[-1][1]).result
    assert text == f"{SUMMARY} Shares ro"
    assert (done.signal.score, done.entity, done.summary, done.truncated) == (1, ENTITY, SUMMARY, True)
