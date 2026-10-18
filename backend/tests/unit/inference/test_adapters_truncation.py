"""Output cut at num_predict: salvaged, sentence-cut and flagged instead of retried or rejected."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx

from veritrade.errors import SchemaViolation
from veritrade.inference.adapters import (
    AnalyzeResult,
    SummaryResult,
    analyze,
    analyze_stream,
    cut_to_sentence,
    summarize,
    summarize_stream,
)
from veritrade.inference.incremental import ScoreClosed, SummaryText
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import ModelSpec
from veritrade.prompts import registry as prompts

from .conftest import chat_body, content_chunks, stream_response

ARTICLE = "Acme Corp cut its full-year guidance after demand slowed."


def _cut(content: str) -> dict[str, Any]:
    return chat_body(content, done_reason="length", eval_count=160)


def _sent(route: respx.Route, index: int) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls[index].request.content)
    return body


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Acme rose 5%. Shares fell 2% in ear", "Acme rose 5%."),
        ('Acme hailed "record demand." Shares', 'Acme hailed "record demand."'),
        ("Is it over? Not yet! Analysts", "Is it over? Not yet!"),
        ("Sales grew.\nRevenue rose 3.", "Sales grew."),
        ("Revenue rose 3.", "Revenue rose 3."),
        ("No sentence has ended yet", "No sentence has ended yet"),
        ("  ", ""),
    ],
)
def test_cut_to_sentence(text: str, expected: str) -> None:
    assert cut_to_sentence(text) == expected


@pytest.mark.parametrize(
    ("content", "summary"),
    [
        ('{"summary": "Acme cut guidance. Shares sl', "Acme cut guidance."),
        ('{"summary": "Acme cut guidance. Analysts called it \\', "Acme cut guidance."),
        ('{"summary": "Acme cut guidance. Analysts saw \\u00', "Acme cut guidance."),
        ('{"summary": "Acme \\"slashed\\" guidance after dem', 'Acme "slashed" guidance after dem'),
    ],
    ids=["in-string", "in-escape", "in-unicode-escape", "no-complete-sentence"],
)
async def test_summary_cut_at_the_limit_is_salvaged_without_a_retry(
    client: OllamaClient,
    mock_ollama: respx.MockRouter,
    ordinal_spec: ModelSpec,
    caplog: pytest.LogCaptureFixture,
    content: str,
    summary: str,
) -> None:
    route = mock_ollama.post("/api/chat").respond(json=_cut(content))
    with caplog.at_level(logging.WARNING, logger="veritrade.inference.adapters"):
        result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == (summary, True)
    assert result.chat.content == content
    assert route.call_count == 1
    [record] = caplog.records
    assert record.getMessage().startswith(
        f"summary output from {ordinal_spec.name} hit num_predict after 160 eval tokens"
    )


async def test_summary_cut_before_any_text_is_a_violation(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    mock_ollama.post("/api/chat").respond(json=_cut('{"summary": "  '))
    with pytest.raises(SchemaViolation, match="before writing a summary"):
        await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)


async def test_output_closed_exactly_at_the_limit_is_not_truncated(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    mock_ollama.post("/api/chat").respond(json=_cut('{"summary": "Acme cut guidance."}'))
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == ("Acme cut guidance.", False)


async def test_malformed_output_that_stopped_is_retried_once_on_the_same_budget(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    route = mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"summary": "Acme')),
            httpx.Response(200, json=chat_body('{"summary": "Acme cut guidance."}')),
        ]
    )
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == ("Acme cut guidance.", False)
    assert [_sent(route, i)["options"]["num_predict"] for i in range(2)] == [160, 160]


async def test_a_retry_that_hits_the_limit_is_salvaged(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    route = mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"summary": "Acme')),
            httpx.Response(200, json=_cut('{"summary": "Acme cut guidance. Shares sl')),
        ]
    )
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == ("Acme cut guidance.", True)
    assert route.call_count == 2


async def test_sentiment_first_analyze_cut_in_the_summary_keeps_score_and_entity(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    content = '{"score": -2, "entity": "Acme Corp", "summary": "Acme cut guidance. Shares sl'
    route = mock_ollama.post("/api/chat").respond(json=_cut(content))
    result = await analyze(client, ordinal_spec, prompts.get("analyze", ordinal_spec.name), ARTICLE)
    assert (result.verdict.sentiment, result.verdict.score) == ("bearish", -2)
    assert (result.entity, result.output.summary) == ("Acme Corp", "Acme cut guidance.")
    assert result.truncated is True
    assert route.call_count == 1


async def test_summary_first_analyze_cut_before_the_score_is_a_violation(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    spec = ordinal_spec.model_copy(update={"field_order": "summary-first"})
    content = '{"summary": "Acme cut guidance.", "entity": "Ac'
    route = mock_ollama.post("/api/chat").respond(json=_cut(content))
    with pytest.raises(SchemaViolation, match="before writing the score"):
        await analyze(client, spec, prompts.get("analyze", spec.name), ARTICLE)
    assert route.call_count == 1


async def test_summary_stream_cut_at_the_limit_ends_with_the_salvaged_result(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"summary": "', "Acme cut guidance.", " Shares sl"]
    chunks = content_chunks(pieces, done_reason="length")
    mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks))
    prompt = prompts.get("summary", ordinal_spec.name)
    items = [item async for item in summarize_stream(client, ordinal_spec, prompt, ARTICLE)]
    assert "".join(i.delta for i in items if isinstance(i, SummaryText)) == "Acme cut guidance. Shares sl"
    final = items[-1]
    assert isinstance(final, SummaryResult)
    assert (final.output.summary, final.truncated) == ("Acme cut guidance.", True)
    assert final.chat.ttft_ms is not None


async def test_stream_closed_exactly_at_the_limit_is_not_truncated(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    chunks = content_chunks(['{"summary": "Acme cut guidance."}'], done_reason="length")
    mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks))
    prompt = prompts.get("summary", ordinal_spec.name)
    items = [item async for item in summarize_stream(client, ordinal_spec, prompt, ARTICLE)]
    final = items[-1]
    assert isinstance(final, SummaryResult)
    assert final.truncated is False


async def test_analyze_stream_cut_in_the_summary_ends_with_the_salvaged_result(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"score": ', "-2", ', "entity": "Acme", ', '"summary": "Acme cut guidance.', " Shares sl"]
    chunks = content_chunks(pieces, done_reason="length")
    mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks))
    prompt = prompts.get("analyze", ordinal_spec.name)
    items = [item async for item in analyze_stream(client, ordinal_spec, prompt, ARTICLE)]
    assert items[0] == ScoreClosed(-2)
    final = items[-1]
    assert isinstance(final, AnalyzeResult)
    assert (final.verdict.score, final.entity, final.output.summary, final.truncated) == (
        -2,
        "Acme",
        "Acme cut guidance.",
        True,
    )


async def test_summary_first_analyze_stream_cut_before_the_score_is_a_violation(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    spec = ordinal_spec.model_copy(update={"field_order": "summary-first"})
    chunks = content_chunks(['{"summary": "Acme cut guidance. Shares sl'], done_reason="length")
    mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks))
    with pytest.raises(SchemaViolation, match="before writing the score"):
        async for _ in analyze_stream(client, spec, prompts.get("analyze", spec.name), ARTICLE):
            pass
