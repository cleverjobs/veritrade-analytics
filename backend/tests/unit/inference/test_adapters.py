from __future__ import annotations

import json
from contextlib import aclosing
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
    chat_options,
    classify_signal,
    fit_input,
    summarize,
    summarize_stream,
)
from veritrade.inference.incremental import EntityClosed, ScoreClosed, SummaryText
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.schemas import LabelOut, OrdinalOut, SummaryOut, ollama_format
from veritrade.inference.types import ChatResult, DecisionResult
from veritrade.models_registry import ModelSpec
from veritrade.prompts import registry as prompts

from .conftest import BASE_URL, chat_body, content_chunks, lp, plain, stream_response

ARTICLE = "Acme Corp raised its full-year outlook after revenue rose 12%."


def _sent(route: respx.Route, index: int = -1) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls[index].request.content)
    return body


def _label_logprobs(
    first: str, alternatives: list[tuple[str, float]], rest: list[str]
) -> list[dict[str, Any]]:
    return [
        *plain(['{"', "label", '":', ' "']),
        lp(first, dict(alternatives)[first], alternatives),
        *plain(rest),
    ]


def test_fit_input() -> None:
    assert fit_input("short text", 100) == "short text"
    text = "word " * 100
    cut = fit_input(text, 10)
    assert len(cut) <= 40
    assert cut.endswith("word")
    assert fit_input("x" * 100, 10) == "x" * 40


def test_chat_options(label_spec: ModelSpec, decision_spec: ModelSpec) -> None:
    assert chat_options(label_spec, "signal") == {"num_ctx": 8192, "num_predict": 16}
    assert chat_options(label_spec, "analyze") == {"num_ctx": 8192, "num_predict": 200}
    assert chat_options(decision_spec, "signal") == {}


async def test_decision_signal(
    client: OllamaClient, mock_ollama: respx.MockRouter, decision_spec: ModelSpec
) -> None:
    answer = {
        "type": "choice",
        "choice": "bullish",
        "probabilities": {"neutral": 0.3, "bullish": 0.65, "bearish": 0.05},
        "confidence": 0.4,
    }
    route = mock_ollama.post("/v1/systemone").respond(
        json={"model": decision_spec.name, "answers": {"sentiment": answer}, "usage": {"input_tokens": 90}}
    )
    prompt = prompts.get("signal", decision_spec.name)
    verdict, raw = await classify_signal(client, decision_spec, prompt, ARTICLE + " filler" * 200)
    assert isinstance(raw, DecisionResult)
    assert (verdict.sentiment, verdict.abstained, verdict.policy) == ("bullish", False, "probability")
    assert verdict.confidence == pytest.approx(0.65)
    sent = _sent(route)
    assert sent["state"].startswith("<article>\nAcme Corp")
    assert len(sent["state"]) <= decision_spec.max_input_tokens * 4 + len("<article>\n\n</article>")
    assert list(sent["questions"]["sentiment"]["criteria"]) == ["neutral", "bullish", "bearish"]


async def test_label_logprob_signal(
    client: OllamaClient, mock_ollama: respx.MockRouter, label_spec: ModelSpec
) -> None:
    logprobs = _label_logprobs("bull", [("bull", 0.7), ("neutral", 0.2), ("bear", 0.1)], ["ish", '"}'])
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"label": "bullish"}', logprobs=logprobs))
    verdict, raw = await classify_signal(client, label_spec, prompts.get("signal", label_spec.name), ARTICLE)
    assert isinstance(raw, ChatResult)
    assert (verdict.sentiment, verdict.policy) == ("bullish", "label-logprob")
    assert verdict.probabilities == pytest.approx({"neutral": 0.2, "bullish": 0.7, "bearish": 0.1})
    sent = _sent(route)
    assert sent["format"] == ollama_format(LabelOut)
    assert (sent["logprobs"], sent["top_logprobs"]) == (True, 20)
    assert sent["options"]["num_predict"] == 16
    assert sent["messages"][-1]["content"] == f"<article>\n{ARTICLE}\n</article>"


async def test_label_collision_falls_back_to_the_score_sign(
    client: OllamaClient, mock_ollama: respx.MockRouter, label_spec: ModelSpec
) -> None:
    colliding = _label_logprobs("b", [("b", 0.9), ("neutral", 0.1)], ["ullish", '"}'])
    sign = [
        *plain(['{"', "score", '":', " "]),
        lp("1", 0.8, [("1", 0.8), ("0", 0.15), ("-", 0.05)]),
        *plain(["}"]),
    ]
    route = mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"label": "bullish"}', logprobs=colliding)),
            httpx.Response(200, json=chat_body('{"score": 1}', logprobs=sign)),
        ]
    )
    verdict, raw = await classify_signal(client, label_spec, prompts.get("signal", label_spec.name), ARTICLE)
    assert route.call_count == 2
    assert _sent(route, 1)["format"] == ollama_format(OrdinalOut)
    assert (verdict.sentiment, verdict.score, verdict.policy) == ("bullish", 1, "label-logprob")
    assert verdict.probabilities == pytest.approx({"neutral": 0.15, "bullish": 0.8, "bearish": 0.05})
    assert isinstance(raw, ChatResult)
    assert raw.timings.eval_tokens == 24
    assert raw.content == '{"score": 1}'


async def test_without_logprobs_the_chain_ends_at_the_ordinal_policy(
    client: OllamaClient, mock_ollama: respx.MockRouter, label_spec: ModelSpec
) -> None:
    mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"label": "bearish"}')),
            httpx.Response(200, json=chat_body('{"score": -2}')),
        ]
    )
    verdict, _ = await classify_signal(client, label_spec, prompts.get("signal", label_spec.name), ARTICLE)
    assert (verdict.sentiment, verdict.score, verdict.policy) == ("bearish", -2, "ordinal")


async def test_ordinal_signal_schema(
    client: OllamaClient, mock_ollama: respx.MockRouter, label_spec: ModelSpec
) -> None:
    spec = label_spec.model_copy(update={"signal_schema": "ordinal", "policy": "ordinal", "ordinal_t": 2})
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"score": 1}'))
    verdict, _ = await classify_signal(client, spec, prompts.get("signal", spec.name), ARTICLE)
    assert _sent(route)["format"] == ollama_format(OrdinalOut)
    assert (verdict.sentiment, verdict.abstained, verdict.score) == ("neutral", True, 1)


async def test_summarize(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"summary": "Acme raised its outlook."}'))
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert result.output.summary == "Acme raised its outlook."
    assert _sent(route)["format"] == ollama_format(SummaryOut)
    assert _sent(route)["options"]["num_predict"] == 160


async def test_summarize_stream(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"summary": "', "Acme ", "raised \\u00e9", 'tc."}']
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces), delay=0.01))
    items = [
        item
        async for item in summarize_stream(
            client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE
        )
    ]
    texts = [i.delta for i in items if isinstance(i, SummaryText)]
    assert "".join(texts) == "Acme raised étc."
    assert len(texts) == 3
    final = items[-1]
    assert isinstance(final, SummaryResult)
    assert final.output.summary == "Acme raised étc."
    assert final.chat.ttft_ms is not None


async def test_analyze(client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec) -> None:
    content = '{"score": 1, "entity": " Acme Corp (ACME) ", "summary": "Acme raised its outlook."}'
    route = mock_ollama.post("/api/chat").respond(json=chat_body(content))
    result = await analyze(client, ordinal_spec, prompts.get("analyze", ordinal_spec.name), ARTICLE)
    assert list(_sent(route)["format"]["properties"]) == ["score", "entity", "summary"]
    assert (result.verdict.sentiment, result.verdict.abstained, result.verdict.policy) == (
        "neutral",
        True,
        "ordinal",
    )
    assert result.entity == "Acme Corp (ACME)"


async def test_analyze_stream_summary_first(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    spec = ordinal_spec.model_copy(update={"field_order": "summary-first", "ordinal_t": 1})
    pieces = ['{"summary": "Acme ', 'cut guidance."', ', "entity": "', 'Acme"', ', "score": -', "2}"]
    route = mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces)))
    items = [item async for item in analyze_stream(client, spec, prompts.get("analyze", spec.name), ARTICLE)]
    assert list(_sent(route)["format"]["properties"]) == ["summary", "entity", "score"]
    assert "summary" in _sent(route)["messages"][0]["content"].split("keys in this order:")[1].split(",")[0]
    kinds = [type(i).__name__ for i in items]
    assert kinds == ["SummaryText", "SummaryText", "EntityClosed", "ScoreClosed", "AnalyzeResult"]
    assert items[3] == ScoreClosed(-2)
    final = items[-1]
    assert isinstance(final, AnalyzeResult)
    assert (final.verdict.sentiment, final.verdict.score) == ("bearish", -2)


async def test_analyze_stream_sentiment_first_emits_score_first(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"score": ', "2", ', "entity": "Acme", ', '"summary": "Up', ' 12%."}']
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces)))
    prompt = prompts.get("analyze", ordinal_spec.name)
    items = [item async for item in analyze_stream(client, ordinal_spec, prompt, ARTICLE)]
    assert items[0] == ScoreClosed(2)
    assert items[1] == EntityClosed("Acme")
    assert isinstance(items[-1], AnalyzeResult)


@pytest.mark.parametrize(
    "pieces",
    [
        ['{"score": 1, "entity": "A", "summary": ""}'],
        ['{"score": 1, "entity": "A", "summ'],
    ],
)
async def test_analyze_stream_rejects_invalid_output(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec, pieces: list[str]
) -> None:
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces)))
    prompt = prompts.get("analyze", ordinal_spec.name)
    with pytest.raises(SchemaViolation):
        async for _ in analyze_stream(client, ordinal_spec, prompt, ARTICLE):
            pass


async def test_stopping_after_the_score_closes_the_upstream_stream(
    mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    # respx records a copy of the response, so capture the one the client actually streams from.
    seen: list[httpx.Response] = []

    async def capture(response: httpx.Response) -> None:
        seen.append(response)

    pieces = ['{"score": ', "-1", ', "entity": "Acme", ', '"summary": "Never read."}']
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces), delay=0.01))
    http = httpx.AsyncClient(base_url=BASE_URL, event_hooks={"response": [capture]})
    client = OllamaClient(BASE_URL, http=http)
    stream = analyze_stream(client, ordinal_spec, prompts.get("analyze", ordinal_spec.name), ARTICLE)
    async with aclosing(stream):
        async for item in stream:
            if isinstance(item, ScoreClosed):
                break
    assert seen[0].is_closed
    await client.aclose()
