from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from veritrade.config import Settings
from veritrade.errors import InferenceTimeout, LlmUnreachable, ModelUnavailable, SchemaViolation
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.schemas import LabelOut, SummaryOut, ollama_format
from veritrade.inference.types import ChatResult

from .conftest import chat_body, content_chunks, lp, stream_response

MESSAGES = [{"role": "user", "content": "hi"}]


def _sent(route: respx.Route, index: int = -1) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls[index].request.content)
    return body


async def test_chat_sends_fixed_decoding_and_parses_the_response(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    logprobs = [lp('{"', 0.9, [('{"', 0.9), ("{", 0.1)])]
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"label": "neutral"}', logprobs=logprobs))
    result = await client.chat(
        "qwen3.5:0.8b-q8_0",
        MESSAGES,
        format=ollama_format(LabelOut),
        options={"num_ctx": 8192, "num_predict": 16, "temperature": 0.7},
        logprobs=True,
        top_logprobs=20,
    )
    sent = _sent(route)
    assert sent["think"] is False
    assert sent["stream"] is False
    assert sent["keep_alive"] == "5m"
    assert sent["options"] == {
        "num_ctx": 8192,
        "num_predict": 16,
        "temperature": 0,
        "seed": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repeat_penalty": 1,
    }
    assert sent["format"] == ollama_format(LabelOut)
    assert (sent["logprobs"], sent["top_logprobs"]) == (True, 20)
    assert result.content == '{"label": "neutral"}'
    assert result.timings.total_ms == pytest.approx(250.0)
    assert result.timings.load_ms == pytest.approx(10.0)
    assert result.timings.prompt_eval_ms == pytest.approx(80.0)
    assert result.timings.eval_ms == pytest.approx(150.0)
    assert (result.timings.prompt_tokens, result.timings.eval_tokens) == (120, 12)
    assert result.logprobs is not None
    assert result.logprobs[0].token == '{"'
    assert [t for t, _ in result.logprobs[0].top] == ['{"', "{"]
    assert result.ttft_ms is None
    assert result.wall_ms > 0


async def test_chat_without_logprobs_omits_the_fields(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    route = mock_ollama.post("/api/chat").respond(json=chat_body("hello"))
    result = await client.chat("m", MESSAGES)
    assert "logprobs" not in _sent(route)
    assert "format" not in _sent(route)
    assert result.logprobs is None


async def test_chat_json_validates(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.post("/api/chat").respond(json=chat_body('{"summary": "Acme rose 5%."}'))
    out, chat = await client.chat_json("m", MESSAGES, SummaryOut, options={"num_predict": 50})
    assert out == SummaryOut(summary="Acme rose 5%.")
    assert chat.content == '{"summary": "Acme rose 5%."}'


async def test_chat_json_retries_once_with_a_bigger_budget_when_cut(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    route = mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"summary": "Acme ro', done_reason="length")),
            httpx.Response(200, json=chat_body('{"summary": "Acme rose 5%."}')),
        ]
    )
    out, _ = await client.chat_json("m", MESSAGES, SummaryOut, options={"num_predict": 50})
    assert out.summary == "Acme rose 5%."
    assert [_sent(route, i)["options"]["num_predict"] for i in range(2)] == [50, 100]


async def test_chat_json_gives_up_after_one_retry(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"label": "positive"}'))
    with pytest.raises(SchemaViolation, match="twice") as info:
        await client.chat_json("m", MESSAGES, LabelOut)
    assert route.call_count == 2
    assert info.value.http_status == 502


async def test_chat_stream_yields_deltas_then_result(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    pieces = ['{"', "summary", '": "', "Acme", " rose", '"}']
    chunks = content_chunks(pieces)
    chunks[1]["logprobs"] = [lp("summary", 0.9)]
    chunks[3]["logprobs"] = [lp("Acme", 0.8)]
    route = mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks, delay=0.03))
    items = [item async for item in client.chat_stream("m", MESSAGES, format=ollama_format(SummaryOut))]
    assert items[:-1] == pieces
    final = items[-1]
    assert isinstance(final, ChatResult)
    assert final.content == "".join(pieces)
    assert final.ttft_ms is not None
    assert final.ttft_ms >= 25
    assert final.wall_ms >= final.ttft_ms
    assert final.timings.eval_tokens == len(pieces)
    assert [t.token for t in final.logprobs or ()] == ["summary", "Acme"]
    assert _sent(route)["stream"] is True


async def test_chat_stream_ignores_empty_deltas_for_ttft(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(["", "", "x"])))
    items = [item async for item in client.chat_stream("m", MESSAGES)]
    assert items[:-1] == ["x"]


@pytest.mark.parametrize(
    ("chunks", "error", "message"),
    [
        ([{"error": "model runner has unexpectedly stopped"}], ModelUnavailable, "unexpectedly stopped"),
        (content_chunks(["{"])[:-1], SchemaViolation, "without a done message"),
    ],
)
async def test_chat_stream_failures(
    client: OllamaClient,
    mock_ollama: respx.MockRouter,
    chunks: list[dict[str, Any]],
    error: type[Exception],
    message: str,
) -> None:
    mock_ollama.post("/api/chat").mock(return_value=stream_response(chunks))
    with pytest.raises(error, match=message):
        async for _ in client.chat_stream("m", MESSAGES):
            pass


async def test_chat_stream_http_error(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.post("/api/chat").respond(404, json={"error": "model 'x:1b' not found"})
    with pytest.raises(ModelUnavailable, match="not found"):
        async for _ in client.chat_stream("x:1b", MESSAGES):
            pass


@pytest.mark.parametrize(
    ("side_effect", "error"),
    [
        (httpx.ConnectError("refused"), LlmUnreachable),
        (httpx.ConnectTimeout("slow connect"), LlmUnreachable),
        (httpx.ReadTimeout("slow"), InferenceTimeout),
        (httpx.RemoteProtocolError("reset"), LlmUnreachable),
    ],
)
async def test_transport_errors_are_typed(
    client: OllamaClient, mock_ollama: respx.MockRouter, side_effect: Exception, error: type[Exception]
) -> None:
    mock_ollama.post("/api/chat").mock(side_effect=side_effect)
    with pytest.raises(error):
        await client.chat("m", MESSAGES)
    with pytest.raises(error):
        async for _ in client.chat_stream("m", MESSAGES):
            pass


async def test_unreachable_has_a_hint(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.get("/api/version").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LlmUnreachable) as info:
        await client.version()
    assert info.value.http_status == 503
    assert info.value.hint
    assert "OLLAMA_BASE_URL" in info.value.hint


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (404, {"error": "model 'x:1b' not found"}, "not available"),
        (500, {"error": "model 'x:1b' not found, try pulling it first"}, "not available"),
        (400, {"error": '"tev1:0.8b-q8_0" does not support chat'}, "does not support chat"),
        (500, "plain text failure", "plain text failure"),
    ],
)
async def test_http_errors_map_to_model_unavailable(
    client: OllamaClient, mock_ollama: respx.MockRouter, status: int, body: Any, message: str
) -> None:
    kwargs = {"json": body} if isinstance(body, dict) else {"text": body}
    mock_ollama.post("/api/chat").respond(status, **kwargs)
    with pytest.raises(ModelUnavailable, match=message) as info:
        await client.chat("x:1b", MESSAGES)
    assert info.value.upstream_status == status
    if status == 404:
        assert info.value.hint
        assert "ollama pull x:1b" in info.value.hint


async def test_non_json_success_is_a_schema_violation(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    mock_ollama.get("/api/tags").respond(200, text="<html>")
    with pytest.raises(SchemaViolation):
        await client.tags()


SYSTEMONE = {
    "model": "nimble:9b-q4_K_M",
    "answers": {
        "sentiment": {
            "type": "choice",
            "choice": "bullish",
            "probabilities": {"neutral": 0.15, "bullish": 0.84, "bearish": 0.01},
            "confidence": 0.76,
        }
    },
    "usage": {"input_tokens": 143, "output_tokens": 0},
}


async def test_decide_sentiment(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    route = mock_ollama.post("/v1/systemone").respond(json=SYSTEMONE)
    criteria = {"neutral": "flat", "bullish": None, "bearish": "down"}
    result = await client.decide_sentiment("nimble:9b-q4_K_M", "<article>x</article>", "Impact?", criteria)  # type: ignore[arg-type]
    sent = _sent(route)
    question = sent["questions"]["sentiment"]
    assert question == {"type": "choice", "instructions": "Impact?", "criteria": criteria}
    assert list(question["criteria"]) == ["neutral", "bullish", "bearish"]
    assert sent["keep_alive"] == "5m"
    assert "options" not in sent
    assert "think" not in sent
    assert result.choice == "bullish"
    assert list(result.probabilities) == ["neutral", "bullish", "bearish"]
    assert result.probabilities["bullish"] == pytest.approx(0.84)
    assert result.confidence == pytest.approx(0.76)
    assert result.input_tokens == 143
    assert result.wall_ms > 0


@pytest.mark.parametrize(
    "answer",
    [
        {
            "choice": "positive",
            "probabilities": {"neutral": 0.5, "bullish": 0.5, "bearish": 0},
            "confidence": 0.1,
        },
        {"choice": "bullish", "probabilities": {"neutral": 0.5, "bullish": 0.5}, "confidence": 0.1},
        None,
    ],
)
async def test_decide_sentiment_rejects_unexpected_answers(
    client: OllamaClient, mock_ollama: respx.MockRouter, answer: dict[str, Any] | None
) -> None:
    body = {"model": "m", "answers": {"sentiment": answer} if answer else {}, "usage": {}}
    mock_ollama.post("/v1/systemone").respond(json=body)
    criteria = {"neutral": None, "bullish": None, "bearish": None}
    with pytest.raises(SchemaViolation):
        await client.decide_sentiment("m", "x", "q", criteria)  # type: ignore[arg-type]


async def test_decide_sentiment_needs_all_three_criteria(client: OllamaClient) -> None:
    with pytest.raises(ValueError, match="criteria"):
        await client.decide_sentiment("m", "x", "q", {"bullish": None, "bearish": None})


async def test_systemone_on_a_generative_model(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    error = {"error": 'model "qwen3.5:0.8b-q8_0" is not a llama.cpp decision model'}
    mock_ollama.post("/v1/systemone").respond(400, json=error)
    with pytest.raises(ModelUnavailable, match=r"not a llama\.cpp decision model"):
        await client.systemone("qwen3.5:0.8b-q8_0", "x", {})


async def test_utilities(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.post("/api/embed").respond(json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    mock_ollama.get("/api/tags").respond(json={"models": [{"name": "a"}]})
    mock_ollama.get("/api/ps").respond(json={"models": []})
    mock_ollama.post("/api/show").respond(json={"capabilities": ["completion"]})
    mock_ollama.get("/api/version").respond(json={"version": "0.40.6"})
    unload = mock_ollama.post("/api/generate").respond(json={"done": True, "done_reason": "unload"})
    assert await client.embed("e", ["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert await client.tags() == [{"name": "a"}]
    assert await client.ps() == []
    assert (await client.show("a"))["capabilities"] == ["completion"]
    assert await client.version() == "0.40.6"
    await client.unload("a")
    assert _sent(unload) == {"model": "a", "keep_alive": 0}


async def test_embed_count_mismatch(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.post("/api/embed").respond(json={"embeddings": [[0.1]]})
    with pytest.raises(SchemaViolation):
        await client.embed("e", ["a", "b"])


async def test_from_settings_and_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434/")
    monkeypatch.setenv("VT_OLLAMA_KEEP_ALIVE", "10m")
    async with OllamaClient.from_settings(Settings()) as ollama:
        assert ollama.base_url == "http://ollama.test:11434"
        assert ollama.keep_alive == "10m"
