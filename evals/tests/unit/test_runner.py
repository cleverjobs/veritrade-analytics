from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from veritrade.errors import ModelUnavailable
from veritrade.inference.ollama import FIXED_OPTIONS, OllamaClient
from veritrade.inference.types import Endpoint, Sentiment
from veritrade.models_registry import allowlist
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals.datasets import Example
from veritrade_evals.runner import (
    RunAborted,
    RunSpec,
    Variant,
    call_once,
    make_resident,
    model_digest,
    run_cell,
    serving_spec,
)

BASE_URL = "http://ollama.test"
DIGEST = "e" * 64
DECISION = "nimble:9b-q4_K_M"
SMALL = "qwen3.5:0.8b-q8_0"
COMBINED = "ministral-3:3b"
EXAMPLES = [
    Example("x:0", "Zentor Mills beats estimates", "bullish", "tfns", "train"),
    Example("x:1", "Orvane Bank posts a loss", "bearish", "tfns", "train"),
    Example("x:2", "Pellow Foods sets a meeting date", "neutral", "tfns", "train"),
]


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client() -> AsyncIterator[OllamaClient]:
    ollama = OllamaClient(BASE_URL, timeout=2.0, keep_alive="5m")
    yield ollama
    await ollama.aclose()


def cell(model: str, variant: Variant, version: str = "v1") -> RunSpec:
    base = allowlist()[model]
    endpoint: Endpoint = "analyze" if variant == "sentiment-first" else "signal"
    prompt = PromptRegistry().get(endpoint, model, kind=base.kind, version=version)
    return RunSpec(f"{version}-{variant}", serving_spec(base, variant), prompt, variant)


def lp(token: str, p: float, alternatives: Sequence[tuple[str, float]] = ()) -> dict[str, Any]:
    top = [{"token": t, "logprob": math.log(q)} for t, q in alternatives]
    return {"token": token, "logprob": math.log(p), "top_logprobs": top}


def plain(tokens: Sequence[str]) -> list[dict[str, Any]]:
    return [lp(t, 0.99, [(t, 0.99)]) for t in tokens]


def chat_body(content: str, logprobs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "m",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 120_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 300,
        "prompt_eval_duration": 80_000_000,
        "eval_count": 7,
        "eval_duration": 40_000_000,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    return body


def label_logprobs(first: str, alternatives: Sequence[tuple[str, float]], rest: Sequence[str]) -> list[Any]:
    return [*plain(['{"', "label", '":', ' "']), lp(first, alternatives[0][1], alternatives), *plain(rest)]


def decision_body(probs: dict[Sentiment, float], choice: Sentiment = "bullish") -> dict[str, Any]:
    answer = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.5}
    return {"model": DECISION, "answers": {"sentiment": answer}, "usage": {"input_tokens": 210}}


def ndjson(pieces: Sequence[str], *, hang_after: int | None = None) -> httpx.Response:
    chunks: list[dict[str, Any]] = [
        {"model": "m", "message": {"role": "assistant", "content": p}, "done": False} for p in pieces
    ]
    chunks.append(chat_body("") | {"message": {"role": "assistant", "content": ""}})

    async def lines() -> AsyncIterator[bytes]:
        for i, chunk in enumerate(chunks):
            if i == hang_after:
                await asyncio.sleep(5)
            yield (json.dumps(chunk) + "\n").encode()

    return httpx.Response(200, content=lines(), headers={"content-type": "application/x-ndjson"})


# --- raw outputs per model kind --------------------------------------------------------------


async def test_decision_keeps_normalised_probabilities(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    mock_ollama.post("/v1/systemone").respond(
        json=decision_body({"neutral": 0.2, "bullish": 0.6, "bearish": 0.2}, choice="bullish")
    )
    pred, t = await call_once(client, cell(DECISION, "decision"), "Zentor Mills beats estimates")
    assert pred["kind"] == "decision"
    assert pred["choice"] == "bullish"
    assert pred["probs"] == pytest.approx({"neutral": 0.2, "bullish": 0.6, "bearish": 0.2})
    assert pred["model_confidence"] == 0.5
    assert t["prompt_tokens"] == 210
    assert t["wall_ms"] > 0


async def test_label_keeps_first_token_probabilities_and_ollama_timings(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    logprobs = label_logprobs("bull", [("bull", 0.6), ("neutral", 0.3), ("bear", 0.1)], ["ish", '"}'])
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"label": "bullish"}', logprobs))
    pred, t = await call_once(client, cell(SMALL, "label"), "Zentor Mills beats estimates")
    assert route.call_count == 1
    assert pred["kind"] == "label"
    assert (pred["label"], pred["fallback"], pred["score"]) == ("bullish", "none", None)
    assert pred["probs"] == pytest.approx({"neutral": 0.3, "bullish": 0.6, "bearish": 0.1})
    assert t["total_ms"] == pytest.approx(120.0)
    assert (t["prompt_tokens"], t["eval_tokens"]) == (300, 7)


async def test_label_collision_records_the_score_sign_fallback(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    colliding = label_logprobs("b", [("b", 0.9), ("neutral", 0.1)], ["ullish", '"}'])
    sign = [
        *plain(['{"', "score", '":', " "]),
        lp("1", 0.7, [("1", 0.7), ("0", 0.2), ("-", 0.1)]),
        *plain(["}"]),
    ]
    mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"label": "bullish"}', colliding)),
            httpx.Response(200, json=chat_body('{"score": 1}', sign)),
        ]
    )
    pred, t = await call_once(client, cell(SMALL, "label"), "Zentor Mills beats estimates")
    assert (pred["fallback"], pred["label"], pred["score"]) == ("score-sign", None, 1)
    assert pred["probs"] == pytest.approx({"neutral": 0.2, "bullish": 0.7, "bearish": 0.1})
    assert t["eval_tokens"] == 14


async def test_label_without_logprobs_ends_at_the_ordinal_fallback(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    mock_ollama.post("/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=chat_body('{"label": "bearish"}')),
            httpx.Response(200, json=chat_body('{"score": -2}')),
        ]
    )
    pred, _ = await call_once(client, cell(SMALL, "label"), "Orvane Bank posts a loss")
    assert (pred["fallback"], pred["probs"], pred["score"]) == ("ordinal", None, -2)


async def test_ordinal_keeps_score_and_sign_probabilities(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    sign = [*plain(['{"', "score", '":', " "]), lp("-", 0.8, [("-", 0.8), ("0", 0.2)]), *plain(["1", "}"])]
    route = mock_ollama.post("/api/chat").respond(json=chat_body('{"score": -1}', sign))
    pred, _ = await call_once(client, cell(SMALL, "ordinal"), "Orvane Bank posts a loss")
    assert json.loads(route.calls[0].request.content)["format"]["properties"] == {
        "score": {"enum": [-2, -1, 0, 1, 2], "type": "integer"}
    }
    assert (pred["kind"], pred["score"]) == ("ordinal", -1)
    assert pred["sign_probs"] == pytest.approx({"neutral": 0.2, "bullish": 0.0, "bearish": 0.8})


async def test_sentiment_first_stops_reading_once_the_score_closes(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    pieces = ['{"score": ', "-1", ', "entity": "Orvane Bank", ', '"summary": "Never read."}']
    route = mock_ollama.post("/api/chat").mock(side_effect=lambda request: ndjson(pieces, hang_after=2))
    t0 = time.perf_counter()
    pred, t = await call_once(client, cell(COMBINED, "sentiment-first"), "Orvane Bank posts a loss")
    # The upstream stalls for 5 s after the score; only an early stop returns well before that.
    assert time.perf_counter() - t0 < 2.0
    assert pred == {"kind": "analyze", "score": -1}
    assert set(t) == {"wall_ms"}
    sent = json.loads(route.calls[0].request.content)
    assert sent["stream"] is True
    assert list(sent["format"]["properties"]) == ["score", "entity", "summary"]


# --- cell runs -------------------------------------------------------------------------------


async def test_resume_makes_no_repeat_calls_and_stores_no_text(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    route = mock_ollama.post("/v1/systemone").respond(
        json=decision_body({"neutral": 0.1, "bullish": 0.8, "bearish": 0.1})
    )
    run = cell(DECISION, "decision")
    first = await run_cell(client, run, EXAMPLES[:2], split="dev300", model_digest=DIGEST, root=tmp_path)
    assert (first.called, first.cached, route.call_count) == (2, 0, 2)

    again = await run_cell(client, run, EXAMPLES[:2], split="dev300", model_digest=DIGEST, root=tmp_path)
    assert (again.called, again.cached, route.call_count) == (0, 2, 2)
    assert again.key == first.key

    grown = await run_cell(client, run, EXAMPLES, split="dev300", model_digest=DIGEST, root=tmp_path)
    assert (grown.called, grown.cached, route.call_count) == (1, 2, 3)

    stored = first.path.read_text(encoding="utf-8")
    assert all(e.text not in stored for e in EXAMPLES)
    assert [json.loads(line)["id"] for line in stored.splitlines()] == ["x:0", "x:1", "x:2"]


async def test_cache_key_follows_digest_and_prompt(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    mock_ollama.post("/v1/systemone").respond(
        json=decision_body({"neutral": 0.5, "bullish": 0.3, "bearish": 0.2})
    )
    kwargs: dict[str, Any] = {"split": "dev300", "root": tmp_path}
    a = await run_cell(client, cell(DECISION, "decision"), EXAMPLES[:1], model_digest=DIGEST, **kwargs)
    b = await run_cell(client, cell(DECISION, "decision"), EXAMPLES[:1], model_digest="f" * 64, **kwargs)
    c = await run_cell(client, cell(DECISION, "decision", "v2"), EXAMPLES[:1], model_digest=DIGEST, **kwargs)
    assert len({a.key, b.key, c.key}) == 3
    assert b.called == c.called == 1


async def test_failures_are_recorded_and_only_transient_ones_rerun(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    ok = decision_body({"neutral": 0.1, "bullish": 0.8, "bearish": 0.1})
    route = mock_ollama.post("/v1/systemone").mock(
        side_effect=[
            httpx.Response(200, json=ok),
            httpx.Response(500, json={"error": "runner crashed"}),
            httpx.Response(200, json={"model": DECISION, "answers": {}}),
        ]
    )
    run = cell(DECISION, "decision")
    stats = await run_cell(
        client, run, EXAMPLES, split="dev300", model_digest=DIGEST, root=tmp_path, concurrency=1
    )
    assert (stats.called, stats.errors) == (3, 2)
    records = run.cache("dev300", DIGEST, tmp_path).records()
    assert records["x:1"]["error"]["code"] == "model_unavailable"
    assert records["x:1"]["error"]["retryable"] is True
    assert records["x:2"]["error"]["code"] == "schema_violation"
    assert records["x:2"]["error"]["retryable"] is False

    route.side_effect = None
    route.return_value = httpx.Response(200, json=ok)
    resumed = await run_cell(client, run, EXAMPLES, split="dev300", model_digest=DIGEST, root=tmp_path)
    assert (resumed.cached, resumed.called, route.call_count) == (2, 1, 4)


async def test_unreachable_ollama_aborts_the_run(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    mock_ollama.post("/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
    run = cell(DECISION, "decision")
    with pytest.raises(RunAborted):
        await run_cell(client, run, EXAMPLES, split="dev300", model_digest=DIGEST, root=tmp_path)
    assert run.cache("dev300", DIGEST, tmp_path).records() == {}


async def test_concurrency_is_bounded(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    in_flight = peak = 0

    async def slow(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=decision_body({"neutral": 0.4, "bullish": 0.3, "bearish": 0.3}))

    mock_ollama.post("/v1/systemone").mock(side_effect=slow)
    many = [Example(f"x:{i}", f"headline {i}", "neutral", "tfns", "train") for i in range(9)]
    stats = await run_cell(
        client,
        cell(DECISION, "decision"),
        many,
        split="dev300",
        model_digest=DIGEST,
        root=tmp_path,
        concurrency=3,
    )
    assert stats.called == 9
    assert peak == 3


# --- residency and identity ------------------------------------------------------------------


async def test_make_resident_unloads_other_models_then_warms_up(
    client: OllamaClient, mock_ollama: respx.MockRouter
) -> None:
    mock_ollama.get("/api/ps").respond(json={"models": [{"name": "qwen3.5:2b-q4_K_M"}, {"name": DECISION}]})
    unload = mock_ollama.post("/api/generate").respond(json={"done": True})
    decide = mock_ollama.post("/v1/systemone").respond(
        json=decision_body({"neutral": 0.4, "bullish": 0.3, "bearish": 0.3})
    )
    await make_resident(client, cell(DECISION, "decision"))
    assert [json.loads(c.request.content) for c in unload.calls] == [
        {"model": "qwen3.5:2b-q4_K_M", "keep_alive": 0}
    ]
    assert decide.call_count == 1


async def test_model_digest_comes_from_tags(client: OllamaClient, mock_ollama: respx.MockRouter) -> None:
    mock_ollama.get("/api/tags").respond(
        json={"models": [{"name": DECISION, "model": DECISION, "digest": DIGEST}]}
    )
    assert await model_digest(client, DECISION) == DIGEST
    with pytest.raises(ModelUnavailable):
        await model_digest(client, SMALL)


def test_prompt_sha_and_run_config_separate_every_cell() -> None:
    cells = [cell(SMALL, v, version) for v in ("label", "ordinal") for version in ("v1", "v2")]
    assert len({c.prompt_sha() for c in cells}) == 4
    assert cells[0].prompt_sha() == cell(SMALL, "label").prompt_sha()
    assert cells[0].run_config("dev300") == {
        "runner": "1",
        "split": "dev300",
        "endpoint": "signal",
        "variant": "label",
        "prompt_version": cells[0].prompt.version,
        "max_input_tokens": 3000,
        "options": {"num_ctx": 8192, "num_predict": 16, **FIXED_OPTIONS},
        "top_logprobs": 20,
    }
    assert cell(COMBINED, "sentiment-first").run_config("dev300")["stop_after_score"] is True
    assert "options" not in cell(DECISION, "decision").run_config("dev300")


def test_serving_spec_rejects_a_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="cannot run"):
        serving_spec(allowlist()[DECISION], "label")
    with pytest.raises(ValueError, match="cannot run"):
        serving_spec(allowlist()[SMALL], "decision")
    assert serving_spec(allowlist()[SMALL], "ordinal").policy == "ordinal"
