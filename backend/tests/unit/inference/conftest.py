from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import httpx
import pytest
import respx

from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import ModelSpec

BASE_URL = "http://ollama.test"


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client() -> AsyncIterator[OllamaClient]:
    ollama = OllamaClient(BASE_URL, timeout=2.0, keep_alive="5m")
    yield ollama
    await ollama.aclose()


@pytest.fixture
def decision_spec() -> ModelSpec:
    return ModelSpec(
        name="nimble:9b-q4_K_M",
        kind="decision",
        endpoints=("signal",),
        max_input_tokens=100,
        policy="probability",
        min_p=0.5,
    )


@pytest.fixture
def label_spec() -> ModelSpec:
    return ModelSpec(
        name="qwen3.5:0.8b-q8_0",
        kind="generative",
        endpoints=("signal", "summary", "analyze"),
        num_ctx=8192,
        max_input_tokens=100,
        num_predict={"signal": 16, "summary": 160, "analyze": 200},
        signal_schema="label",
        policy="label-logprob",
        min_p=0.5,
        ordinal_t=1,
    )


@pytest.fixture
def ordinal_spec() -> ModelSpec:
    return ModelSpec(
        name="qwen3.5:4b-q4_K_M",
        kind="generative",
        endpoints=("summary", "analyze"),
        num_ctx=8192,
        max_input_tokens=3000,
        num_predict={"summary": 160, "analyze": 200},
        policy="ordinal",
        ordinal_t=2,
        field_order="sentiment-first",
    )


def lp(token: str, p: float, alternatives: Sequence[tuple[str, float]] = ()) -> dict[str, Any]:
    """One Ollama logprob entry; alternatives are (token, probability) and include the token itself."""
    top = [{"token": t, "logprob": math.log(q), "bytes": list(t.encode())} for t, q in alternatives]
    return {"token": token, "logprob": math.log(p), "bytes": list(token.encode()), "top_logprobs": top}


def plain(tokens: Sequence[str]) -> list[dict[str, Any]]:
    return [lp(t, 0.99, [(t, 0.99)]) for t in tokens]


def chat_body(content: str, *, logprobs: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "m",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 250_000_000,
        "load_duration": 10_000_000,
        "prompt_eval_count": 120,
        "prompt_eval_duration": 80_000_000,
        "eval_count": 12,
        "eval_duration": 150_000_000,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    return body | extra


def stream_response(chunks: Sequence[dict[str, Any]], delay: float = 0.0) -> httpx.Response:
    async def lines() -> AsyncIterator[bytes]:
        for chunk in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield (json.dumps(chunk) + "\n").encode()

    return httpx.Response(200, content=lines(), headers={"content-type": "application/x-ndjson"})


def content_chunks(pieces: Sequence[str], **final: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = [
        {"model": "m", "message": {"role": "assistant", "content": piece}, "done": False} for piece in pieces
    ]
    done = chat_body("", eval_count=len(pieces)) | final
    return [*chunks, done]
