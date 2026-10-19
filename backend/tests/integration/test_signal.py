from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from veritrade.api.contracts import CacheInfo, SignalResponse

from .conftest import AppFactory
from .fakes import FakeOllama, Page, skeleton_pages


def _sent(route: Any) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls.last.request.content)
    return body


@pytest.mark.parametrize("source", sorted(skeleton_pages()))
async def test_decision_model_reads_the_lede_of_every_source(
    api: httpx.AsyncClient, ollama: FakeOllama, pages: dict[str, Page], source: str
) -> None:
    page = pages[source]
    response = await api.post("/api/signal", json={"url": page.url, "model": "fake-decision:1b"})
    assert response.status_code == 200, response.text
    body = SignalResponse.model_validate(response.json())
    assert (body.signal.sentiment, body.signal.abstained, body.signal.policy) == (
        "bullish",
        False,
        "probability",
    )
    assert body.signal.probabilities == pytest.approx({"neutral": 0.06, "bullish": 0.9, "bearish": 0.04})
    assert (body.signal.threshold, body.signal.directional_tau) == (0.5, None)
    meta = body.meta
    assert (meta.source, meta.endpoint, meta.model_kind, meta.scope) == (source, "signal", "decision", "lede")
    assert (meta.url, meta.final_url, meta.cached, meta.backend) == (
        page.url,
        page.url,
        False,
        "test-backend",
    )
    assert meta.prompt_version.startswith("signal/fake-decision:1b/")
    assert meta.word_count > 0
    assert body.timings.cache == CacheInfo(fetch="miss", result="miss")
    assert body.timings.inference is not None
    assert body.timings.inference.prompt_tokens == 264
    assert body.timings.inference.cold_start is False
    assert body.timings.total_ms >= body.timings.fetch_ms + body.timings.inference.wall_ms
    state = _sent(ollama.systemone)["state"]
    # The lede is the headline plus the opening paragraphs, never the end of the article.
    assert page.body_markers[0] in state
    assert page.body_markers[-1] not in state


async def test_generative_model_thresholds_label_logprobs(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    response = await api.post("/api/signal", json={"url": page.url, "model": "fake-label:1b"})
    assert response.status_code == 200, response.text
    body = SignalResponse.model_validate(response.json())
    signal = body.signal
    assert (signal.sentiment, signal.abstained, signal.policy) == ("bullish", False, "label-logprob")
    assert signal.confidence == pytest.approx(0.9)
    assert (signal.threshold, signal.directional_tau) == (0.5, None)
    assert body.timings.inference is not None
    assert body.timings.inference.cold_start is False
    sent = _sent(ollama.chat)
    assert list(sent["format"]["properties"]) == ["label"]
    assert sent["logprobs"] is True
    assert sent["options"]["num_predict"] == 16
    assert sent["stream"] is False


async def test_generative_model_abstains_below_the_threshold(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.label_alternatives = {"bull": 0.45, "neutral": 0.4, "bear": 0.15}
    response = await api.post("/api/signal", json={"url": page.url})
    signal = SignalResponse.model_validate(response.json()).signal
    assert (signal.sentiment, signal.abstained) == ("neutral", True)
    assert signal.confidence == pytest.approx(0.45)


async def test_ordinal_model_on_the_full_article(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.score = -2
    response = await api.post(
        "/api/signal", json={"url": page.url, "model": "fake-ordinal:1b", "scope": "full", "parser": "lxml"}
    )
    assert response.status_code == 200, response.text
    body = SignalResponse.model_validate(response.json())
    assert (body.signal.sentiment, body.signal.score, body.signal.policy) == ("bearish", -2, "ordinal")
    assert (body.signal.threshold, body.signal.directional_tau) == (1.0, None)
    assert (body.meta.scope, body.meta.parser) == ("full", "lxml")
    sent = _sent(ollama.chat)
    assert list(sent["format"]["properties"]) == ["score"]
    assert page.body_markers[-1] in sent["messages"][-1]["content"]


async def test_a_model_that_is_not_loaded_is_a_cold_start(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    del ollama.resident["fake-label:1b"]
    response = await api.post("/api/signal", json={"url": page.url})
    inference = SignalResponse.model_validate(response.json()).timings.inference
    assert inference is not None
    assert (inference.cold_start, inference.load_ms) == (True, pytest.approx(5.0))
    assert ollama.ps.call_count == 1


@pytest.mark.parametrize(
    "failure", [httpx.Response(500, json={"error": "scheduler busy"}), httpx.ConnectError("refused")]
)
async def test_a_failed_residency_probe_leaves_cold_start_unknown(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page, failure: httpx.Response | Exception
) -> None:
    ollama.ps_failure = failure
    response = await api.post("/api/signal", json={"url": page.url})
    assert response.status_code == 200, response.text
    inference = SignalResponse.model_validate(response.json()).timings.inference
    assert inference is not None
    assert inference.cold_start is None


async def test_a_hanging_residency_probe_gives_up_at_the_probe_timeout(
    app_factory: AppFactory, ollama: FakeOllama, page: Page
) -> None:
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"models": []})

    ollama.ps.side_effect = hang
    async with app_factory(ollama_probe_timeout_s=0.05) as client:
        response = await client.post("/api/signal", json={"url": page.url})
    assert response.status_code == 200, response.text
    body = SignalResponse.model_validate(response.json())
    assert body.timings.inference is not None
    assert body.timings.inference.cold_start is None
    assert body.timings.total_ms < 5000
