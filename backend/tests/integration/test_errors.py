from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from starlette.types import Message

from veritrade.api.app import create_app
from veritrade.api.contracts import ErrorEvent, ErrorResponse, SummaryResponse
from veritrade.config import Settings
from veritrade.fetch import HttpFetcher
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import Registry

from .fakes import FakeOllama, Page, sse_events

BLOCKED_URL = "https://apnews.com/article/vt-blocked"
PAYWALLED_URL = "https://www.wsj.com/articles/vt-paywalled"
STUB_URL = "https://www.cnbc.com/vt-skeleton/stub"
# Synthetic pages; no publisher text.
CHALLENGE_HTML = "<!doctype html><html><head><title>Just a moment...</title></head><body></body></html>"
PAYWALL_HTML = (
    '<!doctype html><html><head><script type="application/ld+json">'
    '{"@type": "NewsArticle", "headline": "Teaser", "isAccessibleForFree": false, "articleBody": "Teaser."}'
    "</script></head><body><p>Subscribe to read the rest.</p></body></html>"
)
STUB_HTML = "<!doctype html><html><body><article><p>Only a caption.</p></article></body></html>"

Setup = Callable[[respx.MockRouter, FakeOllama], None]


def _publisher(url: str, answer: httpx.Response | Exception) -> Setup:
    def setup(router: respx.MockRouter, _: FakeOllama) -> None:
        route = router.get(url)
        if isinstance(answer, Exception):
            route.side_effect = answer
        else:
            route.return_value = answer

    return setup


def _chat_fails(failure: httpx.Response | Exception) -> Setup:
    def setup(_: respx.MockRouter, ollama: FakeOllama) -> None:
        ollama.chat_failure = failure

    return setup


def _nothing(router: respx.MockRouter, ollama: FakeOllama) -> None:
    return None


ERROR_TABLE: list[tuple[str, dict[str, Any], Setup, int, str]] = [
    ("/api/signal", {"url": "not a url"}, _nothing, 400, "invalid_url"),
    ("/api/signal", {"url": "ftp://files.example.com/a.html"}, _nothing, 400, "invalid_url"),
    ("/api/signal", {"url": "http://10.0.0.7/news/a.html"}, _nothing, 400, "blocked_destination"),
    ("/api/signal", {"url": "https://intranet.test/news/a.html"}, _nothing, 400, "blocked_destination"),
    ("/api/signal", {"url": "https://www.cnbc.com:8443/a.html"}, _nothing, 400, "blocked_destination"),
    ("/api/summary", {"url": "https://www.cnbc.com/a", "model": "gpt-9"}, _nothing, 400, "model_not_allowed"),
    (
        "/api/summary",
        {"url": "https://www.cnbc.com/a", "model": "fake-decision:1b"},
        _nothing,
        400,
        "model_not_allowed",
    ),
    (
        "/api/signal",
        {"url": PAYWALLED_URL},
        _publisher(PAYWALLED_URL, httpx.Response(200, html=PAYWALL_HTML)),
        422,
        "paywalled",
    ),
    (
        "/api/signal",
        {"url": STUB_URL},
        _publisher(STUB_URL, httpx.Response(200, html=STUB_HTML)),
        422,
        "extraction_empty",
    ),
    (
        "/api/signal",
        {"url": BLOCKED_URL},
        _publisher(
            BLOCKED_URL, httpx.Response(403, html=CHALLENGE_HTML, headers={"cf-mitigated": "challenge"})
        ),
        502,
        "source_blocked",
    ),
    (
        "/api/signal",
        {"url": BLOCKED_URL},
        _publisher(BLOCKED_URL, httpx.ConnectTimeout("slow")),
        504,
        "upstream_timeout",
    ),
]


@pytest.mark.parametrize(("path", "body", "setup", "status", "code"), ERROR_TABLE)
async def test_error_table(
    api: httpx.AsyncClient,
    router: respx.MockRouter,
    ollama: FakeOllama,
    path: str,
    body: dict[str, Any],
    setup: Setup,
    status: int,
    code: str,
) -> None:
    setup(router, ollama)
    response = await api.post(path, json=body)
    assert response.status_code == status, response.text
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code == code
    assert error.hint
    assert ollama.inference_calls == 0


async def test_blocked_source_names_a_syndicated_alternative(
    api: httpx.AsyncClient, router: respx.MockRouter
) -> None:
    router.get(BLOCKED_URL).respond(403, html=CHALLENGE_HTML, headers={"cf-mitigated": "challenge"})
    error = ErrorResponse.model_validate(
        (await api.post("/api/signal", json={"url": BLOCKED_URL})).json()
    ).error
    assert error.upstream_status == 403
    assert error.hint is not None
    assert "Yahoo Finance" in error.hint


async def test_unknown_model_is_refused_before_any_fetch(
    api: httpx.AsyncClient, router: respx.MockRouter, page: Page
) -> None:
    response = await api.post("/api/analyze", json={"url": page.url, "model": "gpt-9"})
    assert response.status_code == 400
    assert router.routes[page.source].call_count == 0


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (httpx.ConnectError("connection refused"), 503, "llm_unreachable"),
        (httpx.Response(404, json={"error": "model 'fake-label:1b' not found"}), 503, "model_unavailable"),
        (httpx.ReadTimeout("no answer"), 504, "inference_timeout"),
    ],
)
async def test_ollama_failures(
    api: httpx.AsyncClient,
    ollama: FakeOllama,
    page: Page,
    failure: httpx.Response | Exception,
    status: int,
    code: str,
) -> None:
    _chat_fails(failure)(ollama.router, ollama)
    for path in ("/api/signal", "/api/summary", "/api/analyze"):
        response = await api.post(path, json={"url": page.url})
        assert response.status_code == status, (path, response.text)
        assert ErrorResponse.model_validate(response.json()).error.code == code


@pytest.mark.parametrize("path", ["/api/summary/stream", "/api/analyze/stream"])
async def test_stream_failures_before_the_first_byte_are_json(
    api: httpx.AsyncClient, router: respx.MockRouter, path: str
) -> None:
    router.get(BLOCKED_URL).respond(403, html=CHALLENGE_HTML, headers={"cf-mitigated": "challenge"})
    response = await api.post(path, json={"url": BLOCKED_URL})
    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert ErrorResponse.model_validate(response.json()).error.code == "source_blocked"

    response = await api.post(path, json={"url": BLOCKED_URL, "model": "gpt-9"})
    assert (response.status_code, response.json()["error"]["code"]) == (400, "model_not_allowed")


async def test_ollama_failure_mid_stream_emits_one_error_event(
    api: httpx.AsyncClient, ollama: FakeOllama, page: Page
) -> None:
    ollama.stream_error_after = 8
    response = await api.post("/api/summary/stream", json={"url": page.url})
    assert response.status_code == 200
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "summary_delta" in names
    assert names[-1] == "error"
    assert "done" not in names
    error = ErrorEvent.model_validate(events[-1][1]).error
    assert error.code == "model_unavailable"

    # A failed generation is never cached.
    ollama.stream_error_after = None
    retry = SummaryResponse.model_validate((await api.post("/api/summary", json={"url": page.url})).json())
    assert retry.timings.cache.result == "miss"


async def test_client_disconnect_closes_the_ollama_stream(
    settings: Settings,
    fetcher: HttpFetcher,
    ollama_client: OllamaClient,
    registry: Registry,
    ollama: FakeOllama,
    page: Page,
) -> None:
    ollama.delay = 0.02
    payload = json.dumps({"url": page.url}).encode()
    first_delta = asyncio.Event()
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await first_delta.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and b"event: summary_delta" in message.get("body", b""):
            first_delta.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/summary/stream",
        "raw_path": b"/api/summary/stream",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    app = create_app(settings, fetcher=fetcher, ollama=ollama_client, registry=registry)
    async with LifespanManager(app) as manager:
        await asyncio.wait_for(manager.app(scope, receive, send), timeout=5)
        # Checked as soon as the request cycle ends: the close must not wait for garbage collection.
        [stream] = ollama.streams
        assert stream.closed
    assert first_delta.is_set()
    assert stream.sent < len(stream.lines)
