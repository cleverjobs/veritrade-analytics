from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from veritrade.api.app import create_app
from veritrade.config import Settings
from veritrade.fetch import HttpFetcher
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import Registry

from .fakes import OLLAMA_URL, FakeOllama, FakeResolver, Page, skeleton_pages

# Test roster, independent of the tuned models.toml; every name falls back to the default prompts.
MODELS_TOML = """
[defaults]
signal = "fake-label:1b"
summary = "fake-label:1b"
analyze = "fake-label:1b"

[models."fake-decision:1b"]
kind = "decision"
endpoints = ["signal"]
max_input_tokens = 2000
policy = "probability"
min_p = 0.5

[models."fake-label:1b"]
kind = "generative"
endpoints = ["signal", "summary", "analyze"]
num_ctx = 8192
max_input_tokens = 3000
num_predict = { signal = 16, summary = 160, analyze = 200 }
signal_schema = "label"
policy = "label-logprob"
min_p = 0.5
ordinal_t = 1
field_order = "sentiment-first"

[models."fake-ordinal:1b"]
kind = "generative"
endpoints = ["signal", "analyze"]
num_ctx = 8192
max_input_tokens = 3000
num_predict = { signal = 16, analyze = 200 }
signal_schema = "ordinal"
policy = "ordinal"
ordinal_t = 1
field_order = "summary-first"

[models."fake-plumbing:1b"]
kind = "decision"
endpoints = ["signal"]
max_input_tokens = 1500
policy = "probability"
min_p = 0.5
evaluated = false
"""

AppFactory = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        env="test",
        ollama_base_url=OLLAMA_URL,
        ollama_backend_label="test-backend",
        static_dir=None,
        cors_origins="",
        fetch_deadline_s=2.0,
        ollama_probe_timeout_s=0.5,
    )


@pytest.fixture
def registry(settings: Settings) -> Registry:
    return Registry.load(settings, toml_text=MODELS_TOML)


@pytest.fixture
def resolver() -> FakeResolver:
    return FakeResolver({"intranet.test": ["10.0.0.7"], "loopback.test": ["127.0.0.1"]})


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    # respx refuses unmocked requests by default, so nothing here reaches a real host.
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def ollama(router: respx.MockRouter) -> FakeOllama:
    return FakeOllama(router, installed=("fake-label:1b", "fake-decision:1b"))


@pytest.fixture
def pages(router: respx.MockRouter) -> dict[str, Page]:
    found = skeleton_pages()
    for page in found.values():
        router.get(page.url, name=page.source).respond(200, html=page.html)
    return found


@pytest.fixture
def page(pages: dict[str, Page]) -> Page:
    return pages["cnbc"]


@pytest.fixture
async def fetcher(settings: Settings, resolver: FakeResolver) -> AsyncIterator[HttpFetcher]:
    client = HttpFetcher(settings, resolver=resolver)
    yield client
    await client.aclose()


@pytest.fixture
async def ollama_client() -> AsyncIterator[OllamaClient]:
    client = OllamaClient(OLLAMA_URL, timeout=5.0, keep_alive="5m")
    yield client
    await client.aclose()


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
def app_factory(
    settings: Settings,
    fetcher: HttpFetcher,
    ollama_client: OllamaClient,
    registry: Registry,
    ollama: FakeOllama,
    pages: dict[str, Page],
) -> AppFactory:
    """Serve an app on the test fakes (always routed); keyword overrides replace settings or injections."""

    def build(**overrides: Any) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        injected = {"fetcher": fetcher, "ollama": ollama_client, "registry": registry}
        for name in list(overrides):
            if name in injected:
                injected[name] = overrides.pop(name)
        app = create_app(settings.model_copy(update=overrides), **injected)
        return serve(app)

    return build


@pytest.fixture
async def api(app_factory: AppFactory) -> AsyncIterator[httpx.AsyncClient]:
    async with app_factory() as client:
        yield client
