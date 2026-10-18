from __future__ import annotations

import socket
from collections.abc import AsyncIterator, Iterator

import pytest
import respx

from veritrade.config import Settings
from veritrade.fetch.client import HostRateLimiter, HttpFetcher

PUBLIC_IP = "93.184.215.14"
# Synthetic article markup; no publisher text.
ARTICLE_HTML = (
    "<!doctype html><html><head><title>Acme beats estimates</title></head>"
    "<body><article><h1>Acme beats estimates</h1>"
    + "<p>Synthetic paragraph text for tests.</p>" * 40
    + "</article></body></html>"
)


class FakeResolver:
    """Maps host names to canned answers and records every lookup; unknown hosts get a public address."""

    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self.table = table or {}
        self.calls: list[str] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append(host)
        answer = self.table.get(host, [PUBLIC_IP])
        if answer == ["NXDOMAIN"]:
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")
        if answer == ["SERVFAIL"]:
            raise socket.gaierror(socket.EAI_AGAIN, "temporary failure in name resolution")
        return answer


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


@pytest.fixture
def resolver() -> FakeResolver:
    return FakeResolver(
        {
            "internal.test": ["10.0.0.5"],
            "metadata.test": ["169.254.169.254"],
            "mixed.test": [PUBLIC_IP, "127.0.0.1"],
        }
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(env="test", fetch_max_bytes=100_000, fetch_deadline_s=2.0, fetch_max_redirects=3)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def fetcher(settings: Settings, resolver: FakeResolver, clock: FakeClock) -> AsyncIterator[HttpFetcher]:
    client = HttpFetcher(settings, resolver=resolver, limiter=HostRateLimiter(clock=clock, sleep=clock.sleep))
    yield client
    await client.aclose()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    # assert_all_mocked (respx default) fails any request without a route, so nothing reaches the network.
    with respx.mock(assert_all_called=False) as mock:
        yield mock
