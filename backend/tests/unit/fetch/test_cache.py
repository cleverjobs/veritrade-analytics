from __future__ import annotations

import asyncio
import gc

import pytest

from veritrade.errors import SourceBlocked
from veritrade.fetch.cache import CacheResult, SingleFlightCache, normalize_url

from .conftest import FakeClock


class Factory:
    """Counts calls; optionally blocks on a gate so tests control when the in-flight work finishes."""

    def __init__(self, value: str = "v", *, error: Exception | None = None, gated: bool = False) -> None:
        self.value = value
        self.error = error
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        await self.gate.wait()
        if self.error is not None:
            raise self.error
        return f"{self.value}{self.calls}"


def make_cache(clock: FakeClock, *, max_entries: int = 8) -> SingleFlightCache[str, str]:
    return SingleFlightCache(max_entries=max_entries, ttl_s=60.0, error_ttl_s=5.0, clock=clock)


async def test_miss_then_hit(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory()
    assert await cache.get_or_create("k", factory) == CacheResult("v1", "miss")
    assert await cache.get_or_create("k", factory) == CacheResult("v1", "hit")
    assert factory.calls == 1
    assert len(cache) == 1


async def test_concurrent_callers_share_one_in_flight_call(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory(gated=True)
    first = asyncio.create_task(cache.get_or_create("k", factory))
    second = asyncio.create_task(cache.get_or_create("k", factory))
    await asyncio.sleep(0)
    factory.gate.set()
    results = await asyncio.gather(first, second)
    assert factory.calls == 1
    assert list(results) == [CacheResult("v1", "miss"), CacheResult("v1", "joined")]


async def test_different_keys_do_not_share(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory()
    await asyncio.gather(cache.get_or_create("a", factory), cache.get_or_create("b", factory))
    assert factory.calls == 2


async def test_cancelling_one_caller_does_not_cancel_the_others(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory(gated=True)
    first = asyncio.create_task(cache.get_or_create("k", factory))
    second = asyncio.create_task(cache.get_or_create("k", factory))
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    factory.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == CacheResult("v1", "joined")
    assert factory.calls == 1


async def test_work_finishes_and_is_cached_when_every_caller_gives_up(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory(gated=True)
    caller = asyncio.create_task(cache.get_or_create("k", factory))
    await asyncio.sleep(0)
    caller.cancel()
    factory.gate.set()
    await asyncio.sleep(0.01)
    assert await cache.get_or_create("k", factory) == CacheResult("v1", "hit")


async def test_failure_nobody_awaits_is_not_logged_as_unretrieved(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    cache = make_cache(clock)
    factory = Factory(gated=True, error=RuntimeError("boom"))
    caller = asyncio.create_task(cache.get_or_create("k", factory))
    await asyncio.sleep(0)
    caller.cancel()
    factory.gate.set()
    await asyncio.sleep(0.01)
    gc.collect()
    assert "never retrieved" not in caplog.text


async def test_ttl_expiry(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory()
    await cache.get_or_create("k", factory)
    clock.now += 59.9
    assert (await cache.get_or_create("k", factory)).status == "hit"
    clock.now += 0.2
    assert await cache.get_or_create("k", factory) == CacheResult("v2", "miss")
    assert factory.calls == 2


async def test_typed_errors_are_negatively_cached_for_the_error_ttl(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory(error=SourceBlocked("blocked", upstream_status=403))
    for _ in range(3):
        with pytest.raises(SourceBlocked) as caught:
            await cache.get_or_create("k", factory)
        assert caught.value.upstream_status == 403
    assert factory.calls == 1
    clock.now += 5.1
    with pytest.raises(SourceBlocked):
        await cache.get_or_create("k", factory)
    assert factory.calls == 2


async def test_joined_callers_all_see_the_typed_error(clock: FakeClock) -> None:
    cache = make_cache(clock)
    factory = Factory(gated=True, error=SourceBlocked("blocked"))
    callers = [asyncio.create_task(cache.get_or_create("k", factory)) for _ in range(3)]
    await asyncio.sleep(0)
    factory.gate.set()
    results = await asyncio.gather(*callers, return_exceptions=True)
    assert all(isinstance(result, SourceBlocked) for result in results)
    assert factory.calls == 1


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bad"), OSError("io")])
async def test_untyped_errors_are_not_cached(clock: FakeClock, error: Exception) -> None:
    cache = make_cache(clock)
    factory = Factory(error=error)
    for _ in range(2):
        with pytest.raises(type(error)):
            await cache.get_or_create("k", factory)
    assert factory.calls == 2
    assert len(cache) == 0


async def test_zero_error_ttl_disables_negative_caching(clock: FakeClock) -> None:
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        max_entries=4, ttl_s=60, error_ttl_s=0, clock=clock
    )
    factory = Factory(error=SourceBlocked("x"))
    for _ in range(2):
        with pytest.raises(SourceBlocked):
            await cache.get_or_create("k", factory)
    assert factory.calls == 2


async def test_lru_evicts_the_least_recently_used(clock: FakeClock) -> None:
    cache = make_cache(clock, max_entries=2)
    factories = {key: Factory(key) for key in "abc"}
    await cache.get_or_create("a", factories["a"])
    await cache.get_or_create("b", factories["b"])
    await cache.get_or_create("a", factories["a"])
    await cache.get_or_create("c", factories["c"])
    assert len(cache) == 2
    assert (await cache.get_or_create("a", factories["a"])).status == "hit"
    assert (await cache.get_or_create("b", factories["b"])).status == "miss"
    assert (await cache.get_or_create("c", factories["c"])).status == "miss"


async def test_weight_bound_evicts_and_skips_oversized_values(clock: FakeClock) -> None:
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        max_entries=10, ttl_s=60, error_ttl_s=5, weigh=len, max_weight=10, clock=clock
    )
    await cache.get_or_create("a", Factory("aaaa"))
    await cache.get_or_create("b", Factory("bbbb"))
    assert cache.weight == 10
    await cache.get_or_create("c", Factory("cccc"))
    assert len(cache) == 2
    assert cache.weight == 10
    await cache.get_or_create("huge", Factory("x" * 20))
    assert len(cache) == 2
    assert (await cache.get_or_create("a", Factory("aaaa"))).status == "miss"


def test_clear_and_invalid_size(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.clear()
    assert len(cache) == 0
    assert cache.weight == 0
    with pytest.raises(ValueError, match="max_entries"):
        make_cache(clock, max_entries=0)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://Finance.Yahoo.COM/news/X.html", "https://finance.yahoo.com/news/X.html"),
        ("https://news.test/a#section-2", "https://news.test/a"),
        ("https://news.test/a?utm_source=x&id=7&UTM_Medium=y&fbclid=abc", "https://news.test/a?id=7"),
        ("https://news.test/a?gclid=1&b=2&a=1", "https://news.test/a?b=2&a=1"),
        ("https://news.test/a?q=a%20b&utm_campaign=z", "https://news.test/a?q=a%20b"),
        ("https://news.test/a?utm_source=x", "https://news.test/a"),
        ("https://news.test:443/a", "https://news.test/a"),
        ("http://news.test:80/a", "http://news.test/a"),
        ("https://news.test:8443/a", "https://news.test:8443/a"),
        ("https://news.test", "https://news.test/"),
        ("  https://news.test/a  ", "https://news.test/a"),
        ("https://news.test/a?flag&&utm_id=1", "https://news.test/a?flag"),
        ("https://news.test/a?utm=keep", "https://news.test/a?utm=keep"),
    ],
)
def test_normalize_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


def test_normalize_url_is_idempotent() -> None:
    url = "HTTPS://News.Test:443/Path?utm_x=1&a=b#frag"
    assert normalize_url(normalize_url(url)) == normalize_url(url) == "https://news.test/Path?a=b"
