from __future__ import annotations

import pytest

from veritrade.services.cache import TtlCache


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_entries_expire_after_the_ttl() -> None:
    clock = Clock()
    cache: TtlCache[str, int] = TtlCache(max_entries=4, ttl_s=10, clock=clock)
    cache.put("a", 1)
    clock.now += 9.9
    assert cache.get("a") == 1
    clock.now += 0.1
    assert cache.get("a") is None
    assert len(cache) == 0


def test_least_recently_used_entry_is_evicted() -> None:
    cache: TtlCache[str, int] = TtlCache(max_entries=2, ttl_s=10, clock=Clock())
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


def test_put_refreshes_an_existing_key() -> None:
    clock = Clock()
    cache: TtlCache[str, int] = TtlCache(max_entries=2, ttl_s=10, clock=clock)
    cache.put("a", 1)
    clock.now += 8
    cache.put("a", 2)
    clock.now += 8
    assert cache.get("a") == 2


def test_zero_ttl_disables_the_cache() -> None:
    cache: TtlCache[str, int] = TtlCache(max_entries=2, ttl_s=0)
    cache.put("a", 1)
    assert cache.get("a") is None


def test_needs_room_for_one_entry() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        TtlCache(max_entries=0, ttl_s=1)
