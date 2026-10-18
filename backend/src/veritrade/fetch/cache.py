"""In-process LRU + TTL cache whose concurrent misses for one key share a single in-flight task."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from veritrade.errors import VtError

CacheStatus = Literal["hit", "miss", "joined"]
TRACKING_PARAMS = frozenset({"fbclid", "gclid"})
DEFAULT_PORTS = {"http": ":80", "https": ":443"}


@dataclass(frozen=True, slots=True)
class CacheResult[V]:
    value: V
    status: CacheStatus


@dataclass(frozen=True, slots=True)
class _Value[V]:
    value: V
    expires_at: float
    weight: int


@dataclass(frozen=True, slots=True)
class _Failure:
    error: VtError
    expires_at: float
    weight: int = 0


class SingleFlightCache[K: Hashable, V]:
    """Typed VtErrors are cached for error_ttl_s so a refusing publisher is not hammered; others are not.

    weigh/max_weight bound memory as well as entry count, since one cached page can be several MB.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_s: float,
        error_ttl_s: float,
        weigh: Callable[[V], int] | None = None,
        max_weight: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._error_ttl_s = error_ttl_s
        self._weigh = weigh
        self._max_weight = max_weight
        self._clock = clock
        self._entries: OrderedDict[K, _Value[V] | _Failure] = OrderedDict()
        self._inflight: dict[K, asyncio.Task[V]] = {}
        self._weight = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def weight(self) -> int:
        return self._weight

    def clear(self) -> None:
        self._entries.clear()
        self._weight = 0

    async def get_or_create(self, key: K, factory: Callable[[], Awaitable[V]]) -> CacheResult[V]:
        entry = self._entries.get(key)
        if entry is not None:
            if entry.expires_at > self._clock():
                self._entries.move_to_end(key)
                if isinstance(entry, _Failure):
                    raise entry.error.with_traceback(None)
                return CacheResult(entry.value, "hit")
            self._drop(key)
        status: CacheStatus = "joined"
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._run(key, factory))
            self._inflight[key] = task
            status = "miss"
        # Shielded so a caller that gives up does not cancel work others wait on; shield also marks the
        # error of a task every caller abandoned as retrieved, so asyncio does not log it.
        return CacheResult(await asyncio.shield(task), status)

    async def _run(self, key: K, factory: Callable[[], Awaitable[V]]) -> V:
        try:
            value = await factory()
        except VtError as exc:
            if self._error_ttl_s > 0:
                self._store(key, _Failure(exc, self._clock() + self._error_ttl_s))
            raise
        finally:
            self._inflight.pop(key, None)
        weight = self._weigh(value) if self._weigh else 0
        if self._ttl_s > 0 and (self._max_weight is None or weight <= self._max_weight):
            self._store(key, _Value(value, self._clock() + self._ttl_s, weight))
        return value

    def _store(self, key: K, entry: _Value[V] | _Failure) -> None:
        self._drop(key)
        self._entries[key] = entry
        self._weight += entry.weight
        while len(self._entries) > self._max_entries or (
            self._max_weight is not None and self._weight > self._max_weight
        ):
            _, evicted = self._entries.popitem(last=False)
            self._weight -= evicted.weight

    def _drop(self, key: K) -> None:
        if (entry := self._entries.pop(key, None)) is not None:
            self._weight -= entry.weight


def normalize_url(url: str) -> str:
    """Cache key: lowercase scheme and host, no default port, fragment or click-tracking parameters."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    userinfo, at, hostport = parts.netloc.rpartition("@")
    hostport = hostport.lower()
    if (default := DEFAULT_PORTS.get(scheme)) and hostport.endswith(default):
        hostport = hostport.removesuffix(default)
    query = "&".join(part for part in parts.query.split("&") if part and not _is_tracking(part))
    return urlunsplit((scheme, f"{userinfo}{at}{hostport}", parts.path or "/", query, ""))


def _is_tracking(param: str) -> bool:
    name = unquote_plus(param.partition("=")[0]).lower()
    return name.startswith("utm_") or name in TRACKING_PARAMS
