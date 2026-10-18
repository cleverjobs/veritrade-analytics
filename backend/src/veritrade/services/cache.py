"""Result cache: finished responses keyed by everything that can change them."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from veritrade.api.contracts import AnalyzeResponse, SignalResponse, SummaryResponse
from veritrade.extract.types import ExtractorName, ParserName, Scope
from veritrade.inference.types import Endpoint

CachedResponse = SignalResponse | SummaryResponse | AnalyzeResponse


@dataclass(frozen=True, slots=True)
class ResultKey:
    endpoint: Endpoint
    url: str
    model: str
    prompt_version: str
    parser: ParserName
    extractor: ExtractorName
    scope: Scope


class TtlCache[K: Hashable, V]:
    """LRU with a fixed time to live; no single-flight, since a stream cannot be shared mid-generation."""

    def __init__(
        self, *, max_entries: int, ttl_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._clock = clock
        self._entries: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: K) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        if self._ttl_s <= 0:
            return
        self._entries[key] = (self._clock() + self._ttl_s, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
