from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    html: str
    content_type: str
    fetch_ms: float
    consent_ms: float | None = None
    redirects: int = 0


class Fetcher(Protocol):
    async def get(self, url: str) -> FetchResult: ...

    async def aclose(self) -> None: ...
