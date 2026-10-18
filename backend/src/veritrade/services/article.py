"""URL to Article: single-flight fetch, then extraction (off the event loop when it is slow)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal

from veritrade.extract.registry import extract
from veritrade.extract.types import PARSER_AWARE_EXTRACTORS, Article, ExtractorName, ParserName
from veritrade.fetch import FetchResult, normalize_url
from veritrade.services.runtime import Runtime

FetchStatus = Literal["hit", "miss", "joined", "bypass"]
# Own strategies on these parsers take 0.4-4 ms on real pages, less than handing off is worth; bs4 and the
# third-party extractors take 5-60 ms of Python, which a thread lets concurrent streams interleave with.
INLINE_PARSERS: frozenset[ParserName] = frozenset({"selectolax", "lxml"})


@dataclass(frozen=True, slots=True)
class LoadedArticle:
    page: FetchResult
    article: Article
    fetch_status: FetchStatus
    fetch_ms: float
    consent_ms: float | None


async def load_article(
    runtime: Runtime, url: str, *, parser: ParserName, extractor: ExtractorName, no_cache: bool
) -> LoadedArticle:
    started = time.perf_counter()
    status: FetchStatus
    if no_cache:
        page, status = await runtime.fetcher.get(url), "bypass"
    else:
        fetch = partial(runtime.fetcher.get, url)
        shared = await runtime.fetch_cache.get_or_create(normalize_url(url), fetch)
        page, status = shared.value, shared.status
    if status in ("miss", "bypass"):
        fetch_ms, consent_ms = page.fetch_ms, page.consent_ms
    else:
        # A hit or join reports this request's own wait, not the download it reused.
        fetch_ms, consent_ms = (time.perf_counter() - started) * 1000, None
    article = await extract_article(page, parser=parser, extractor=extractor)
    return LoadedArticle(page, article, status, fetch_ms, consent_ms)


async def extract_article(page: FetchResult, *, parser: ParserName, extractor: ExtractorName) -> Article:
    run = partial(extract, page.html, page.final_url, parser=parser, extractor=extractor)
    if extractor in PARSER_AWARE_EXTRACTORS and parser in INLINE_PARSERS:
        return run()
    return await asyncio.to_thread(run)
