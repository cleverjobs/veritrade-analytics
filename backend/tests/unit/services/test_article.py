from __future__ import annotations

import threading
from typing import Any

import pytest

from veritrade.extract.types import ExtractorName, ParserName
from veritrade.fetch import FetchResult
from veritrade.services import article as article_service

PAGE = FetchResult(
    requested_url="https://news.test/a",
    final_url="https://news.test/a",
    status=200,
    html="<html></html>",
    content_type="text/html",
    fetch_ms=1.0,
)


@pytest.mark.parametrize(
    ("parser", "extractor", "inline"),
    [
        ("selectolax", "cascade", True),
        ("lxml", "density", True),
        ("bs4", "cascade", False),
        ("bs4-lxml", "jsonld", False),
        ("selectolax", "trafilatura", False),
    ],
)
async def test_slow_extractors_run_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, parser: ParserName, extractor: ExtractorName, inline: bool
) -> None:
    threads: list[threading.Thread] = []

    def fake_extract(*_: Any, **__: Any) -> str:
        threads.append(threading.current_thread())
        return "article"

    monkeypatch.setattr(article_service, "extract", fake_extract)
    loop_thread = threading.current_thread()
    result = await article_service.extract_article(PAGE, parser=parser, extractor=extractor)
    assert result == "article"
    assert (threads == [loop_thread]) is inline
