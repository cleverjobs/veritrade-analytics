"""Opt-in smoke against real publishers: `uv run pytest -m live backend/tests/live/test_fetch_live.py`."""

from __future__ import annotations

import json
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import veritrade
from veritrade.config import Settings
from veritrade.fetch.client import HttpFetcher

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "evals" / "data" / "source_probe.json"
SOURCES = Path(veritrade.__file__).parent / "data" / "sources.toml"


def primary_urls() -> list[tuple[str, str]]:
    """First article per primary source that the probe fetched successfully."""
    sources: dict[str, Any] = tomllib.loads(SOURCES.read_text())["sources"]
    probed: dict[str, Any] = json.loads(PROBE.read_text())["sources"]
    picks = []
    for key, source in sorted(sources.items(), key=lambda item: item[1].get("rank", 99)):
        if source["status"] != "primary":
            continue
        ok = [
            a["fetch"]["url"]
            for a in probed.get(key, {}).get("articles", [])
            if a["fetch"]["outcome"] == "ok"
        ]
        if ok:
            picks.append((key, ok[0]))
    return picks


@pytest.fixture
async def fetcher() -> AsyncIterator[HttpFetcher]:
    client = HttpFetcher(Settings())
    yield client
    await client.aclose()


@pytest.mark.parametrize(("source", "url"), primary_urls(), ids=[key for key, _ in primary_urls()])
async def test_primary_source_article_fetches(fetcher: HttpFetcher, source: str, url: str) -> None:
    result = await fetcher.get(url)
    assert result.status == 200, source
    # EDGAR exhibits are SGML-wrapped with upper-case tags.
    assert result.html.lower().count("<p") >= 3, source
    assert result.fetch_ms < Settings().fetch_deadline_s * 1000
