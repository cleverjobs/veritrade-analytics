from __future__ import annotations

import socket
from typing import Any

import pytest

from veritrade.extract import third_party
from veritrade.extract.types import EXTRACTORS, PARSER_AWARE_EXTRACTORS

from ._skeletons import skeletons


def test_every_non_own_extractor_has_an_adapter() -> None:
    assert set(third_party.ADAPTERS) == set(EXTRACTORS) - PARSER_AWARE_EXTRACTORS


def test_adapters_never_touch_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an extractor tried to open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    skeleton = next(s for s in skeletons() if s.source == "fool")

    for name in third_party.ADAPTERS:
        assert third_party.extract(name, skeleton.html, skeleton.url)


def test_strip_tags_returns_every_visible_run() -> None:
    page = "<html><head><title>T</title><style>p{}</style></head><body><p>One <b>two</b></p><div>three</div>"

    assert third_party.extract("strip-tags", page, "https://example.com/") == ["One two", "three"]
