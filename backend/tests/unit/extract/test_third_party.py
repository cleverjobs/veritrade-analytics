from __future__ import annotations

import socket
from typing import Any

import pytest

from veritrade.extract import third_party
from veritrade.extract.registry import extract
from veritrade.extract.types import EXTRACTORS, PARSER_AWARE_EXTRACTORS, ExtractorName

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


PARAGRAPH = (
    "Ostrava Logistics Inc. today reported third quarter revenue of $41.2 million, up 12% from the prior "
    "year, driven by higher freight volumes across its central European network. Operating income rose "
    "to $6.1 million and the company reaffirmed its full year outlook."
)
EXHIBIT = (
    "<html><head><title>Exhibit 99.1</title></head><body>" + f"<p>{PARAGRAPH}</p>" * 4 + "</body></html>"
)
ENVELOPE = (
    "<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>ex991.htm\n<DESCRIPTION>PRESS RELEASE\n<TEXT>\n"
    + EXHIBIT
    + "\n</TEXT>\n</DOCUMENT>\n"
)
SEC_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/ex991.htm"


def test_unwrap_edgar_returns_the_text_body_only() -> None:
    assert third_party.unwrap_edgar(ENVELOPE) == EXHIBIT
    assert third_party.unwrap_edgar(ENVELOPE.replace("\n</TEXT>\n</DOCUMENT>\n", "")) == EXHIBIT
    assert third_party.unwrap_edgar(EXHIBIT) == EXHIBIT
    inline = "<html><body><p><DOCUMENT><TEXT>quoted</p></body></html>"
    assert third_party.unwrap_edgar(inline) == inline


@pytest.mark.parametrize("name", ["trafilatura", "trafilatura-precision", "readability", "justext"])
def test_third_party_extractors_read_edgar_exhibits(name: ExtractorName) -> None:
    article = extract(ENVELOPE, SEC_URL, extractor=name)

    assert article.word_count == len(PARAGRAPH.split()) * 4
    assert "EX-99.1" not in article.text


def test_strip_tags_returns_every_visible_run() -> None:
    page = "<html><head><title>T</title><style>p{}</style></head><body><p>One <b>two</b></p><div>three</div>"

    assert third_party.extract("strip-tags", page, "https://example.com/") == ["One two", "three"]
