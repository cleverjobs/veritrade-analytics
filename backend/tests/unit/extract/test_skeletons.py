"""Every extractor on every skeleton: body-marker recall and leakage of non-body text."""

from __future__ import annotations

import pytest

from veritrade.extract import third_party
from veritrade.extract.registry import extract
from veritrade.extract.types import PARSERS, ExtractorName, ParserName

from ._skeletons import BODY_MARKER, OTHER_MARKER, Skeleton, skeletons

SKELETONS = skeletons()
OWN_EXACT: tuple[ExtractorName, ...] = ("jsonld", "profile", "cascade")
THIRD_PARTY: tuple[ExtractorName, ...] = (
    "trafilatura",
    "trafilatura-precision",
    "readability",
    "newspaper",
    "justext",
    "strip-tags",
)


def _ids(skeleton: Skeleton) -> str:
    return skeleton.source


def test_every_primary_source_has_a_skeleton() -> None:
    assert len(SKELETONS) >= 10
    assert all(s.body_markers for s in SKELETONS)


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("extractor", OWN_EXACT)
@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_own_strategy_recovers_the_body_without_leakage(
    skeleton: Skeleton, extractor: ExtractorName, parser: ParserName
) -> None:
    if extractor == "jsonld" and not skeleton.expected["jsonld_body"]:
        pytest.skip("page has no JSON-LD articleBody")
    article = extract(skeleton.html, skeleton.url, parser=parser, extractor=extractor)

    assert article.source == skeleton.source
    assert skeleton.recall(article.text) >= 0.9
    assert OTHER_MARKER.findall(article.text) == []
    if title_marker := skeleton.expected["title_marker"]:
        assert title_marker in article.title


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_density_finds_most_of_the_body(skeleton: Skeleton, parser: ParserName) -> None:
    """A leaked line holds only non-body text; an inline chip inside a body paragraph does not count."""
    article = extract(skeleton.html, skeleton.url, parser=parser, extractor="density")

    leaked = [p for p in article.paragraphs if OTHER_MARKER.search(p) and not BODY_MARKER.search(p)]
    assert skeleton.recall(article.text) >= 0.8
    assert len(leaked) <= 0.1 * len(article.paragraphs)


@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_cascade_takes_jsonld_only_when_the_page_has_a_body(skeleton: Skeleton) -> None:
    article = extract(skeleton.html, skeleton.url, extractor="cascade")

    assert article.method == ("jsonld" if skeleton.expected["jsonld_body"] else "profile")
    # A JSON-LD hit never builds a DOM.
    assert (article.parse_ms == 0.0) is (article.method == "jsonld")


@pytest.mark.parametrize("extractor", THIRD_PARTY)
@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_third_party_smoke(skeleton: Skeleton, extractor: ExtractorName) -> None:
    paragraphs = third_party.extract(extractor, skeleton.html, skeleton.url)

    assert paragraphs
    assert all(isinstance(p, str) and p for p in paragraphs)
