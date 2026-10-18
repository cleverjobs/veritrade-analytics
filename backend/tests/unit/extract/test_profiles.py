from __future__ import annotations

import pytest

from veritrade.extract import profiles
from veritrade.extract.dom import parse
from veritrade.extract.profiles import Fate, Profile, refine, source_for_url, stop_marker, triage
from veritrade.extract.types import PARSERS, ParserName

RELEASE = Profile(source="pr", body="p", lede_paragraphs=2, min_words=3, stop_markers=("About ", "Contacts"))


def fates(texts: list[str], profile: Profile | None = RELEASE) -> list[str]:
    return [v.fate.value for v in triage(texts, profile)]


def test_sources_load_with_profiles_for_every_primary() -> None:
    sources = profiles.sources()
    primary = [s for s in sources.values() if s.status == "primary"]

    assert len(primary) == 10
    assert all(s.profile is not None and s.profile.body for s in primary)
    assert profiles.profile_for("sec") is not None
    assert profiles.profile_for("sec").min_words == 4  # type: ignore[union-attr]
    assert profiles.profile_for("apnews") is None
    assert profiles.profile_for(None) is None


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://finance.yahoo.com/news/x-1.html", "yahoo"),
        ("https://uk.finance.yahoo.com/news/x-1.html", "yahoo"),
        ("https://edition.cnn.com/2026/10/17/business/x", "cnn"),
        ("https://www.bbc.co.uk/news/articles/abc", "bbc"),
        ("https://FORTUNE.com./2026/10/17/x/", "fortune"),
        ("https://apnews.com/article/x", "apnews"),
        ("https://notfortune.com/x", None),
        ("https://yahoo.com/news/x", None),
        ("not a url", None),
    ],
)
def test_source_for_url_matches_listed_domains_and_subdomains(url: str, key: str | None) -> None:
    assert source_for_url(url) == key


def test_short_paragraphs_and_boilerplate_lines_are_dropped() -> None:
    texts = ["CITY, Oct 1 /Wire/ -- Acme announced results.", "Acme Corp", "Story continues", ""]

    assert fates(texts) == ["keep", "short", "boilerplate", "empty"]
    assert triage(texts, RELEASE)[2].phrase == "story continues"


def test_stop_markers_never_cut_the_lede() -> None:
    texts = [
        "About Acme: the lede that starts like a marker.",
        "Second lede paragraph with words.",
        "Third paragraph of the body.",
        "About Acme",
        "Acme makes things for many people.",
    ]

    assert fates(texts) == ["keep", "keep", "keep", "stop", "after_stop"]
    assert refine(texts, RELEASE) == texts[:3]


def test_name_style_marker_needs_a_capital_or_a_short_heading() -> None:
    markers = ("About ",)
    long_tail = " of the 2,000 staff will move to the new site next year, the company said."

    assert stop_marker("About 40%" + long_tail, markers) is None
    assert stop_marker("About half" + long_tail, markers) is None
    assert stop_marker("About the Company", markers) == "About "
    assert stop_marker("About Acme Corp. Acme is a maker of tools used by millions.", markers) == "About "
    assert stop_marker("about acme", markers) == "About "


def test_generic_stops_apply_without_a_profile() -> None:
    texts = ["Lede one has words.", "Lede two has words.", "Body.", "Forward-Looking Statements", "Legal."]

    assert fates(texts, None) == ["keep", "keep", "keep", "stop", "after_stop"]


def test_inner_matches_of_a_selector_list_are_not_duplicated() -> None:
    texts = ["Outer item with an inner paragraph and more", "inner paragraph", "and more", "Next one."]

    assert fates(texts, Profile(source="x", body="p")) == ["keep", "nested", "nested", "keep"]


@pytest.mark.parametrize("parser", PARSERS)
def test_profile_strategy_applies_drops_and_stops(parser: ParserName) -> None:
    page = """<html><body><h1>Headline here</h1><div id="article-body">
      <p>Shares of Acme <span class="ticker-mention">( ACME -1.2% )</span> rose after the update today.</p>
      <p>Second paragraph of the story has words.</p>
      <div class="image"><p>Caption text for an image.</p></div>
      <p>Third paragraph of the story has words.</p>
      <p>Before you buy stock in Acme, consider this pitch.</p>
      <p>Pitch body that must not appear.</p>
    </div></body></html>"""
    paragraphs = profiles.extract(parse(page, parser), "https://www.fool.com/investing/x/", "fool")

    assert paragraphs == [
        "Shares of Acme rose after the update today.",
        "Second paragraph of the story has words.",
        "Third paragraph of the story has words.",
    ]


def test_profile_strategy_is_empty_for_unknown_sources() -> None:
    assert profiles.extract(parse("<p>text</p>"), "https://example.com/", None) == []


def test_fate_values_are_stable() -> None:
    assert {f.value for f in Fate} == {
        "keep", "empty", "nested", "short", "boilerplate", "contact", "stop", "after_stop"
    }  # fmt: skip


def test_short_lines_with_an_email_are_dropped_but_do_not_end_the_body() -> None:
    texts = [
        "Lede one has words.",
        "Send tips to desk@example.com.",
        "Lede two has words.",
        "Jane Doe, Acme Corp, jane.doe@acme.co.uk, +1 555 0100",
        "Body continues here with words.",
        "Write to desk@example.com if you have a question about this long story that runs on and on.",
    ]

    assert fates(texts, None) == ["keep", "contact", "keep", "contact", "keep", "keep"]
