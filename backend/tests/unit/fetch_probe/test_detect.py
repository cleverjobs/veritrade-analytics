from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from veritrade.errors import ConsentRequired, Paywalled, SourceBlocked, VtError
from veritrade.fetch.detect import SCAN_WINDOW, classify, jsonld_access, jsonld_blocks

ARTICLE_URL = "https://news.example.com/2026/01/01/story"
FILLER = "lorem ipsum dolor sit amet " * 10


def page(body: str = "", head: str = "", paragraphs: int = 12) -> str:
    text = "".join(f"<p>{FILLER}</p>" for _ in range(paragraphs))
    return f"<html><head><title>Story</title>{head}</head><body>{body}{text}</body></html>"


def ld(payload: object) -> str:
    return f'<script type="application/ld+json">{json.dumps(payload)}</script>'


def raised(
    kind: type[VtError], status: int, html: str, url: str, headers: Mapping[str, str] | None = None
) -> VtError:
    with pytest.raises(kind) as caught:
        classify(status, headers or {}, html, url)
    assert caught.value.hint
    return caught.value


def test_ordinary_article_passes() -> None:
    classify(200, {"content-type": "text/html"}, page(), ARTICLE_URL)


@pytest.mark.parametrize("status", [401, 403, 429, 451])
def test_refusal_statuses_are_source_blocked(status: int) -> None:
    error = raised(SourceBlocked, status, "<html></html>", ARTICLE_URL)
    assert error.upstream_status == status
    assert error.retryable is (status == 429)


def test_cloudflare_managed_challenge_names_vendor_and_suggests_syndication() -> None:
    challenge = "<title>Just a moment...</title><script>window._cf_chl_opt={}</script>"
    headers = {"cf-mitigated": "challenge", "server": "cloudflare"}
    error = raised(SourceBlocked, 403, challenge, "https://apnews.com/x", headers)
    assert "Cloudflare bot protection" in error.message
    assert "Yahoo Finance" in str(error.hint)


def test_bare_403_names_the_edge() -> None:
    error = raised(SourceBlocked, 403, "<html></html>", ARTICLE_URL, {"Server": "AkamaiGHost"})
    assert "Akamai edge" in error.message


@pytest.mark.parametrize(
    ("headers", "body", "vendor"),
    [
        ({}, "<title>Just a moment...</title>", "Cloudflare"),
        ({}, '<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></script>', "Cloudflare"),
        ({}, "<title>Attention Required! | Cloudflare</title>", "Cloudflare"),
        ({}, '<script src="https://geo.captcha-delivery.com/captcha/"></script>', "DataDome"),
        ({"X-DD-B": "1"}, "<html></html>", "DataDome"),
        ({}, '<div id="px-captcha"></div>', "PerimeterX"),
        ({}, "<p>Press &amp; Hold to confirm you are a human</p>", "PerimeterX"),
        ({}, "<title>Access Denied</title><p>Reference #18.abc</p>", "Akamai"),
        ({}, '<div id="sec-if-cpt-container" role="main"></div>', "Akamai"),
        ({}, '<script src="/_Incapsula_Resource?x=1"></script>', "Imperva"),
    ],
)
def test_challenge_pages_served_with_200_are_blocked(headers: dict[str, str], body: str, vendor: str) -> None:
    error = raised(SourceBlocked, 200, f"<html>{body}</html>", ARTICLE_URL, headers)
    assert f"{vendor} bot protection" in error.message


def test_business_wire_challenge_gets_its_hint() -> None:
    error = raised(
        SourceBlocked, 200, '<div id="sec-if-cpt-container"></div>', "https://www.businesswire.com/"
    )
    assert "Business Wire" in str(error.hint)


def test_passive_cloudflare_beacon_is_not_a_challenge() -> None:
    beacon = '<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>'
    classify(200, {"cf-ray": "abc", "server": "cloudflare"}, page(head=beacon), ARTICLE_URL)


def test_markers_past_the_scan_window_are_ignored() -> None:
    classify(200, {}, page() + " " * SCAN_WINDOW + "<p>Press & Hold</p>", ARTICLE_URL)


def test_hint_matches_host_suffix_only() -> None:
    assert "Reuters" in str(raised(SourceBlocked, 403, "", "https://www.reuters.com/markets/x").hint)
    assert "Reuters" not in str(raised(SourceBlocked, 403, "", "https://notreuters.com/x").hint)


def test_landing_on_the_consent_host_is_consent_required() -> None:
    error = raised(ConsentRequired, 200, "<html></html>", "https://consent.yahoo.com/v2/collectConsent?s=1")
    assert "Yahoo" in str(error.hint)


def test_consent_form_markup_is_consent_required(consent_page: str) -> None:
    raised(ConsentRequired, 200, consent_page, "https://finance.yahoo.com/news/x.html")


@pytest.mark.parametrize("flag", [False, "False", "false"])
def test_not_free_teaser_is_paywalled(flag: object) -> None:
    html = page(head=ld({"@type": "NewsArticle", "isAccessibleForFree": flag}), paragraphs=2)
    assert "WSJ" in str(raised(Paywalled, 200, html, "https://www.wsj.com/articles/x").hint)


def test_metered_page_with_full_visible_text_is_not_paywalled() -> None:
    classify(200, {}, page(head=ld({"@type": "NewsArticle", "isAccessibleForFree": False})), ARTICLE_URL)


def test_metered_page_with_full_jsonld_body_is_not_paywalled() -> None:
    graph = {"@graph": [{"@type": "NewsArticle", "isAccessibleForFree": "false", "articleBody": FILLER * 5}]}
    classify(200, {}, page(head=ld(graph), paragraphs=0), ARTICLE_URL)


def test_free_short_page_is_not_paywalled() -> None:
    classify(
        200,
        {},
        page(head=ld({"@type": "NewsArticle", "isAccessibleForFree": True}), paragraphs=1),
        ARTICLE_URL,
    )


def test_jsonld_scan_skips_broken_blocks_and_walks_graphs() -> None:
    html = (
        '<script type="application/ld+json">{not json</script>'
        + ld([{"@type": "WebPage"}, {"@graph": [{"@type": "NewsArticle", "isAccessibleForFree": True}]}])
        + ld({"@type": "NewsArticle", "isAccessibleForFree": False, "articleBody": "one two three"})
    )
    assert len(list(jsonld_blocks(html))) == 2
    assert jsonld_access(html) == (False, 3)
