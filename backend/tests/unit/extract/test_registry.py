from __future__ import annotations

import json

import pytest

from veritrade.errors import ErrorCode, ExtractionEmpty
from veritrade.extract import registry
from veritrade.extract.registry import extract, source_for_url
from veritrade.extract.types import PARSERS, ExtractorName, ParserName

SENTENCE = "The regional office said the new plan was shared with the board during the week."
BODY = "".join(f"<p>{SENTENCE} Item {i}.</p>" for i in range(8))


def page(ld: dict[str, object] | None = None, head: str = "") -> str:
    script = f'<script type="application/ld+json">{json.dumps(ld)}</script>' if ld else ""
    return f"<html><head>{script}{head}</head><body><div class='story'>{BODY}</div></body></html>"


@pytest.mark.parametrize("parser", PARSERS)
def test_article_fields_from_json_ld(parser: ParserName) -> None:
    ld = {
        "@type": "NewsArticle",
        "headline": "LD title",
        "datePublished": "2026-10-17",
        "isAccessibleForFree": False,
    }

    article = extract(page(ld), "https://example.com/a", parser=parser, extractor="density")

    assert (article.title, article.published_at, article.paywalled) == ("LD title", "2026-10-17", True)
    assert (article.source, article.method, article.parser, article.extractor) == (
        None,
        "density",
        parser,
        "density",
    )
    assert len(article.paragraphs) == 8
    assert article.parse_ms > 0.0
    assert article.extract_ms > 0.0


def test_fallbacks_to_meta_tags() -> None:
    head = (
        '<meta property="og:title" content="OG title"><title>Tag title</title>'
        '<meta content="2026-10-16T08:00:00Z" property="article:published_time">'
    )

    article = extract(page(head=head), "https://example.com/a")

    assert (article.title, article.published_at, article.paywalled) == (
        "OG title",
        "2026-10-16T08:00:00Z",
        False,
    )
    assert article.method == "density"


def test_profile_title_selector_wins_over_og_title_for_known_sources() -> None:
    html = (
        page(head='<meta property="og:title" content="OG title">')
        .replace(
            "<body>", '<body><h1 class="cover-title">Visible headline</h1><div data-testid="article-body">'
        )
        .replace("</body>", "</div></body>")
    )

    article = extract(html, "https://finance.yahoo.com/news/acme-1.html", extractor="profile")

    assert (article.source, article.title, article.method) == ("yahoo", "Visible headline", "profile")


@pytest.mark.parametrize("extractor", ["trafilatura", "strip-tags"])
def test_third_party_reports_no_parser(extractor: ExtractorName) -> None:
    html = page({"@type": "NewsArticle", "headline": "LD"})

    article = extract(html, "https://example.com/a", extractor=extractor)

    assert (article.parser, article.parse_ms, article.method, article.title) == (None, 0.0, extractor, "LD")
    assert article.word_count >= registry.MIN_WORDS


def test_too_little_text_raises_extraction_empty() -> None:
    with pytest.raises(ExtractionEmpty) as caught:
        extract("<html><body><p>Only a few words.</p></body></html>", "https://example.com/a")

    assert caught.value.code is ErrorCode.EXTRACTION_EMPTY
    assert caught.value.http_status == 422
    assert "example.com" in caught.value.message
    assert caught.value.hint


def test_source_lookup_is_reexported() -> None:
    assert source_for_url("https://www.cnbc.com/2026/10/17/x.html") == "cnbc"
