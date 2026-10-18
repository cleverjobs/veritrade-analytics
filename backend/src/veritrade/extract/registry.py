"""Single entry point from HTML to Article; parse and extract time are measured separately."""

from __future__ import annotations

import html as html_lib
import re
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from veritrade.errors import ExtractionEmpty
from veritrade.extract import cascade, density, jsonld, profiles, third_party
from veritrade.extract.dom import Dom, head_title, normalize, parse, scan_json_ld
from veritrade.extract.profiles import source_for_url
from veritrade.extract.types import PARSER_AWARE_EXTRACTORS, Article, ExtractorName, ParserName

__all__ = ["MIN_WORDS", "extract", "source_for_url"]

MIN_WORDS = 40

OWN: dict[ExtractorName, Callable[[Dom, str, str | None], list[str]]] = {
    "jsonld": jsonld.extract,
    "profile": profiles.extract,
    "density": density.extract,
}
_PUBLISHED_META = re.compile(
    r"""<meta\b[^>]*?(?:property|name|itemprop)\s*=\s*["']"""
    r"""(?:article:published_time|datePublished|pubdate|publishdate)["'][^>]*>""",
    re.IGNORECASE,
)
_CONTENT_ATTR = re.compile(r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)


def _published_meta(html: str) -> str | None:
    if (meta := _PUBLISHED_META.search(html)) and (content := _CONTENT_ATTR.search(meta.group(0))):
        return normalize(html_lib.unescape(content.group(1) or content.group(2) or "")) or None
    return None


def _dom_title(dom: Dom, source: str | None) -> str:
    profile = profiles.profile_for(source)
    if profile is not None and profile.title:
        for node in dom.select(profile.title):
            if text := node.text():
                return text
    return dom.title()


def extract(
    html: str,
    url: str,
    *,
    parser: ParserName = "selectolax",
    extractor: ExtractorName = "cascade",
) -> Article:
    """Raises ExtractionEmpty when fewer than MIN_WORDS words come out."""
    source = source_for_url(url)
    start = time.perf_counter()
    dom: Dom | None = None
    if extractor in PARSER_AWARE_EXTRACTORS:
        dom = parse(html, parser)
        if extractor == "cascade":
            method, paragraphs = cascade.run(dom, url, source)
        else:
            method, paragraphs = extractor, OWN[extractor](dom, url, source)
        objects = dom.json_ld()
    else:
        method, paragraphs = extractor, third_party.extract(extractor, html, url)
        objects = scan_json_ld(html)
    words = sum(len(p.split()) for p in paragraphs)
    if words < MIN_WORDS:
        raise ExtractionEmpty(
            f"The {extractor} extractor found {words} words on {urlsplit(url).hostname or url}; "
            f"at least {MIN_WORDS} are needed",
            hint="Try another extractor (for example trafilatura), or check that the URL is a single "
            "article rather than a section or video page.",
        )
    title = jsonld.headline(objects) or (_dom_title(dom, source) if dom is not None else head_title(html))
    published_at = jsonld.date_published(objects) or _published_meta(html)
    total_ms = (time.perf_counter() - start) * 1000
    parse_ms = dom.parse_ms if dom is not None else 0.0
    return Article(
        url=url,
        title=title,
        paragraphs=tuple(paragraphs),
        published_at=published_at,
        source=source,
        method=method,
        parser=parser if dom is not None else None,
        extractor=extractor,
        parse_ms=parse_ms,
        extract_ms=total_ms - parse_ms,
        paywalled=jsonld.is_paywalled(objects),
    )
