"""Off-the-shelf extractors behind one signature; each is imported on first use and brings its own parser."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

from veritrade.extract.dom import normalize, parse, strip_xml_declaration
from veritrade.extract.types import ExtractorName

_EDGAR_TEXT = re.compile(r"\A\s*<DOCUMENT>.*?<TEXT>(.*?)(?:</TEXT>|\Z)", re.IGNORECASE | re.DOTALL)


def unwrap_edgar(html: str) -> str:
    """The <TEXT> body of an EDGAR SGML document (trafilatura rejects the envelope); other pages as given."""
    match = _EDGAR_TEXT.match(html)
    return match.group(1).strip() if match else html


def _lines(text: str | None) -> list[str]:
    return [line for raw in (text or "").splitlines() if (line := normalize(raw))]


def _trafilatura(html: str, url: str, *, precision: bool) -> list[str]:
    import trafilatura

    text = trafilatura.extract(
        html,
        url=url,
        favor_precision=precision,
        include_comments=False,
        include_tables=False,
        deduplicate=False,
        with_metadata=False,
        output_format="txt",
    )
    return _lines(text)


def _readability(html: str, url: str) -> list[str]:
    from readability import Document

    fragment = Document(strip_xml_declaration(html), url=url).summary(html_partial=True)
    return [block.text for block in parse(fragment, "lxml").iter_blocks()]


def _newspaper(html: str, url: str) -> list[str]:
    with warnings.catch_warnings():
        # Import-time notice about the optional NLP extra, which plain text extraction never uses.
        warnings.simplefilter("ignore", UserWarning)
        from newspaper import Article

    # Parse the given HTML only: no download, no image probing, no on-disk URL memo.
    article = Article(
        url, language="en", fetch_images=False, follow_meta_refresh=False, memorize_articles=False
    )
    article.download(input_html=html)
    article.parse()
    return _lines(article.text)


def _justext(html: str, url: str) -> list[str]:
    import justext

    found = justext.justext(html.encode("utf-8"), justext.get_stoplist("English"), encoding="utf-8")
    return [text for p in found if not p.is_boilerplate and (text := normalize(p.text))]


def _strip_tags(html: str, url: str) -> list[str]:
    """Baseline: every visible text run on the page, scripts and styles removed."""
    return [block.text for block in parse(html, "selectolax").iter_blocks()]


ADAPTERS: dict[ExtractorName, Callable[[str, str], list[str]]] = {
    "trafilatura": lambda html, url: _trafilatura(html, url, precision=False),
    "trafilatura-precision": lambda html, url: _trafilatura(html, url, precision=True),
    "readability": _readability,
    "newspaper": _newspaper,
    "justext": _justext,
    "strip-tags": _strip_tags,
}


def extract(name: ExtractorName, html: str, url: str) -> list[str]:
    return ADAPTERS[name](unwrap_edgar(html), url)
