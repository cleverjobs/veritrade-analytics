"""Pure, network-free classification of a fetched page into blocked, consent-walled or paywalled."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from veritrade.errors import ConsentRequired, Paywalled, SourceBlocked, VtError

BLOCK_STATUSES = frozenset({401, 403, 429, 451})
# Interstitials are a few KB with markers near the top; a bounded window keeps 6 MB pages cheap.
SCAN_WINDOW = 65_536
PAYWALL_MIN_WORDS = 200
CONSENT_HOSTS = frozenset({"consent.yahoo.com", "guce.yahoo.com"})

# (vendor, lowercase needle) pairs; each needle only occurs on the vendor's interstitial, not on
# normal pages it protects (e.g. Cloudflare's passive /challenge-platform/scripts/ beacon is excluded).
_BODY_MARKERS: tuple[tuple[str, str], ...] = (
    ("Cloudflare", "<title>just a moment...</title>"),
    ("Cloudflare", "_cf_chl_opt"),
    ("Cloudflare", "/cdn-cgi/challenge-platform/h/"),
    ("Cloudflare", "attention required! | cloudflare"),
    ("DataDome", "captcha-delivery.com"),
    ("PerimeterX", "px-captcha"),
    ("PerimeterX", "press &amp; hold"),
    ("PerimeterX", "press & hold"),
    ("Akamai", "sec-if-cpt-container"),
    ("Akamai", "errors.edgesuite.net"),
    ("Imperva", "_incapsula_resource"),
)
_CONSENT_MARKERS = ('class="consent-form"', "consent.yahoo.com/v2/collectconsent")

_PAID_ELSEWHERE = "try Yahoo Finance, CNBC or the company's own press release."
# Host suffix -> hint, matched on the final URL so a syndicated copy is suggested where one usually exists.
_HINTS: dict[type[VtError], tuple[str, dict[str, str]]] = {
    SourceBlocked: (
        "The publisher refused an automated request; try the same story from another source.",
        {
            "reuters.com": "Reuters blocks automated clients; its wire stories are usually syndicated on "
            "Yahoo Finance.",
            "apnews.com": "AP News serves a bot challenge to automated clients; AP stories are syndicated on "
            "Yahoo Finance and many local outlets.",
            "businesswire.com": "Business Wire serves a bot challenge; the same release is usually on Yahoo "
            "Finance or the issuer's investor-relations site.",
            "benzinga.com": "Benzinga blocks automated clients; its stories are syndicated on Yahoo Finance.",
            "investing.com": "Investing.com blocks automated clients; the underlying wire or company news is "
            "usually on Yahoo Finance or CNBC.",
            "bloomberg.com": f"Bloomberg blocks automated clients; {_PAID_ELSEWHERE}",
        },
    ),
    ConsentRequired: (
        "The publisher demands a cookie-consent choice first; try another source for the story.",
        {
            "yahoo.com": "Yahoo's EU consent page could not be completed automatically; open the article in "
            "a browser or run the service from a non-EU network.",
        },
    ),
    Paywalled: (
        "The article is behind a paywall; pick a free source such as Yahoo Finance or a press-release wire.",
        {
            "wsj.com": f"WSJ is paywalled; {_PAID_ELSEWHERE}",
            "barrons.com": f"Barron's is paywalled; {_PAID_ELSEWHERE}",
            "ft.com": f"The FT is paywalled; {_PAID_ELSEWHERE}",
            "bloomberg.com": f"Bloomberg is paywalled; {_PAID_ELSEWHERE}",
        },
    ),
}


def classify(status: int, headers: Mapping[str, str], html: str, final_url: str) -> None:
    """Raise SourceBlocked, ConsentRequired or Paywalled; other 4xx/5xx statuses are the caller's to map."""
    host = (urlsplit(final_url).hostname or "").lower()
    lowered = {key.lower(): value for key, value in headers.items()}
    window = html[:SCAN_WINDOW].lower()

    if host in CONSENT_HOSTS or any(marker in window for marker in _CONSENT_MARKERS):
        raise ConsentRequired(
            f"{host or final_url} answered with a cookie-consent wall",
            hint=_hint(ConsentRequired, host),
            upstream_status=status,
        )

    vendor = _challenge_vendor(lowered, window)
    if status in BLOCK_STATUSES or vendor:
        edge = _edge_vendor(lowered)
        by = f", {vendor} bot protection" if vendor else f", {edge} edge" if edge else ""
        raise SourceBlocked(
            f"{host or final_url} refused the request (HTTP {status}{by})",
            hint=_hint(SourceBlocked, host),
            retryable=status == 429,
            upstream_status=status,
        )

    if "isAccessibleForFree" in html:
        free, body_words = jsonld_access(html)
        # Metered sites flag every article as not free yet serve it whole; only a short body is a paywall.
        if free is False and body_words < PAYWALL_MIN_WORDS and paragraph_words(html) < PAYWALL_MIN_WORDS:
            raise Paywalled(
                f"{host or final_url} marks the article as not free and serves only a teaser",
                hint=_hint(Paywalled, host),
                upstream_status=status,
            )


def _challenge_vendor(headers: Mapping[str, str], window: str) -> str | None:
    if headers.get("cf-mitigated", "").lower() == "challenge":
        return "Cloudflare"
    if "x-dd-b" in headers:
        return "DataDome"
    if "<title>access denied</title>" in window and "reference #" in window:
        return "Akamai"
    for vendor, needle in _BODY_MARKERS:
        if needle in window:
            return vendor
    return None


def _edge_vendor(headers: Mapping[str, str]) -> str | None:
    """Which CDN answered, so a bare 403 still says who refused it."""
    server = headers.get("server", "").lower()
    if "cf-ray" in headers or server == "cloudflare":
        return "Cloudflare"
    if "akamai" in server:
        return "Akamai"
    if "x-datadome" in headers:
        return "DataDome"
    return None


def _hint(kind: type[VtError], host: str) -> str:
    default, by_host = _HINTS[kind]
    for suffix, hint in by_host.items():
        if host == suffix or host.endswith("." + suffix):
            return hint
    return default


def jsonld_blocks(html: str) -> Iterator[Any]:
    """Yield every parseable JSON-LD payload; plain str.find keeps multi-MB pages at a few ms."""
    pos = 0
    while (start := html.find("application/ld+json", pos)) != -1:
        open_end = html.find(">", start)
        close = html.find("</script", open_end)
        if open_end == -1 or close == -1:
            return
        pos = close
        try:
            yield json.loads(html[open_end + 1 : close])
        except ValueError:
            continue


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def jsonld_nodes(html: str) -> Iterator[dict[str, Any]]:
    """Every JSON object nested anywhere in the page's JSON-LD (covers @graph and arrays)."""
    for block in jsonld_blocks(html):
        yield from _walk(block)


def jsonld_access(html: str) -> tuple[bool | None, int]:
    """isAccessibleForFree across JSON-LD (False wins) and the longest articleBody word count."""
    free: bool | None = None
    words = 0
    for node in jsonld_nodes(html):
        value = node.get("isAccessibleForFree")
        if value is not None:
            parsed = value if isinstance(value, bool) else str(value).strip().lower() == "true"
            free = parsed if free is None else free and parsed
        body = node.get("articleBody")
        if isinstance(body, str):
            words = max(words, len(body.split()))
    return free, words


_P_RE = re.compile(r"<p[\s>].*?</p>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def paragraph_words(html: str) -> int:
    """Rough visible word count of <p> elements; only used on the rare not-free path."""
    return sum(len(_TAG_RE.sub(" ", match).split()) for match in _P_RE.findall(html))
