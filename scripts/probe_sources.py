"""Live source probe: which candidate sources a plain httpx client can fetch, and what extractors need.

Run `uv run python scripts/probe_sources.py` (live) or `--reanalyze` (re-score the saved HTML offline).
Writes metadata only to evals/data/source_probe.json; raw HTML stays in evals/data/private/probe/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import statistics
import time
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import lxml.html
from selectolax.lexbor import LexborHTMLParser, LexborNode

from veritrade.errors import VtError
from veritrade.fetch import consent, detect
from veritrade.fetch.headers import accept_encoding, browser_headers
from veritrade_evals.discovery import (
    MIN_GAP_S,
    SEC_MIN_GAP_S,
    Politeness,
    edgar_exhibit,
    feed_links,
    normalize,
    pick,
    section_links,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES_TOML = ROOT / "backend/src/veritrade/data/sources.toml"
OUT_JSON = ROOT / "evals/data/source_probe.json"
RAW_DIR = ROOT / "evals/data/private/probe"

YAHOO_CONSENT_ENTRY = (
    "https://guce.yahoo.com/consent?brandType=eu&lang=en-US&done=https%3A%2F%2Ffinance.yahoo.com%2F"
)
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_REDIRECTS = 5
MAX_EDGAR_INDEXES = 25
MIN_P_WORDS = 6
USABLE_BODY_WORDS = 120
NEWS_TYPES = frozenset(
    {"NewsArticle", "Article", "ReportageNewsArticle", "AnalysisNewsArticle", "BlogPosting", "Report"}
)
FRAMEWORK_MARKERS = {
    "next_data": 'id="__NEXT_DATA__"',
    "next_flight": "self.__next_f",
    "nuxt": "__NUXT__",
    "initial_state": "__INITIAL_STATE__",
    "preloaded_state": "__PRELOADED_STATE__",
    "apollo_state": "__APOLLO_STATE__",
    "fusion": "Fusion.globalContent",
    "remix": "__remixContext",
    "sveltekit": "data-sveltekit",
}
NOISE_RE = re.compile(
    r"related|recirc|read-?more|more-?(?:on|from)|newsletter|sign-?up|subscribe|promo|advert|sponsor|"
    r"(?:^|[-_ ])ads?(?:$|[-_ ])|ticker|quote|chart|video|share|social|disclaimer|author|bio\b|embed|"
    r"caption|trending|taboola|outbrain|teaser|cta|paywall|premium|comment|feedback|toolbar",
    re.IGNORECASE,
)
PRESS_RELEASE_RE = re.compile(r"press|release|earnings|results|announc", re.IGNORECASE)
PAYWALL_RE = re.compile(r"paywall|subscriber|premium|metered|locked|regwall", re.IGNORECASE)
# Paragraph-start labels for press-release / filing boilerplate; only the label is recorded, never text.
PR_MARKERS = (
    ("forward-looking statements", "forward_looking"),
    ("forward looking statements", "forward_looking"),
    ("cautionary", "forward_looking"),
    ("safe harbor", "forward_looking"),
    ("about ", "about"),
    ("media contact", "contacts"),
    ("investor contact", "contacts"),
    ("investor relations", "contacts"),
    ("contacts", "contacts"),
    ("contact:", "contacts"),
    ("source:", "source_line"),
    ("source ", "source_line"),
    ("view original content", "view_original"),
    ("###", "end_marker"),
)
XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")
STABLE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
HASHY_TOKEN = re.compile(r"(?:^|[-_])(?=[a-z0-9]*\d)[a-z0-9]{5,}$|__[A-Za-z0-9]{4,}$")
SELECTOR_ATTRS = ("data-testid", "data-component", "itemprop", "data-module", "data-test")


# --- polite fetching (Politeness lives in veritrade_evals.discovery) -----------------------------------


@dataclass
class Fetched:
    url: str
    final_url: str
    status: int | None = None
    http_version: str | None = None
    content_type: str | None = None
    server: str | None = None
    redirects: list[dict[str, Any]] = field(default_factory=list)
    size_bytes: int = 0
    fetch_ms: float = 0.0
    consent_ms: float | None = None
    detect_ms: float | None = None
    outcome: str = "ok"
    error: dict[str, Any] | None = None
    body: bytes = field(default=b"", repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def meta(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("body")
        data.pop("headers")
        return data


async def polite_get(client: httpx.AsyncClient, polite: Politeness, url: str) -> tuple[httpx.Response, float]:
    """One request after the per-domain gap; the returned ms excludes the politeness wait."""
    lock = await polite.wait(url)
    started = time.perf_counter()
    try:
        response = await client.get(url, headers=browser_headers(url), follow_redirects=False)
        return response, (time.perf_counter() - started) * 1000
    finally:
        polite.done(url, lock)


async def fetch(client: httpx.AsyncClient, polite: Politeness, url: str, *, classify: bool = True) -> Fetched:
    """Manual redirect loop like the serving fetcher; a consent hop is answered once, then retried."""
    result = Fetched(url=url, final_url=url)
    network_ms = 0.0
    current, consent_used = url, False
    try:
        for _ in range(MAX_REDIRECTS + 1):
            response, hop_ms = await polite_get(client, polite, current)
            network_ms += hop_ms
            location = response.headers.get("location")
            if response.status_code in consent.REDIRECT_STATUSES and location:
                target = urljoin(current, location)
                result.redirects.append({"status": response.status_code, "to": target})
                if consent.is_consent_url(target) and not consent_used:
                    outcome = await consent.reject_all(client, target)
                    result.consent_ms = round(outcome.elapsed_ms, 1)
                    consent_used, current = True, url
                    continue
                current = target
                if consent.is_consent_url(target):
                    break
                continue
            break
        else:
            raise httpx.TooManyRedirects(f"more than {MAX_REDIRECTS} redirects")
        result.final_url = current
        result.status = response.status_code
        result.http_version = response.http_version
        result.content_type = response.headers.get("content-type")
        result.server = response.headers.get("server")
        result.body = response.content
        result.headers = dict(response.headers)
        result.size_bytes = len(response.content)
        if classify:
            started = time.perf_counter()
            try:
                detect.classify(response.status_code, result.headers, result.text, result.final_url)
            finally:
                result.detect_ms = round((time.perf_counter() - started) * 1000, 2)
    except VtError as exc:
        result.outcome = exc.code.value
        result.error = {"code": exc.code.value, "message": exc.message, "hint": exc.hint}
    except httpx.HTTPError as exc:
        result.outcome = "transport_error"
        result.error = {"code": type(exc).__name__, "message": str(exc) or type(exc).__name__, "hint": None}
    result.fetch_ms = round(network_ms, 1)
    return result


# --- discovery (URL and feed helpers live in veritrade_evals.discovery) ---------------------------------


async def discover(
    client: httpx.AsyncClient, polite: Politeness, key: str, cfg: dict[str, Any], limit: int
) -> tuple[list[str], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    entries = [(url, "feed") for url in cfg.get("feeds", [])]
    entries += [(url, "section") for url in cfg.get("sections", [])]
    for url, kind in entries:
        got = await fetch(client, polite, url, classify=kind == "section")
        if got.outcome == "ok" and got.status != 200:
            got.outcome = f"http_{got.status}"
        save_raw(key, f"discovery-{len(attempts) + 1}", got.body)
        links = feed_links(got.body) if kind == "feed" and got.outcome == "ok" else []
        if kind == "section" and got.outcome == "ok":
            links = section_links(got.text, got.final_url)
        record = got.meta() | {"kind": kind, "links_found": len(links)}
        attempts.append(record)
        if cfg.get("discovery") == "edgar-8k-ex99":
            exhibits = await resolve_edgar(client, polite, links, limit, record)
            if exhibits:
                return exhibits, attempts
            continue
        chosen = pick(links, cfg, limit)
        record["links_matching"] = len(chosen)
        if len(chosen) == limit:
            return chosen, attempts
    return [], attempts


async def resolve_edgar(
    client: httpx.AsyncClient, polite: Politeness, index_urls: list[str], limit: int, record: dict[str, Any]
) -> list[str]:
    releases: list[str] = []
    others: list[str] = []
    checked = 0
    for index_url in dict.fromkeys(index_urls[:MAX_EDGAR_INDEXES]):
        checked += 1
        got = await fetch(client, polite, normalize(index_url), classify=False)
        if got.status == 200 and (found := edgar_exhibit(got.text, got.final_url)):
            url, description = found
            # EX-99.1 is also used for financial statements and decks; press releases are what users paste.
            (releases if PRESS_RELEASE_RE.search(description) else others).append(normalize(url))
            if len(releases) == limit:
                break
    record["edgar_indexes_checked"] = checked
    record["edgar_press_releases"] = len(releases)
    exhibits = (releases + others)[:limit]
    record["links_matching"] = len(exhibits)
    return exhibits


def save_raw(key: str, name: str, body: bytes) -> None:
    folder = RAW_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.html").write_bytes(body)


# --- page analysis (metadata only: selectors, counts, labels; never text) -------------------------------


def words_of(node: LexborNode) -> int:
    return len(node.text(deep=True, separator=" ").split())


def lxml_doc(html: str) -> Any:
    """lxml refuses str input that still carries an XML encoding declaration (XHTML pages)."""
    return lxml.html.fromstring(XML_DECLARATION.sub("", html, count=1))


def stable(token: str) -> bool:
    return bool(STABLE_TOKEN.match(token)) and not HASHY_TOKEN.search(token) and len(token) <= 40


def css_for(node: LexborNode, depth: int = 0) -> str:
    """Simple selector (tag, id, attribute, class, descendant) valid in both selectolax and lxml cssselect."""
    tag = node.tag or "div"
    attrs = node.attributes
    node_id = attrs.get("id") or ""
    # Slug-like ids are generated from headings; skipping them keeps article text out of the report.
    if node_id and stable(node_id) and node_id.count("-") <= 2:
        return f"{tag}#{node_id}"
    for name in SELECTOR_ATTRS:
        value = attrs.get(name)
        if value and stable(value):
            return f'{tag}[{name}="{value}"]'
    classes = [cls for cls in (attrs.get("class") or "").split() if stable(cls)][:2]
    if classes:
        return tag + "".join(f".{cls}" for cls in classes)
    parent = node.parent
    if parent is not None and parent.tag not in (None, "html", "body") and depth < 3:
        return f"{css_for(parent, depth + 1)} {tag}"
    return tag


def jsonld_summary(html: str) -> dict[str, Any]:
    types: set[str] = set()
    body_words = 0
    free: Any = None
    has_part: list[str] = []
    headline = date = author = False
    for node in detect.jsonld_nodes(html):
        raw_type = node.get("@type")
        node_types = {raw_type} if isinstance(raw_type, str) else set(raw_type or [])
        types |= {str(t) for t in node_types}
        if node_types & NEWS_TYPES:
            headline |= bool(node.get("headline"))
            date |= bool(node.get("datePublished"))
            author |= bool(node.get("author"))
            body = node.get("articleBody")
            if isinstance(body, str):
                body_words = max(body_words, len(body.split()))
        if "isAccessibleForFree" in node:
            free = node["isAccessibleForFree"]
        if node.get("cssSelector"):
            has_part.append(str(node["cssSelector"]))
    return {
        "types": sorted(types),
        "news_article": bool(types & NEWS_TYPES),
        "article_body_words": body_words,
        "is_accessible_for_free": free,
        "headline": headline,
        "date_published": date,
        "author": author,
        "has_part_selectors": sorted(set(has_part)),
    }


def framework_json(html: str, tree: LexborHTMLParser) -> dict[str, Any]:
    found: dict[str, Any] = {name: marker in html for name, marker in FRAMEWORK_MARKERS.items()}
    if found["next_data"] and (node := tree.css_first("script#__NEXT_DATA__")) is not None:
        found["next_data_bytes"] = len(node.text() or "")
    return found


def lxml_counts(doc: Any, selector: str) -> tuple[int, int]:
    nodes = doc.cssselect(selector)
    words = sum(len(node.text_content().split()) for node in nodes)
    return len(nodes), words


def body_candidates(tree: LexborHTMLParser, doc: Any) -> tuple[list[dict[str, Any]], LexborNode | None]:
    """Ancestors of substantive <p>s ranked by the words they hold; best = tightest one with >= 85% of max."""
    agg: dict[int, list[Any]] = {}
    for p in tree.css("p"):
        words = words_of(p)
        if words < MIN_P_WORDS:
            continue
        node, depth = p.parent, 1
        while node is not None and node.tag not in (None, "html", "body") and depth <= 6:
            entry = agg.setdefault(node.mem_id, [node, 0, 0, depth])
            entry[1] += 1
            entry[2] += words
            entry[3] = min(entry[3], depth)
            node, depth = node.parent, depth + 1
    if not agg:
        return [], None
    ranked = sorted(agg.values(), key=lambda e: (-e[2], e[3]))
    top_words = ranked[0][2]
    best = min((e for e in ranked if e[2] >= 0.85 * top_words), key=lambda e: e[3])[0]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node, paragraphs, words, depth in ranked[:12]:
        selector = css_for(node)
        if selector in seen:
            continue
        seen.add(selector)
        sx_nodes = tree.css(f"{selector} p")
        lx_count, lx_words = lxml_counts(doc, f"{selector} p")
        out.append(
            {
                "selector": selector,
                "is_best": node.mem_id == best.mem_id,
                "p_direct_depth": depth,
                "substantive_p": paragraphs,
                "words": words,
                "matches": len(tree.css(selector)),
                "p_selectolax": len(sx_nodes),
                "p_words_selectolax": sum(words_of(n) for n in sx_nodes),
                "p_lxml": lx_count,
                "p_words_lxml": lx_words,
            }
        )
        if len(out) == 6:
            break
    return out, best


def noise_blocks(container: LexborNode) -> list[dict[str, Any]]:
    found: Counter[str] = Counter()
    words: Counter[str] = Counter()
    for node in container.css("*"):
        if node.tag in ("p", "a", "span", "b", "strong", "em", "i", "br", "sup", "sub"):
            continue
        attrs = node.attributes
        label = " ".join(str(attrs.get(name) or "") for name in ("class", "id", *SELECTOR_ATTRS))
        if label.strip() and NOISE_RE.search(label):
            selector = css_for(node)
            found[selector] += 1
            words[selector] += words_of(node)
    return [{"selector": sel, "count": n, "words": words[sel]} for sel, n in found.most_common(12)]


def boilerplate_markers(container: LexborNode) -> list[dict[str, Any]]:
    paragraphs = [p for p in container.css("p") if words_of(p) > 0]
    hits: list[dict[str, Any]] = []
    for index, p in enumerate(paragraphs):
        head = p.text(deep=True, separator=" ").strip().lower()[:40]
        for prefix, label in PR_MARKERS:
            if head.startswith(prefix):
                hit = {"label": label, "paragraph_index": index, "of": len(paragraphs), "words": words_of(p)}
                hits.append(hit)
                break
    return hits


def apply_profile(texts: list[str], profile: dict[str, Any]) -> tuple[list[str], int | None]:
    """Profile semantics extractors mirror: drop short paragraphs, cut at the first stop past the lede."""
    min_words = int(profile.get("min_words", 1))
    lede_n = int(profile.get("lede_paragraphs", 2))
    stops = tuple(marker.lower() for marker in profile.get("stop_markers", []))
    kept = [text for text in texts if len(text.split()) >= min_words]
    for index, text in enumerate(kept):
        if index >= lede_n and stops and text.lower().startswith(stops):
            return kept[:index], index
    return kept, None


def _head(text: str) -> str:
    return "".join(text.split())[:32]


def profile_check(html: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Apply a sources.toml profile with both parsers; word counts must agree for selectors to be portable."""
    sx = LexborHTMLParser(html)
    for selector in profile.get("drop", []):
        for node in sx.css(selector):
            node.decompose()
    sx_texts = [" ".join(n.text(deep=True, separator=" ").split()) for n in sx.css(profile["body"])]
    doc = lxml_doc(html)
    for selector in profile.get("drop", []):
        for el in doc.cssselect(selector):
            el.drop_tree()
    lx_texts = [" ".join(el.text_content().split()) for el in doc.cssselect(profile["body"])]
    sx_kept, cut = apply_profile(sx_texts, profile)
    lx_kept, _ = apply_profile(lx_texts, profile)
    lede_n = int(profile.get("lede_paragraphs", 2))
    leaked = sorted(
        {
            label
            for text in sx_kept[lede_n:]
            for prefix, label in PR_MARKERS
            if text.lower().startswith(prefix)
        }
    )
    result: dict[str, Any] = {
        "paragraphs_selectolax": len(sx_kept),
        "paragraphs_lxml": len(lx_kept),
        "words_selectolax": sum(len(t.split()) for t in sx_kept),
        "words_lxml": sum(len(t.split()) for t in lx_kept),
        # Non-whitespace characters compare the parsers without their different word-joining rules.
        "chars_selectolax": sum(len("".join(t.split())) for t in sx_kept),
        "chars_lxml": sum(len("".join(t.split())) for t in lx_kept),
        "lede_words": sum(len(t.split()) for t in sx_kept[:lede_n]),
        # Whitespace-free heads: selectolax splits words at inline tags, lxml glues text across <br>.
        "same_order": [_head(t) for t in sx_kept] == [_head(t) for t in lx_kept],
        "stopped_at": cut,
        "boilerplate_kept": leaked,
    }
    if title := profile.get("title"):
        heading = LexborHTMLParser(html).css_first(title)
        result["title_found"] = heading is not None and bool(heading.text(strip=True))
    if paywall := profile.get("paywall"):
        region = LexborHTMLParser(html).css_first(paywall)
        result["paywall_region_words"] = words_of(region) if region is not None else None
    return result


def analyze(html: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    tree = LexborHTMLParser(html)
    doc = lxml_doc(html)
    candidates, best = body_candidates(tree, doc)
    h1 = tree.css("h1")
    analysis: dict[str, Any] = {
        "jsonld": jsonld_summary(html),
        "framework_json": framework_json(html, tree),
        "title": {
            "h1_count": len(h1),
            "h1_selector": css_for(h1[0]) if h1 else None,
            "og_title": tree.css_first('meta[property="og:title"]') is not None,
        },
        "body_candidates": candidates,
        "page_p_words": sum(words_of(p) for p in tree.css("p")),
        "paywall_classes": sorted(
            {
                css_for(node)
                for node in tree.css("[class]")
                if PAYWALL_RE.search(node.attributes.get("class") or "")
            }
        )[:8],
    }
    if best is not None:
        paragraphs = [p for p in best.css("p") if words_of(p) > 0]
        analysis["best"] = {
            "selector": css_for(best),
            "lede_words": [words_of(p) for p in paragraphs[:3]],
            "tables": len(best.css("table")),
            "noise": noise_blocks(best),
            "boilerplate_markers": boilerplate_markers(best),
        }
    if profile:
        analysis["profile_check"] = profile_check(html, profile)
    return analysis


# --- orchestration --------------------------------------------------------------------------------------


def _profile_ok(check: dict[str, Any]) -> bool:
    chars, other = int(check["chars_selectolax"]), int(check["chars_lxml"])
    enough = int(check["words_selectolax"]) >= USABLE_BODY_WORDS
    return enough and abs(chars - other) <= 0.01 * chars and bool(check["same_order"])


def summarize(articles: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [a for a in articles if a["fetch"]["outcome"] == "ok"]
    usable = [
        a
        for a in ok
        if max((c["words"] for c in a["analysis"].get("body_candidates", [])), default=0) >= USABLE_BODY_WORDS
        or a["analysis"]["jsonld"]["article_body_words"] >= USABLE_BODY_WORDS
    ]
    fetch_ms = [a["fetch"]["fetch_ms"] for a in ok]
    checks = [a["analysis"]["profile_check"] for a in ok if "profile_check" in a["analysis"]]
    summary: dict[str, Any] = {
        "attempted": len(articles),
        "ok": len(ok),
        "usable_body": len(usable),
        "outcomes": dict(Counter(a["fetch"]["outcome"] for a in articles)),
        "fetch_ms_median": round(statistics.median(fetch_ms), 1) if fetch_ms else None,
        "bytes_median": int(statistics.median(a["fetch"]["size_bytes"] for a in ok)) if ok else None,
        "jsonld_news_article": sum(a["analysis"]["jsonld"]["news_article"] for a in ok),
        "jsonld_article_body": sum(a["analysis"]["jsonld"]["article_body_words"] > 0 for a in ok),
        "consent_handled": sum(a["fetch"]["consent_ms"] is not None for a in articles),
        "verdict": "usable" if len(usable) >= 2 else "not_usable",
    }
    if checks:
        summary["profile_ok"] = sum(_profile_ok(check) for check in checks)
    return summary


async def probe_articles(
    client: httpx.AsyncClient, polite: Politeness, key: str, cfg: dict[str, Any], urls: list[str]
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for n, url in enumerate(urls, start=1):
        got = await fetch(client, polite, url)
        save_raw(key, str(n), got.body)
        analysis: dict[str, Any] = {"jsonld": {"news_article": False, "article_body_words": 0}}
        if got.outcome == "ok" and got.body:
            analysis = analyze(got.text, cfg.get("profile"))
        articles.append({"n": n, "fetch": got.meta(), "analysis": analysis})
        line = f"{got.status} {got.outcome} {got.fetch_ms:.0f}ms {got.size_bytes}B"
        print(f"  {key}#{n} {line} {url}", flush=True)
    return articles


def harvest(key: str, cfg: dict[str, Any], limit: int) -> list[str]:
    """Links to this source in other sources' captures, for when its own feeds and sections refuse us."""
    links: list[str] = []
    for path in sorted(RAW_DIR.glob("*/*.html")):
        if path.parent.name not in (key, "_explore"):
            links += section_links(path.read_text(errors="replace"), "https://invalid.example/")
    return pick(links, cfg, limit)


async def probe_source(
    client: httpx.AsyncClient, polite: Politeness, key: str, cfg: dict[str, Any], limit: int
) -> dict[str, Any]:
    urls, attempts = await discover(client, polite, key, cfg, limit)
    print(f"{key}: discovered {len(urls)} via {[a['url'] for a in attempts]}", flush=True)
    articles = await probe_articles(client, polite, key, cfg, urls)
    return {
        "name": cfg["name"],
        "kind": cfg["kind"],
        "discovery": attempts,
        "discovered_by": "own" if urls else "failed",
        "articles": articles,
        "summary": summarize(articles),
    }


def reanalyze(previous: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    """Recompute page analysis from saved HTML so profiles can be tuned without re-hitting publishers."""
    for key, entry in previous["sources"].items():
        cfg = sources.get(key, {})
        for article in entry["articles"]:
            path = RAW_DIR / key / f"{article['n']}.html"
            if article["fetch"]["outcome"] == "ok" and path.exists():
                article["analysis"] = analyze(path.read_text(errors="replace"), cfg.get("profile"))
        entry["summary"] = summarize(entry["articles"])
    previous["reanalyzed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return previous


def tcf_purpose_bits(tc_string: str) -> str:
    """Purpose-consent bitfield of a TCF v2 core string (24 bits at offset 152)."""
    core = tc_string.split(".")[0]
    raw = base64.urlsafe_b64decode(core + "=" * (-len(core) % 4))
    return "".join(f"{byte:08b}" for byte in raw)[152:176]


async def consent_check() -> dict[str, Any]:
    """Run reject-all from Yahoo's consent entry point even when articles skip the wall; record names only."""
    async with httpx.AsyncClient(http2=True, timeout=TIMEOUT) as client:
        try:
            outcome = await consent.reject_all(client, YAHOO_CONSENT_ENTRY)
        except (VtError, httpx.HTTPError) as exc:
            return {"entry": YAHOO_CONSENT_ENTRY, "ok": False, "error": str(exc)}
        tc_string = next((c.value for c in client.cookies.jar if c.name == "EuConsent"), None)
        return {
            "entry": YAHOO_CONSENT_ENTRY,
            "ok": True,
            "done_url": outcome.done_url,
            "elapsed_ms": round(outcome.elapsed_ms, 1),
            "cookies": sorted({c.name for c in client.cookies.jar}),
            "tcf_purposes_consented": tcf_purpose_bits(tc_string).count("1") if tc_string else None,
        }


async def run_live(sources: dict[str, Any], keys: list[str], limit: int, vantage: str) -> dict[str, Any]:
    polite = Politeness()
    yahoo_consent = await consent_check()
    async with httpx.AsyncClient(http2=True, timeout=TIMEOUT) as client:
        results = await asyncio.gather(*(probe_source(client, polite, k, sources[k], limit) for k in keys))
        # Second pass runs after every source has saved its pages, so blocked hubs can borrow their links.
        for key, entry in zip(keys, results, strict=True):
            if entry["articles"] or not (urls := harvest(key, sources[key], limit)):
                continue
            entry["discovered_by"] = "harvested from other sources' captured pages"
            entry["articles"] = await probe_articles(client, polite, key, sources[key], urls)
            entry["summary"] = summarize(entry["articles"])
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "vantage": vantage,
        "client": {
            "library": f"httpx {httpx.__version__}",
            "http2": True,
            "headers": sorted(browser_headers("https://example.com/")),
            "user_agent": browser_headers("https://example.com/")["User-Agent"],
            "accept_encoding": accept_encoding(),
            "redirects": "manual, max 5; consent hop answered with reject-all once",
        },
        "politeness": {"min_gap_s": MIN_GAP_S, "sec_min_gap_s": SEC_MIN_GAP_S, "per": "registrable domain"},
        "yahoo_consent_check": yahoo_consent,
        "sources": dict(zip(keys, results, strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated source keys")
    parser.add_argument("--articles", type=int, default=3)
    parser.add_argument("--vantage", default="local", help="where the probe ran, e.g. 'EU (BG) residential'")
    parser.add_argument("--reanalyze", action="store_true", help="re-score saved HTML, no network")
    args = parser.parse_args()

    sources: dict[str, Any] = tomllib.loads(SOURCES_TOML.read_text())["sources"]
    if args.reanalyze:
        report = reanalyze(json.loads(OUT_JSON.read_text()), sources)
    else:
        keys = args.only.split(",") if args.only else list(sources)
        report = asyncio.run(run_live(sources, keys, args.articles, args.vantage))
        if args.only and OUT_JSON.exists():
            merged = json.loads(OUT_JSON.read_text())
            merged["sources"].update(report["sources"])
            report = merged | {k: v for k, v in report.items() if k != "sources"}
    # The decision lives in sources.toml; echoing it here keeps evidence and verdict side by side.
    report["registry_status"] = {key: cfg["status"] for key, cfg in sources.items()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for key, entry in report["sources"].items():
        s = entry["summary"]
        counts = f"ok={s['ok']}/{s['attempted']} body={s['usable_body']}"
        print(f"{key:14} {s['verdict']:11} {counts} {s['outcomes']}")


if __name__ == "__main__":
    main()
