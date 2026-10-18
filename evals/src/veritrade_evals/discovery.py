"""Candidate article discovery: feeds, sitemaps, archive and listing pages, EDGAR full-text search.

Requests are sequential per domain with a minimum gap, checked against robots.txt and cached for a few hours,
so agents can re-run discovery with new queries without hitting publishers again.
"""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import html as html_lib
import json
import re
import time
import tomllib
import urllib.robotparser
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, quote_plus, urlencode, urljoin, urlsplit, urlunsplit

import httpx
import lxml.html
from lxml import etree

from veritrade.extract.dom import normalize as normalize_text
from veritrade.fetch.headers import browser_headers, is_sec_host

MIN_GAP_S = 1.0
SEC_MIN_GAP_S = 0.2  # 5 req/s, half of EDGAR's fair-access ceiling
TIMEOUT = httpx.Timeout(45.0, connect=10.0)
TIMEOUT_RETRIES = 1
TIMEOUT_ERRORS = frozenset({"ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"})
MAX_ARCHIVE_BYTES = 32_000_000
CACHE_TTL_S = 6 * 3600
EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_PAGE = 100
MAX_EDGAR_RESOLVE = 120
MAX_EDGAR_FEED = 40
MONTH_ABBR = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")

ArchiveKind = Literal["feed", "sitemap", "monthly", "daily", "listing", "paged", "search", "edgar-fts"]
IndexSelect = Literal["all", "newest"]
PAGED_KINDS = frozenset({"paged", "search", "edgar-fts", "daily"})
# 8-K items whose filings usually attach a press release as EX-99.1.
RELEASE_ITEMS = frozenset({"2.02", "7.01", "8.01", "1.01", "2.05", "2.06"})


# --- polite fetching (shared with scripts/probe_sources.py) ---------------------------------------------


class Politeness:
    """Sequential requests per registrable domain with a minimum gap; SEC gets its own faster ceiling."""

    def __init__(self, min_gap_s: float = MIN_GAP_S, sec_min_gap_s: float = SEC_MIN_GAP_S) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}
        self._min_gap_s = min_gap_s
        self._sec_min_gap_s = sec_min_gap_s

    @staticmethod
    def domain(url: str) -> str:
        host = (urlsplit(url).hostname or "").lower()
        return ".".join(host.split(".")[-3:] if host.endswith(".co.uk") else host.split(".")[-2:])

    async def wait(self, url: str) -> asyncio.Lock:
        key = self.domain(url)
        lock = self._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        gap = self._sec_min_gap_s if is_sec_host(urlsplit(url).hostname or "") else self._min_gap_s
        delay = self._last.get(key, 0.0) + gap - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        return lock

    def done(self, url: str, lock: asyncio.Lock) -> None:
        self._last[self.domain(url)] = time.monotonic()
        lock.release()


# --- URL helpers ------------------------------------------------------------------------------------------


def normalize(url: str) -> str:
    """https, no query or fragment: article URLs here are path-addressed and feeds append tracking params."""
    parts = urlsplit(url.strip())
    return urlunsplit(("https", parts.netloc.lower(), parts.path, "", ""))


def feed_links(body: bytes) -> list[str]:
    """Item links of an RSS or Atom document, namespace-agnostic and tolerant of broken XML."""
    return [link.url for link in feed_items(body)]


def section_links(html: str, base: str) -> list[str]:
    hrefs = re.findall(r"""(?:href=|<loc>)["']?([^"'<> ]+)""", html)
    return [urljoin(base, href) for href in hrefs]


def pick(urls: Iterable[str], cfg: Mapping[str, Any], limit: int) -> list[str]:
    pattern = re.compile(cfg["article_pattern"])
    exclude = re.compile(cfg["exclude_pattern"]) if cfg.get("exclude_pattern") else None
    chosen: list[str] = []
    for raw in urls:
        url = normalize(raw)
        if url in chosen or not pattern.match(url) or (exclude and exclude.search(url)):
            continue
        chosen.append(url)
        if len(chosen) == limit:
            break
    return chosen


def edgar_exhibit(index_html: str, index_url: str) -> tuple[str, str] | None:
    """EX-99.1 document link and description from an EDGAR filing index (inline-XBRL viewer unwrapped)."""
    doc = lxml.html.fromstring(index_html)
    for row in doc.cssselect("table.tableFile tr"):
        cells = [cell.text_content().strip() for cell in row.cssselect("td")]
        if len(cells) >= 4 and cells[3].upper().startswith("EX-99.1"):
            links = row.cssselect("a[href]")
            if links:
                href = str(links[0].get("href")).replace("/ix?doc=", "")
                return urljoin(index_url, href), cells[1]
    return None


# --- dates ------------------------------------------------------------------------------------------------

_ISO_DATE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
_URL_DATE = re.compile(r"/(20\d\d)/(\d\d|[a-z]{3})/(\d\d)(?=/|$)")
_TEXT_DATE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? (\d{1,2}), (20\d\d)"
    r"(?:,? \d{1,2}:\d\d(?: ?[AP]M)?(?: [A-Z]{2,4})?)?"
)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str | None) -> date | None:
    """ISO, RFC 822 (RSS) or 'Aug 17, 2026' text; anything else is undated rather than guessed."""
    if not text:
        return None
    text = text.strip()
    if m := _ISO_DATE.match(text):
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    try:
        return email.utils.parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        pass
    if m := _TEXT_DATE.search(text):
        return _safe_date(int(m[3]), MONTH_ABBR.index(m[1][:3].lower()) + 1, int(m[2]))
    return None


def url_date(url: str) -> date | None:
    """Date in the path, /2026/10/18/ or the Guardian's /2026/oct/18/."""
    m = _URL_DATE.search(urlsplit(url).path)
    if not m:
        return None
    month = int(m[2]) if m[2].isdigit() else MONTH_ABBR.index(m[2]) + 1 if m[2] in MONTH_ABBR else 0
    return _safe_date(int(m[1]), month, int(m[3])) if month else None


def months_between(since: date, until: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = since.year, since.month
    while (year, month) <= (until.year, until.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def days_between(since: date, until: date) -> list[date]:
    return [since + timedelta(days=n) for n in range((until - since).days + 1)]


# --- parsers (pure) ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Link:
    url: str
    title: str | None = None
    published: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)


def _xml_root(body: bytes) -> Any:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=True)
    return etree.fromstring(body, parser) if body.strip() else None


def _local(el: Any) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _child_text(el: Any, *names: str) -> str | None:
    for child in el.iter():
        if _local(child) in names and child.text and child.text.strip():
            return str(child.text).strip()
    return None


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


def feed_items(body: bytes) -> list[Link]:
    """Items of an RSS or Atom feed with title and date; non-English items are skipped."""
    root = _xml_root(body)
    if root is None:
        return []
    links: list[Link] = []
    for item in root.iter("{*}item", "{*}entry"):
        # Wires mix translated copies into one feed; the models are evaluated on English only.
        language = next((str(el.text or "") for el in item.iter("{*}language")), "en")
        if not language.lower().startswith("en"):
            continue
        href = None
        for link in item.iter("{*}link"):
            href = link.get("href") or (link.text or "").strip()
            if href:
                break
        if not href:
            continue
        title = _child_text(item, "title")
        when = parse_date(_child_text(item, "pubDate", "published", "updated", "date"))
        links.append(Link(href, normalize_text(html_lib.unescape(title)) if title else None, _iso(when)))
    return links


def sitemap_entries(body: bytes) -> tuple[str, list[Link]]:
    """('sitemapindex' | 'urlset', entries); image and video locs are ignored, news titles and dates kept."""
    root = _xml_root(body)
    if root is None:
        return "", []
    kind = _local(root)
    links: list[Link] = []
    for el in root:
        if _local(el) not in ("url", "sitemap"):
            continue
        loc = next((str(c.text).strip() for c in el if _local(c) == "loc" and c.text), None)
        if not loc:
            continue
        news = next((c for c in el if _local(c) == "news"), None)
        title = _child_text(news, "title") if news is not None else None
        published = parse_date(_child_text(news, "publication_date")) if news is not None else None
        lastmod = next((str(c.text).strip() for c in el if _local(c) == "lastmod" and c.text), None)
        extra = {"lastmod": lastmod} if lastmod else {}
        links.append(Link(loc, normalize_text(title) if title else None, _iso(published), extra))
    return kind, links


_ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"""\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_HEADING = re.compile(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]\s*>", re.IGNORECASE | re.DOTALL)
_DATETIME_ATTR = re.compile(r"""\bdatetime\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_NOISE = re.compile(r"<(script|style|svg)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _fragment_text(fragment: str) -> str:
    return normalize_text(html_lib.unescape(_TAG.sub(" ", _NOISE.sub(" ", fragment))))


def listing_links(html: str, base: str) -> list[Link]:
    """Anchors of an HTML listing with their heading text and a date found inside the anchor, one per URL."""
    found: dict[str, Link] = {}
    for m in _ANCHOR.finditer(html):
        href = _HREF.search(m.group(1))
        if href is None:
            continue
        raw = next(g for g in href.groups() if g is not None)
        url = urljoin(base, html_lib.unescape(raw).strip())
        if not url.startswith(("https://", "http://")):
            continue
        inner = m.group(2)
        heading = _HEADING.search(inner)
        text = _fragment_text(heading.group(1) if heading else inner)
        when = parse_date(stamp.group(1)) if (stamp := _DATETIME_ATTR.search(inner)) else None
        if (in_text := _TEXT_DATE.search(text)) is not None:
            when = when or parse_date(in_text.group(0))
            text = normalize_text(text.replace(in_text.group(0), " "))
        previous = found.get(url)
        title = text or None
        if previous is not None:
            title = max((t for t in (previous.title, title) if t), key=len, default=None)
            when = when or parse_date(previous.published)
        found[url] = Link(url, title, _iso(when))
    return list(found.values())


def parse_any(body: bytes, base: str) -> tuple[str, list[Link]]:
    """Sniff a response: ('sitemapindex' | 'urlset' | 'feed' | 'html', links)."""
    head = body[:2048].lstrip().lower()
    if b"<urlset" in head or b"<sitemapindex" in head:
        return sitemap_entries(body)
    if b"<rss" in head or b"<feed" in head or b"<rdf:rdf" in head:
        return "feed", feed_items(body)
    return "html", listing_links(body.decode("utf-8", "replace"), base)


@dataclass(frozen=True, slots=True)
class EdgarHit:
    adsh: str
    filename: str
    cik: str
    file_type: str
    file_date: str | None
    items: tuple[str, ...]
    company: str
    description: str

    @property
    def folder(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{self.adsh.replace('-', '')}"

    @property
    def url(self) -> str:
        return f"{self.folder}/{self.filename}"

    @property
    def index_url(self) -> str:
        return f"{self.folder}/{self.adsh}-index.htm"

    def link(self, url: str | None = None) -> Link:
        title = " | ".join(part for part in (self.company, self.description) if part)
        extra = {"form": "8-K", "file_type": self.file_type, "items": ",".join(self.items)}
        return Link(url or self.url, title or None, self.file_date, extra)


_CIK_SUFFIX = re.compile(r"\s*\(CIK \d+\)\s*$")


def edgar_fts_hits(payload: Mapping[str, Any]) -> list[EdgarHit]:
    """Rows of an efts.sec.gov search-index response; `_id` is '<accession>:<file name>'."""
    hits: list[EdgarHit] = []
    rows = payload.get("hits", {}).get("hits", [])
    for row in rows if isinstance(rows, list) else []:
        source = row.get("_source", {})
        adsh, _, filename = str(row.get("_id", "")).partition(":")
        ciks = source.get("ciks") or []
        if not adsh or not filename or not ciks:
            continue
        names = source.get("display_names") or [""]
        hits.append(
            EdgarHit(
                adsh=str(source.get("adsh") or adsh),
                filename=filename,
                cik=str(ciks[0]),
                file_type=str(source.get("file_type") or ""),
                file_date=source.get("file_date"),
                items=tuple(str(i) for i in source.get("items") or ()),
                company=normalize_text(_CIK_SUFFIX.sub("", str(names[0]))),
                description=normalize_text(str(source.get("file_description") or "")),
            )
        )
    return hits


# --- source registry and archives ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Archive:
    """`pages`: listing pages, search pages per query, or day pages reached through `next_pattern` links."""

    kind: ArchiveKind
    url: str
    pages: int = 1
    select: IndexSelect = "all"
    next_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class Request:
    url: str
    query: str | None = None
    day: date | None = None


# Archive pages link to sibling day pages and other nav that the article pattern would accept.
ARCHIVE_EXCLUDE = re.compile(r"/(?:all|altdate)$")

# Verified live on 2026-10-18; feeds and sections from sources.toml run first.
ARCHIVES: dict[str, tuple[Archive, ...]] = {
    "yahoo": (
        Archive("listing", "https://finance.yahoo.com/topic/earnings/"),
        Archive("listing", "https://finance.yahoo.com/topic/latest-news/"),
        Archive("listing", "https://finance.yahoo.com/topic/economic-news/"),
        # 50 stories per page; later pages of a day hang off a `_start<epoch ms>` cursor link.
        Archive(
            "daily",
            "https://finance.yahoo.com/sitemap/{yyyy}_{mm}_{dd}/",
            pages=2,
            next_pattern=r"^https://finance\.yahoo\.com/sitemap/\d{4}_\d\d_\d\d_start\d+/$",
        ),
    ),
    "cnbc": (
        Archive("sitemap", "https://www.cnbc.com/sitemap_news.xml"),
        Archive("sitemap", "https://www.cnbc.com/sitemapAll.xml", select="newest"),
    ),
    "fool": (
        Archive("sitemap", "https://www.fool.com/news-sitemap.xml"),
        Archive("monthly", "https://www.fool.com/sitemap/{yyyy}/{mm}"),
    ),
    "prnewswire": (
        Archive("sitemap", "https://www.prnewswire.com/sitemap-news.xml"),
        Archive(
            "paged",
            "https://www.prnewswire.com/news-releases/financial-services-latest-news/earnings-list/"
            "?page={page}&pagesize=100",
            pages=10,
        ),
        Archive(
            "paged",
            "https://www.prnewswire.com/news-releases/news-releases-list/?page={page}&pagesize=100",
            pages=5,
        ),
        Archive(
            "search",
            "https://www.prnewswire.com/search/news/?keyword={query}&page={page}&pagesize=100",
            pages=3,
        ),
    ),
    "globenewswire": (
        Archive("sitemap", "https://www.globenewswire.com/NewsRoom/GoogleSitemap"),
        Archive("monthly", "https://sitemaps.globenewswire.com/news/en/{yyyy}-{mm}.xml"),
    ),
    "sec": (Archive("edgar-fts", EDGAR_FTS, pages=5),),
    "cnn": (Archive("monthly", "https://edition.cnn.com/sitemap/article/business/{yyyy}/{mm}.xml"),),
    "fortune": (
        Archive("sitemap", "https://fortune.com/feed/googlenews/articles.xml"),
        Archive("monthly", "https://fortune.com/articles-{yyyy}-{mm}.xml"),
        Archive("paged", "https://fortune.com/section/finance/page/{page}/", pages=10),
    ),
    "guardian": (Archive("daily", "https://www.theguardian.com/business/{yyyy}/{mon}/{dd}/all"),),
    "bbc": (Archive("paged", "https://www.bbc.co.uk/news/business?page={page}", pages=40),),
}


def load_sources() -> dict[str, dict[str, Any]]:
    text = resources.files("veritrade").joinpath("data/sources.toml").read_text(encoding="utf-8")
    sources: dict[str, dict[str, Any]] = tomllib.loads(text)["sources"]
    return sources


def primary_sources(sources: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    registry = sources if sources is not None else load_sources()
    keys = [k for k, cfg in registry.items() if cfg.get("status") == "primary"]
    return sorted(keys, key=lambda k: int(registry[k].get("rank", 99)))


def source_archives(key: str, cfg: Mapping[str, Any], listing_urls: Sequence[str] = ()) -> list[Archive]:
    archives = [Archive("feed", url) for url in cfg.get("feeds", [])]
    archives += [Archive("listing", url) for url in cfg.get("sections", [])]
    archives += ARCHIVES.get(key, ())
    archives += [Archive("listing", url) for url in listing_urls]
    return archives


def on_source_domain(url: str, cfg: Mapping[str, Any]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in cfg.get("domains", []))


def expand(archive: Archive, since: date, until: date, queries: Sequence[str]) -> list[Request]:
    """Concrete requests for one archive over the window, newest first for daily archives."""
    template = archive.url
    if archive.kind == "monthly":
        return [Request(template.format(yyyy=y, mm=f"{m:02d}")) for y, m in months_between(since, until)]
    if archive.kind == "daily":
        return [
            Request(
                template.format(
                    yyyy=d.year, mm=f"{d.month:02d}", mon=MONTH_ABBR[d.month - 1], dd=f"{d.day:02d}"
                ),
                day=d,
            )
            for d in reversed(days_between(since, until))
        ]
    if archive.kind == "paged":
        return [Request(template.format(page=p)) for p in range(1, archive.pages + 1)]
    if archive.kind == "search":
        return [
            Request(template.format(query=quote_plus(q), page=p), query=q)
            for q in queries
            for p in range(1, archive.pages + 1)
        ]
    return [Request(template)]


def edgar_fts_url(query: str | None, since: date, until: date, offset: int) -> str:
    params = {"forms": "8-K", "dateRange": "custom", "startdt": since.isoformat(), "enddt": until.isoformat()}
    if query:
        # Multi-word queries are phrases ("net loss"), which is what directional keywords need.
        params["q"] = f'"{query}"' if " " in query and not query.startswith('"') else query
    if offset:
        params["from"] = str(offset)
    # EFTS reads "+" literally inside a quoted phrase, so spaces must be %20.
    return f"{EDGAR_FTS}?{urlencode(params, quote_via=quote)}"


_STEM_TOKEN = re.compile(r"[a-z0-9]+")


def query_matches(query: str, haystack: str) -> bool:
    """Every query word, loosely stemmed, starts some word of the title or URL slug."""
    words = _STEM_TOKEN.findall(haystack.lower())
    for token in _STEM_TOKEN.findall(query.lower()):
        stem = token if len(token) <= 4 else token[: max(4, len(token) - 2)]
        if not any(word.startswith(stem) for word in words):
            return False
    return True


def slug_text(url: str) -> str:
    return " ".join(_STEM_TOKEN.findall(urlsplit(url).path.lower()))


# --- network ----------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Response:
    url: str
    final_url: str
    status: int | None
    body: bytes = b""
    error: str | None = None
    cached: bool = False


class ArchiveClient:
    """Polite GET for archive pages: robots.txt, per-domain gap, size cap, gzip, and a short disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        ttl_s: float = CACHE_TTL_S,
        refresh: bool = False,
        polite: Politeness | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._ttl_s = ttl_s
        self._refresh = refresh
        self._polite = polite or Politeness()
        self._client = client or httpx.AsyncClient(
            http2=True, timeout=TIMEOUT, follow_redirects=True, max_redirects=5, trust_env=False
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.requests = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha1(url.encode()).hexdigest()
        return self._cache_dir / f"{key}.bin", self._cache_dir / f"{key}.json"

    async def _raw_get(self, url: str) -> Response:
        # Wire search and sitemap pages are slow at times; one polite retry after a timeout.
        for attempt in range(1 + TIMEOUT_RETRIES):
            got = await self._once(url)
            if got.error not in TIMEOUT_ERRORS or attempt == TIMEOUT_RETRIES:
                return got
        return got

    async def _once(self, url: str) -> Response:
        lock = await self._polite.wait(url)
        self.requests += 1
        try:
            async with self._client.stream("GET", url, headers=browser_headers(url)) as response:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_ARCHIVE_BYTES:
                        return Response(url, str(response.url), response.status_code, error="too_large")
                return Response(url, str(response.url), response.status_code, _gunzip(bytes(body)))
        except httpx.HTTPError as exc:
            return Response(url, url, None, error=type(exc).__name__)
        finally:
            self._polite.done(url, lock)

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        parser = self._robots.get(host)
        if parser is None:
            got = await self._raw_get(f"{host}/robots.txt")
            parser = urllib.robotparser.RobotFileParser()
            # A missing robots.txt allows everything; a server error means stay away for now.
            if got.status == 200:
                parser.parse(got.body.decode("utf-8", "replace").splitlines())
            elif got.status is not None and got.status >= 500:
                parser.parse(["User-agent: *", "Disallow: /"])
            else:
                parser.parse([])
            self._robots[host] = parser
        return parser.can_fetch(browser_headers(url)["User-Agent"], url)

    async def get(self, url: str) -> Response:
        body_path, meta_path = self._cache_paths(url)
        if not self._refresh and meta_path.exists() and body_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if time.time() - float(meta["fetched_at"]) < self._ttl_s:
                return Response(url, meta["final_url"], meta["status"], body_path.read_bytes(), cached=True)
        if not await self.allowed(url):
            return Response(url, url, None, error="robots_disallowed")
        got = await self._raw_get(url)
        if got.status == 200 and got.error is None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(got.body)
            meta = {"url": url, "final_url": got.final_url, "status": got.status, "fetched_at": time.time()}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return got


def _gunzip(body: bytes) -> bytes:
    if body[:2] != b"\x1f\x8b":
        return body
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return inflater.decompress(body, MAX_ARCHIVE_BYTES)


# --- discovery run ----------------------------------------------------------------------------------------


@dataclass(slots=True)
class ArchiveStats:
    kind: str
    url: str
    requests: int = 0
    ok: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    links: int = 0
    matched: int = 0

    def failed(self, code: str) -> None:
        self.errors[code] = self.errors.get(code, 0) + 1


@dataclass(frozen=True, slots=True)
class Found:
    link: Link
    via: str
    via_url: str
    query: str | None


@dataclass(slots=True)
class DiscoverRun:
    source: str
    found: list[Found] = field(default_factory=list)
    archives: list[ArchiveStats] = field(default_factory=list)


class Discoverer:
    """Collects article links for one source from every archive it has, filtered by its article pattern."""

    def __init__(self, client: ArchiveClient, sources: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._client = client
        self._sources = sources if sources is not None else load_sources()

    def pattern(self, key: str) -> re.Pattern[str]:
        return re.compile(self._sources[key]["article_pattern"])

    def keep(self, key: str, url: str) -> bool:
        cfg = self._sources[key]
        exclude = cfg.get("exclude_pattern")
        if not self.pattern(key).match(url) or ARCHIVE_EXCLUDE.search(url):
            return False
        return not (exclude and re.search(exclude, url))

    async def run(
        self,
        key: str,
        *,
        since: date,
        until: date,
        queries: Sequence[str] = (),
        listing_urls: Sequence[str] = (),
        pages: int | None = None,
    ) -> DiscoverRun:
        cfg = self._sources[key]
        run = DiscoverRun(key)
        for archive in source_archives(key, cfg, listing_urls):
            if archive.kind == "search" and not queries:
                continue
            if pages is not None and archive.kind in PAGED_KINDS:
                archive = replace(archive, pages=pages)
            stats = ArchiveStats(archive.kind, archive.url)
            run.archives.append(stats)
            if archive.kind == "edgar-fts":
                await self._edgar_fts(run, stats, archive, since, until, queries)
            elif archive.kind == "feed" and cfg.get("discovery") == "edgar-8k-ex99":
                await self._edgar_feed(run, stats, archive.url)
            else:
                for request in expand(archive, since, until, queries):
                    await self._collect(run, stats, archive, request)
        return run

    async def _fetch(self, stats: ArchiveStats, url: str) -> Response | None:
        stats.requests += 1
        got = await self._client.get(url)
        if got.error or got.status != 200:
            stats.failed(got.error or f"http_{got.status}")
            return None
        stats.ok += 1
        return got

    async def _collect(
        self, run: DiscoverRun, stats: ArchiveStats, archive: Archive, request: Request
    ) -> None:
        url, seen, visited = request.url, 0, {request.url}
        while True:
            got = await self._fetch(stats, url)
            if got is None:
                return
            kind, links = parse_any(got.body, got.final_url)
            if kind == "sitemapindex":
                children = [link.url for link in links]
                if archive.select == "newest":
                    children = sorted(children, key=_trailing_number)[-1:]
                for child in children:
                    await self._collect(run, stats, Archive("sitemap", child), replace(request, url=child))
                return
            if request.day is not None:
                # A day archive dates its own stories; links that carry a date in the URL keep that date.
                day = request.day.isoformat()
                links = [
                    link if link.published or url_date(link.url) else replace(link, published=day)
                    for link in links
                ]
            self._add(run, stats, links, archive.kind, url, request.query)
            seen += 1
            following = _next_page(links, archive.next_pattern, visited) if archive.next_pattern else None
            if following is None or seen >= archive.pages:
                return
            url = following
            visited.add(url)

    def _add(
        self,
        run: DiscoverRun,
        stats: ArchiveStats,
        links: Sequence[Link],
        kind: str,
        url: str,
        query: str | None,
        *,
        count: bool = True,
    ) -> None:
        if count:
            stats.links += len(links)
        for link in links:
            canonical = normalize(link.url)
            if self.keep(run.source, canonical):
                stats.matched += 1
                run.found.append(
                    Found(Link(canonical, link.title, link.published, link.extra), kind, url, query)
                )

    async def _edgar_fts(
        self,
        run: DiscoverRun,
        stats: ArchiveStats,
        archive: Archive,
        since: date,
        until: date,
        queries: Sequence[str],
    ) -> None:
        exhibits: dict[str, Link] = {}
        main_docs: dict[str, tuple[EdgarHit, str | None]] = {}
        searches: list[str | None] = [*queries] or [None]
        for query in searches:
            for page in range(archive.pages):
                url = edgar_fts_url(query, since, until, page * EDGAR_PAGE)
                got = await self._fetch(stats, url)
                if got is None:
                    break
                try:
                    hits = edgar_fts_hits(json.loads(got.body))
                except ValueError:
                    stats.failed("bad_json")
                    break
                stats.links += len(hits)
                for hit in hits:
                    if hit.file_type.upper().startswith("EX-99.1"):
                        if hit.adsh not in exhibits:
                            exhibits[hit.adsh] = hit.link()
                            self._add(run, stats, [exhibits[hit.adsh]], "edgar-fts", url, query, count=False)
                    elif hit.file_type.upper().startswith("8-K"):
                        main_docs.setdefault(hit.adsh, (hit, query))
                if len(hits) < EDGAR_PAGE:
                    break
        # Without a query, and when a query matches the cover text, the hit is the 8-K itself; its EX-99.1
        # comes from the filing index, earnings and other release-bearing items first.
        pending = [found for adsh, found in main_docs.items() if adsh not in exhibits]
        pending.sort(key=lambda found: RELEASE_ITEMS.isdisjoint(found[0].items))
        for hit, hit_query in pending[:MAX_EDGAR_RESOLVE]:
            got = await self._fetch(stats, hit.index_url)
            if got is not None and (
                found := edgar_exhibit(got.body.decode("utf-8", "replace"), got.final_url)
            ):
                link = hit.link(found[0])
                self._add(run, stats, [link], "edgar-fts", hit.index_url, hit_query, count=False)

    async def _edgar_feed(self, run: DiscoverRun, stats: ArchiveStats, feed_url: str) -> None:
        got = await self._fetch(stats, feed_url)
        if got is None:
            return
        entries = feed_items(got.body)
        stats.links += len(entries)
        for entry in entries[:MAX_EDGAR_FEED]:
            index = await self._fetch(stats, normalize(entry.url))
            if index is not None and (
                found := edgar_exhibit(index.body.decode("utf-8", "replace"), index.final_url)
            ):
                link = Link(found[0], entry.title, entry.published, {"form": "8-K", "file_type": "EX-99.1"})
                self._add(run, stats, [link], "feed", feed_url, None, count=False)


def _next_page(links: Sequence[Link], pattern: str, visited: set[str]) -> str | None:
    rx = re.compile(pattern)
    return next((link.url for link in links if rx.match(link.url) and link.url not in visited), None)


def _trailing_number(url: str) -> int:
    m = re.search(r"(\d+)\D*$", url)
    return int(m.group(1)) if m else -1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
