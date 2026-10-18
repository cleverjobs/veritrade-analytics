from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from veritrade_evals import discovery
from veritrade_evals.discovery import (
    Archive,
    ArchiveClient,
    Discoverer,
    Link,
    Politeness,
    Request,
    Response,
    edgar_fts_hits,
    edgar_fts_url,
    expand,
    feed_items,
    feed_links,
    listing_links,
    months_between,
    parse_any,
    parse_date,
    query_matches,
    sitemap_entries,
    url_date,
)

RSS = b"""<?xml version="1.0"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
<item><title>Acme lifts outlook</title><link>https://example.com/a?utm_source=rss</link>
<pubDate>Fri, 16 Oct 2026 12:00:00 GMT</pubDate></item>
<item><title>Acme hebt Prognose</title><link>https://example.com/de</link><dc:language>de</dc:language></item>
<item><title>No link</title></item>
</channel></rss>"""

ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>8-K - ACME CORP</title><link href="https://www.sec.gov/Archives/edgar/data/1/0001-index.htm"/>
<updated>2026-10-15T16:01:02-04:00</updated></entry></feed>"""

URLSET = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
 xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"
 xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
<url><loc>https://example.com/2026/10/16/acme-results.html</loc>
 <news:news><news:publication_date>2026-10-16T09:00:00Z</news:publication_date>
 <news:title>Acme results</news:title></news:news>
 <image:image><image:loc>https://img.example.com/a.jpg</image:loc></image:image></url>
<url><loc>https://example.com/2026/09/01/older.html</loc><lastmod>2026-09-02</lastmod></url>
</urlset>"""

INDEX = b"""<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://example.com/all1.xml</loc></sitemap>
<sitemap><loc>https://example.com/all12.xml</loc></sitemap>
<sitemap><loc>https://example.com/all3.xml</loc></sitemap>
</sitemapindex>"""

LISTING = """<html><body>
<a href="/news-releases/acme-reports-record-revenue-301.html"><img src="x.png"></a>
<a class="card" href="/news-releases/acme-reports-record-revenue-301.html?utm=1">
  <div><h3><small>Aug 17, 2026, 08:55 ET</small> Acme reports record revenue</h3>
  <p>Snippet text.</p></div></a>
<a href="https://example.com/story"><time datetime="2026-10-01T10:00:00Z">Oct 1</time> Story title</a>
<a href="javascript:void(0)">menu</a>
<a href="#top">top</a>
</body></html>"""

FTS = {
    "hits": {
        "hits": [
            {
                "_id": "0001104659-26-104735:agx-ex99d1.htm",
                "_source": {
                    "adsh": "0001104659-26-104735",
                    "ciks": ["0000100591"],
                    "file_type": "EX-99.1",
                    "file_date": "2026-09-02",
                    "items": ["2.02", "9.01"],
                    "display_names": ["ARGAN INC  (AGX)  (CIK 0000100591)"],
                    "file_description": "PRESS RELEASE",
                },
            },
            {
                "_id": "0000820318-26-000001:d8k.htm",
                "_source": {
                    "ciks": ["0000820318"],
                    "file_type": "8-K",
                    "file_date": "2026-08-31",
                    "display_names": ["OTHER CO (CIK 0000820318)"],
                },
            },
            {"_id": "broken", "_source": {}},
        ]
    }
}

INDEX_HTML = """<table class="tableFile"><tr><th>Seq</th></tr>
<tr><td>1</td><td>8-K</td>
<td><a href="/Archives/edgar/data/820318/000082031826000001/d8k.htm">d8k.htm</a></td>
<td>8-K</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td>
<td><a href="/ix?doc=/Archives/edgar/data/820318/000082031826000001/ex991.htm">ex991.htm</a></td>
<td>EX-99.1</td></tr></table>"""


def test_feed_items_keep_title_and_date_and_skip_foreign_items() -> None:
    items = feed_items(RSS)
    assert [i.url for i in items] == ["https://example.com/a?utm_source=rss"]
    assert items[0].title == "Acme lifts outlook"
    assert items[0].published == "2026-10-16"
    atom = feed_items(ATOM)
    assert atom[0].url.endswith("0001-index.htm")
    assert atom[0].published == "2026-10-15"


def test_feed_links_matches_the_probe_contract() -> None:
    assert feed_links(RSS) == ["https://example.com/a?utm_source=rss"]
    assert feed_links(b"") == []


def test_sitemap_entries_ignore_images_and_read_news_fields() -> None:
    kind, links = sitemap_entries(URLSET)
    assert kind == "urlset"
    assert [link.url for link in links] == [
        "https://example.com/2026/10/16/acme-results.html",
        "https://example.com/2026/09/01/older.html",
    ]
    assert links[0].title == "Acme results"
    assert links[0].published == "2026-10-16"
    assert links[1].published is None
    assert links[1].extra == {"lastmod": "2026-09-02"}
    assert sitemap_entries(INDEX)[0] == "sitemapindex"


def test_listing_links_take_heading_text_and_in_anchor_dates() -> None:
    links = {link.url: link for link in listing_links(LISTING, "https://www.prnewswire.com/list/")}
    release = links["https://www.prnewswire.com/news-releases/acme-reports-record-revenue-301.html"]
    assert release.title is None or "Snippet" not in release.title
    release_q = links["https://www.prnewswire.com/news-releases/acme-reports-record-revenue-301.html?utm=1"]
    assert release_q.title == "Acme reports record revenue"
    assert release_q.published == "2026-08-17"
    assert links["https://example.com/story"].published == "2026-10-01"
    assert not any(url.startswith("javascript") for url in links)


def test_listing_links_merge_duplicates_keeping_the_longest_title() -> None:
    html = '<a href="/x">Short</a><a href="/x"><h2>A much longer headline</h2></a>'
    [link] = listing_links(html, "https://example.com/")
    assert link.title == "A much longer headline"


def test_parse_any_sniffs_each_format() -> None:
    assert parse_any(URLSET, "https://example.com/")[0] == "urlset"
    assert parse_any(INDEX, "https://example.com/")[0] == "sitemapindex"
    assert parse_any(RSS, "https://example.com/")[0] == "feed"
    assert parse_any(LISTING.encode(), "https://example.com/")[0] == "html"


def test_edgar_fts_hits_build_archive_urls() -> None:
    exhibit, main = edgar_fts_hits(FTS)
    assert exhibit.url == "https://www.sec.gov/Archives/edgar/data/100591/000110465926104735/agx-ex99d1.htm"
    assert exhibit.company == "ARGAN INC (AGX)"
    assert exhibit.link().extra["items"] == "2.02,9.01"
    assert main.index_url == (
        "https://www.sec.gov/Archives/edgar/data/820318/000082031826000001/0000820318-26-000001-index.htm"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-16T09:00:00Z", date(2026, 10, 16)),
        ("Fri, 16 Oct 2026 12:00:00 GMT", date(2026, 10, 16)),
        ("Aug 17, 2026, 08:55 ET", date(2026, 8, 17)),
        ("September 3, 2026", date(2026, 9, 3)),
        ("2026-02-30", None),
        ("yesterday", None),
        (None, None),
    ],
)
def test_parse_date(text: str | None, expected: date | None) -> None:
    assert parse_date(text) == expected


def test_url_date_reads_numeric_and_guardian_months() -> None:
    assert url_date("https://www.cnbc.com/2026/10/16/x.html") == date(2026, 10, 16)
    assert url_date("https://www.theguardian.com/business/2026/oct/16/x") == date(2026, 10, 16)
    assert url_date("https://www.bbc.co.uk/news/articles/c1") is None


def test_expand_templates_over_the_window() -> None:
    since, until = date(2026, 8, 30), date(2026, 10, 2)
    assert months_between(since, until) == [(2026, 8), (2026, 9), (2026, 10)]
    monthly = expand(Archive("monthly", "https://x/{yyyy}-{mm}.xml"), since, until, [])
    assert [r.url for r in monthly] == [
        "https://x/2026-08.xml",
        "https://x/2026-09.xml",
        "https://x/2026-10.xml",
    ]
    daily = expand(Archive("daily", "https://x/{yyyy}/{mon}/{dd}/all"), date(2026, 9, 30), until, [])
    assert [r.url for r in daily] == [
        "https://x/2026/oct/02/all",
        "https://x/2026/oct/01/all",
        "https://x/2026/sep/30/all",
    ]
    assert daily[0].day == date(2026, 10, 2)
    paged = expand(Archive("paged", "https://x/?page={page}", pages=2), since, until, [])
    assert [r.url for r in paged] == ["https://x/?page=1", "https://x/?page=2"]
    search = expand(Archive("search", "https://x/?q={query}&p={page}", pages=1), since, until, ["net loss"])
    assert search == [Request("https://x/?q=net+loss&p=1", query="net loss")]


async def test_daily_archives_follow_cursor_pages_and_date_their_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = "https://example.com/sitemap/2026_10_16/"
    cursor = "https://example.com/sitemap/2026_10_16_start1792/"
    pages = {
        day: (
            b'<a href="/2026/10/09/dated-story.html">x</a><a href="/news/undated-story.html">y</a>'
            b'<a href="/sitemap/2026_10_16_start1792/">next</a>'
        ),
        cursor: b'<a href="/news/second-page.html">z</a><a href="/sitemap/2026_10_16/">back</a>',
    }
    client = FakeClient(tmp_path, pages)
    archive = Archive(
        "daily",
        "https://example.com/sitemap/{yyyy}_{mm}_{dd}/",
        pages=3,
        next_pattern=r"^https://example\.com/sitemap/\d{4}_\d\d_\d\d_start\d+/$",
    )
    monkeypatch.setitem(discovery.ARCHIVES, "wire", (archive,))
    sources = {
        "wire": SOURCES["wire"] | {"article_pattern": r"^https://example\.com/.*story\.html$|.*page\.html$"}
    }
    day_date = date(2026, 10, 16)
    run = await Discoverer(client, sources).run("wire", since=day_date, until=day_date)
    assert client.seen == [day, cursor]
    published = {f.link.url.rsplit("/", 1)[-1]: f.link.published for f in run.found}
    assert published == {
        "dated-story.html": None,
        "undated-story.html": "2026-10-16",
        "second-page.html": "2026-10-16",
    }


def test_edgar_query_phrases_and_offsets() -> None:
    url = edgar_fts_url("net loss", date(2026, 8, 19), date(2026, 10, 18), 100)
    assert "q=%22net%20loss%22" in url
    assert "forms=8-K" in url
    assert "from=100" in url
    assert "q=" not in edgar_fts_url(None, date(2026, 8, 19), date(2026, 10, 18), 0)


@pytest.mark.parametrize(
    ("query", "haystack", "hit"),
    [
        ("lowered guidance", "acme lowers full year guidance", True),
        ("record revenue", "acme-posts-record-quarterly-revenue", True),
        ("net loss", "acme widens loss", False),
        ("cut", "dividend cuts", True),
    ],
)
def test_query_matches_loose_stems(query: str, haystack: str, hit: bool) -> None:
    assert query_matches(query, haystack) is hit


# --- Discoverer over a fake client ------------------------------------------------------------------------


class FakeClient(ArchiveClient):
    def __init__(self, tmp_path: Path, pages: dict[str, bytes]) -> None:
        super().__init__(tmp_path / "cache", polite=Politeness(0.0, 0.0))
        self.pages = pages
        self.seen: list[str] = []

    async def get(self, url: str) -> Response:
        self.seen.append(url)
        if url not in self.pages:
            return Response(url, url, 404)
        return Response(url, url, 200, self.pages[url])


SOURCES: dict[str, dict[str, Any]] = {
    "wire": {
        "status": "primary",
        "rank": 1,
        "domains": ["example.com"],
        "article_pattern": r"^https://example\.com/20\d\d/\d\d/\d\d/[a-z0-9-]+\.html$",
    },
    "sec": {
        "status": "primary",
        "rank": 2,
        "domains": ["www.sec.gov"],
        "article_pattern": r"^https://www\.sec\.gov/Archives/edgar/data/\d+/\d+/[^/]+\.html?$",
    },
}


async def test_discoverer_follows_the_newest_child_and_filters_by_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    newest = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/2026/10/01/kept-story.html?x=1</loc></url>
    <url><loc>https://example.com/video/2026/10/01/clip</loc></url>
    <url><loc>https://example.com/2026/10/01/all</loc></url></urlset>"""
    client = FakeClient(
        tmp_path, {"https://example.com/index.xml": INDEX, "https://example.com/all12.xml": newest}
    )
    monkeypatch.setitem(
        discovery.ARCHIVES, "wire", (Archive("sitemap", "https://example.com/index.xml", select="newest"),)
    )
    run = await Discoverer(client, SOURCES).run("wire", since=date(2026, 9, 1), until=date(2026, 10, 18))
    assert client.seen == ["https://example.com/index.xml", "https://example.com/all12.xml"]
    assert [f.link.url for f in run.found] == ["https://example.com/2026/10/01/kept-story.html"]
    assert run.archives[0].matched == 1


async def test_discoverer_edgar_resolves_cover_document_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    since, until = date(2026, 8, 19), date(2026, 10, 18)
    pages = {
        edgar_fts_url(None, since, until, 0): json.dumps(FTS).encode(),
        "https://www.sec.gov/Archives/edgar/data/820318/000082031826000001/0000820318-26-000001-index.htm": (
            INDEX_HTML.encode()
        ),
    }
    client = FakeClient(tmp_path, pages)
    monkeypatch.setitem(discovery.ARCHIVES, "sec", (Archive("edgar-fts", discovery.EDGAR_FTS, pages=2),))
    run = await Discoverer(client, SOURCES).run("sec", since=since, until=until)
    urls = [f.link.url for f in run.found]
    assert urls == [
        "https://www.sec.gov/Archives/edgar/data/100591/000110465926104735/agx-ex99d1.htm",
        "https://www.sec.gov/Archives/edgar/data/820318/000082031826000001/ex991.htm",
    ]
    # Fewer than a full page of hits ends the query after one request.
    assert sum("efts" in url for url in client.seen) == 1


async def test_search_archives_only_run_with_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(tmp_path, {})
    monkeypatch.setitem(
        discovery.ARCHIVES, "wire", (Archive("search", "https://example.com/s?q={query}&p={page}"),)
    )
    run = await Discoverer(client, SOURCES).run("wire", since=date(2026, 9, 1), until=date(2026, 10, 18))
    assert client.seen == []
    assert run.archives == []
    await Discoverer(client, SOURCES).run(
        "wire", since=date(2026, 9, 1), until=date(2026, 10, 18), queries=["loss"]
    )
    assert client.seen == ["https://example.com/s?q=loss&p=1"]


# --- ArchiveClient: robots, gzip, cache --------------------------------------------------------------------


async def test_archive_client_honours_robots_and_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /search/\n")
        if request.url.path == "/sitemap.xml.gz":
            return httpx.Response(200, content=gzip.compress(URLSET))
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArchiveClient(tmp_path / "cache", polite=Politeness(0.0, 0.0), client=http)
    blocked = await client.get("https://example.com/search/?q=x")
    assert blocked.error == "robots_disallowed"
    first = await client.get("https://example.com/sitemap.xml.gz")
    assert first.status == 200
    assert sitemap_entries(first.body)[0] == "urlset"
    second = await client.get("https://example.com/sitemap.xml.gz")
    assert second.cached
    assert calls == ["/robots.txt", "/sitemap.xml.gz"]
    await client.aclose()


async def test_archive_client_treats_robots_server_errors_as_disallow(tmp_path: Path) -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    client = ArchiveClient(tmp_path / "cache", polite=Politeness(0.0, 0.0), client=http)
    assert not await client.allowed("https://example.com/page")
    await client.aclose()


def test_link_defaults_are_independent() -> None:
    assert Link("a").extra == {}
    assert Link("a").extra is not Link("b").extra


async def test_edgar_resolves_release_bearing_items_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    since, until = date(2026, 8, 19), date(2026, 10, 18)

    def cover(adsh: str, cik: str, items: list[str]) -> dict[str, Any]:
        return {"_id": f"{adsh}:d8k.htm", "_source": {"ciks": [cik], "file_type": "8-K", "items": items}}

    payload = {
        "hits": {
            "hits": [
                cover("0000000001-26-000001", "1", ["5.02"]),
                cover("0000000002-26-000001", "2", ["2.02"]),
            ]
        }
    }
    client = FakeClient(tmp_path, {edgar_fts_url(None, since, until, 0): json.dumps(payload).encode()})
    monkeypatch.setitem(discovery.ARCHIVES, "sec", (Archive("edgar-fts", discovery.EDGAR_FTS),))
    monkeypatch.setattr(discovery, "MAX_EDGAR_RESOLVE", 1)
    await Discoverer(client, SOURCES).run("sec", since=since, until=until)
    assert client.seen[-1].endswith("/data/2/000000000226000001/0000000002-26-000001-index.htm")
    assert len(client.seen) == 2


async def test_archive_client_retries_a_timeout_once(tmp_path: Path) -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if request.url.path == "/slow" and attempts.count("/slow") == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, text="ok" if request.url.path == "/slow" else "")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArchiveClient(tmp_path / "cache", polite=Politeness(0.0, 0.0), client=http)
    got = await client.get("https://example.com/slow")
    assert (got.status, got.body) == (200, b"ok")
    assert attempts == ["/robots.txt", "/slow", "/slow"]
    await client.aclose()
