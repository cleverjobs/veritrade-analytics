from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from veritrade.config import Settings
from veritrade.errors import (
    BlockedDestination,
    ConsentRequired,
    InvalidUrl,
    SourceBlocked,
    TooLarge,
    UnsupportedContent,
    UpstreamHttp,
    UpstreamTimeout,
)
from veritrade.fetch.client import HostRateLimiter, HttpFetcher, decode_html
from veritrade.fetch.headers import CHROME_UA

from .conftest import ARTICLE_HTML, FakeClock, FakeResolver

ARTICLE = "https://news.test/2026/10/18/acme-beats.html"
SEC_EXHIBIT = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/ex99-1.htm"


async def test_happy_path_returns_decoded_html_and_timings(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    route = router.get(ARTICLE).respond(200, html=ARTICLE_HTML)
    result = await fetcher.get(ARTICLE)
    assert result.requested_url == result.final_url == ARTICLE
    assert result.status == 200
    assert result.html == ARTICLE_HTML
    assert result.content_type == "text/html; charset=utf-8"
    assert result.redirects == 0
    assert result.consent_ms is None
    assert result.fetch_ms > 0
    sent = route.calls.last.request.headers
    assert sent["User-Agent"] == CHROME_UA
    assert sent["Sec-Fetch-Mode"] == "navigate"


async def test_redirects_are_followed_and_every_hop_is_checked(
    fetcher: HttpFetcher, router: respx.MockRouter, resolver: FakeResolver
) -> None:
    router.get("http://news.test/short").respond(301, headers={"Location": "https://news.test/a"})
    router.get("https://news.test/a").respond(302, headers={"Location": "/2026/10/18/acme-beats.html"})
    router.get(ARTICLE).respond(200, html=ARTICLE_HTML)
    result = await fetcher.get("http://news.test/short")
    assert result.final_url == ARTICLE
    assert result.requested_url == "http://news.test/short"
    assert result.redirects == 2
    assert resolver.calls == ["news.test"] * 3


@pytest.mark.parametrize(
    "location",
    [
        "https://internal.test/admin",
        "http://metadata.test/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://mixed.test/",
        "https://news.test:8443/",
    ],
)
async def test_redirect_into_private_space_is_blocked_before_the_request(
    fetcher: HttpFetcher, router: respx.MockRouter, location: str
) -> None:
    router.get(ARTICLE).respond(302, headers={"Location": location})
    trap = router.route(url__regex=r"^(?!https://news\.test/2026).*").respond(200, html="<html>secret</html>")
    with pytest.raises(BlockedDestination):
        await fetcher.get(ARTICLE)
    assert not trap.called


async def test_redirect_to_another_scheme_is_invalid(fetcher: HttpFetcher, router: respx.MockRouter) -> None:
    router.get(ARTICLE).respond(302, headers={"Location": "file:///etc/passwd"})
    with pytest.raises(InvalidUrl):
        await fetcher.get(ARTICLE)


async def test_redirect_cap(fetcher: HttpFetcher, router: respx.MockRouter, settings: Settings) -> None:
    cap = settings.fetch_max_redirects
    for hop in range(cap + 1):
        router.get(f"https://news.test/r{hop}").respond(302, headers={"Location": f"/r{hop + 1}"})
    router.get(f"https://news.test/r{cap}").respond(200, html=ARTICLE_HTML)
    assert (await fetcher.get("https://news.test/r0")).redirects == cap

    router.get(f"https://news.test/r{cap}").respond(302, headers={"Location": f"/r{cap + 1}"})
    with pytest.raises(UpstreamHttp, match="redirected more than") as caught:
        await fetcher.get("https://news.test/r0")
    assert caught.value.upstream_status == 302
    assert not caught.value.retryable


async def test_initial_url_is_checked(fetcher: HttpFetcher, router: respx.MockRouter) -> None:
    with pytest.raises(BlockedDestination):
        await fetcher.get("http://internal.test/")
    with pytest.raises(InvalidUrl):
        await fetcher.get("ftp://news.test/a")
    with pytest.raises(InvalidUrl, match="malformed"):
        await fetcher.get("https://news.test/a\x00b")
    with pytest.raises(InvalidUrl, match="malformed"):
        await fetcher.get("http://[::1/")
    assert not router.calls


# --- body size and content type ----------------------------------------------------------------------


async def test_declared_length_over_the_cap_is_refused_unread(
    fetcher: HttpFetcher, router: respx.MockRouter, settings: Settings
) -> None:
    router.get(ARTICLE).respond(200, html="<html>" + "x" * settings.fetch_max_bytes + "</html>")
    with pytest.raises(TooLarge):
        await fetcher.get(ARTICLE)


async def test_streamed_body_over_the_cap_is_refused(
    fetcher: HttpFetcher, router: respx.MockRouter, settings: Settings
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(settings.fetch_max_bytes // 1000 + 2):
            yield b"<p>" + b"x" * 993 + b"</p>"

    router.get(ARTICLE).mock(
        side_effect=lambda request: httpx.Response(
            200, headers={"Content-Type": "text/html"}, content=chunks()
        )
    )
    with pytest.raises(TooLarge, match="more than"):
        await fetcher.get(ARTICLE)


async def test_body_at_the_cap_is_accepted(
    fetcher: HttpFetcher, router: respx.MockRouter, settings: Settings
) -> None:
    body = "<html>" + "x" * (settings.fetch_max_bytes - 13) + "</html>"
    assert len(body) == settings.fetch_max_bytes
    router.get(ARTICLE).respond(200, html=body)
    assert len((await fetcher.get(ARTICLE)).html) == settings.fetch_max_bytes


@pytest.mark.parametrize(
    ("url", "content_type"),
    [
        (ARTICLE, "application/pdf"),
        (ARTICLE, "image/png"),
        (ARTICLE, "application/json"),
        (ARTICLE, "application/rss+xml; charset=utf-8"),
        (ARTICLE, "text/plain"),
        ("https://www.sec.gov/Archives/edgar/data/1/0001/report.pdf", "application/pdf"),
        ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent", "text/plain"),
    ],
)
async def test_non_html_content_is_unsupported(
    fetcher: HttpFetcher, router: respx.MockRouter, url: str, content_type: str
) -> None:
    router.get(url).respond(200, content=b"%PDF-1.7 binary", headers={"Content-Type": content_type})
    with pytest.raises(UnsupportedContent):
        await fetcher.get(url)


@pytest.mark.parametrize(
    ("url", "content_type"),
    [
        (ARTICLE, "application/xhtml+xml; charset=utf-8"),
        (ARTICLE, "TEXT/HTML"),
        (SEC_EXHIBIT, "text/html"),
        (SEC_EXHIBIT, "text/plain"),
        (
            "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001.txt",
            "text/plain",
        ),
    ],
)
async def test_html_and_sec_documents_are_accepted(
    fetcher: HttpFetcher, router: respx.MockRouter, url: str, content_type: str
) -> None:
    router.get(url).respond(200, content=ARTICLE_HTML.encode(), headers={"Content-Type": content_type})
    assert (await fetcher.get(url)).html == ARTICLE_HTML


async def test_missing_content_type_is_sniffed(fetcher: HttpFetcher, router: respx.MockRouter) -> None:
    router.get(ARTICLE).respond(200, content=b"\n  " + ARTICLE_HTML.encode())
    result = await fetcher.get(ARTICLE)
    assert result.content_type == "text/html"
    router.get("https://news.test/blob").respond(200, content=b"\x89PNG\r\n\x1a\n")
    with pytest.raises(UnsupportedContent):
        await fetcher.get("https://news.test/blob")


async def test_sec_requests_carry_the_contact_user_agent(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    route = router.get(SEC_EXHIBIT).respond(200, html=ARTICLE_HTML)
    await fetcher.get(SEC_EXHIBIT)
    agent = route.calls.last.request.headers["User-Agent"]
    assert "Chrome" not in agent
    assert "@" in agent


# --- decoding ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ("Café".encode(), "text/html; charset=utf-8", "Café"),
        ("Café".encode("latin-1"), "text/html; charset=ISO-8859-1", "Café"),
        (b"it\x92s", "text/html; charset=iso-8859-1", "it’s"),
        (b'<meta charset="windows-1252"><p>it\x92s', "text/html", "it’s"),
        (
            b'<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS"><p>\x83e',
            "text/html",
            "テ",
        ),
        (b"<p>it\x92s</p>", "text/html", "it’s"),
        ("<p>naïve</p>".encode(), "text/html", "naïve"),
        (b"\xef\xbb\xbf<p>bom</p>", "text/html; charset=windows-1252", "<p>bom</p>"),
        ("<p>x</p>".encode("utf-16"), "text/html", "<p>x</p>"),
        ("Café".encode(), "text/html; charset=no-such-codec", "Café"),
        ("Café".encode(), "text/html; charset=base64", "Café"),
        ("Café".encode(), 'text/html; charset="utf-8"', "Café"),
    ],
)
def test_decode_html(body: bytes, content_type: str, expected: str) -> None:
    assert expected in decode_html(body, content_type)


async def test_header_charset_is_used_end_to_end(fetcher: HttpFetcher, router: respx.MockRouter) -> None:
    router.get(ARTICLE).respond(
        200,
        content="<html><p>Société Générale</p></html>".encode("latin-1"),
        headers={"Content-Type": "text/html; charset=iso-8859-1"},
    )
    assert "Société Générale" in (await fetcher.get(ARTICLE)).html


# --- status, block and timeout mapping -----------------------------------------------------------------


@pytest.mark.parametrize(("status", "retryable"), [(404, False), (410, False), (500, True), (503, True)])
async def test_upstream_status_mapping(
    fetcher: HttpFetcher, router: respx.MockRouter, status: int, retryable: bool
) -> None:
    router.get(ARTICLE).respond(status, html="<html><body>error page</body></html>")
    with pytest.raises(UpstreamHttp) as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.upstream_status == status
    assert caught.value.retryable is retryable
    assert caught.value.hint


async def test_redirect_without_location_is_an_upstream_error(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    router.get(ARTICLE).respond(302)
    with pytest.raises(UpstreamHttp) as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.upstream_status == 302


@pytest.mark.parametrize("status", [401, 403, 429, 451])
async def test_block_statuses_are_source_blocked(
    fetcher: HttpFetcher, router: respx.MockRouter, status: int
) -> None:
    router.get(ARTICLE).respond(
        status, html="<html><title>Denied</title></html>", headers={"Server": "cloudflare"}
    )
    with pytest.raises(SourceBlocked, match="Cloudflare") as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.upstream_status == status


async def test_challenge_page_served_with_200_is_source_blocked(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    router.get(ARTICLE).respond(200, html="<html><head><title>Just a moment...</title></head></html>")
    with pytest.raises(SourceBlocked):
        await fetcher.get(ARTICLE)


async def test_oversized_error_page_still_maps_to_its_status(
    fetcher: HttpFetcher, router: respx.MockRouter, settings: Settings
) -> None:
    router.get(ARTICLE).respond(503, html="<html>" + "x" * (settings.fetch_max_bytes * 2))
    with pytest.raises(UpstreamHttp) as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.upstream_status == 503


@pytest.mark.parametrize("error", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout])
async def test_httpx_timeouts_map_to_upstream_timeout(
    fetcher: HttpFetcher, router: respx.MockRouter, error: type[httpx.TimeoutException]
) -> None:
    router.get(ARTICLE).mock(side_effect=error)
    with pytest.raises(UpstreamTimeout) as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.retryable


async def test_overall_deadline_maps_to_upstream_timeout(
    resolver: FakeResolver, router: respx.MockRouter
) -> None:
    async def stall(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, html=ARTICLE_HTML)

    router.get(ARTICLE).mock(side_effect=stall)
    fetcher = HttpFetcher(Settings(env="test", fetch_deadline_s=0.05), resolver=resolver)
    try:
        with pytest.raises(UpstreamTimeout, match=r"within 0\.05 s"):
            await fetcher.get(ARTICLE)
    finally:
        await fetcher.aclose()


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError])
async def test_transport_errors_map_to_upstream_http(
    fetcher: HttpFetcher, router: respx.MockRouter, error: type[httpx.TransportError]
) -> None:
    router.get(ARTICLE).mock(side_effect=error)
    with pytest.raises(UpstreamHttp, match=error.__name__) as caught:
        await fetcher.get(ARTICLE)
    assert caught.value.upstream_status is None


# --- Yahoo consent -------------------------------------------------------------------------------------

YAHOO_ARTICLE = "https://finance.yahoo.com/news/acme-beats-123.html"
CONSENT_ENTRY = "https://guce.yahoo.com/consent?brandType=eu&done=https%3A%2F%2Ffinance.yahoo.com%2Fnews"
CONSENT_FORM = "https://consent.yahoo.com/v2/collectConsent?sessionId=3_cc-session_abc"
CONSENT_COPY = "https://guce.yahoo.com/copyConsent?sessionId=3_cc-session_abc"
# Synthetic form with the field names and classes the reject-all parser looks for.
CONSENT_PAGE = """<html><body><form method="post" class="consent-form" action="">
<input type="hidden" name="csrfToken" value="tok-1"><input type="hidden" name="sessionId" value="s-1">
<button type="submit" class="btn accept-all" name="agree" value="agree">Accept all</button>
<button type="submit" class="btn reject-all" name="reject" value="reject">Reject all</button>
</form></body></html>"""


def _mock_consent_flow(router: respx.MockRouter) -> respx.Route:
    router.get(CONSENT_ENTRY).respond(302, headers={"Location": CONSENT_FORM})
    router.get(CONSENT_FORM).respond(200, html=CONSENT_PAGE)
    post = router.post(CONSENT_FORM).respond(302, headers={"Location": CONSENT_COPY})
    router.get(CONSENT_COPY).respond(
        302,
        headers={"Location": YAHOO_ARTICLE, "Set-Cookie": "GUC=AQAB; Domain=.yahoo.com; Path=/; Secure"},
    )
    return post


def _article_behind_wall(request: httpx.Request) -> httpx.Response:
    if "GUC=AQAB" in request.headers.get("cookie", ""):
        return httpx.Response(200, html=ARTICLE_HTML)
    return httpx.Response(302, headers={"Location": CONSENT_ENTRY})


async def test_consent_wall_is_answered_with_reject_all_then_retried_once(
    fetcher: HttpFetcher, router: respx.MockRouter, resolver: FakeResolver
) -> None:
    post = _mock_consent_flow(router)
    article = router.get(YAHOO_ARTICLE).mock(side_effect=_article_behind_wall)
    result = await fetcher.get(YAHOO_ARTICLE)

    assert result.html == ARTICLE_HTML
    assert result.final_url == YAHOO_ARTICLE
    assert result.redirects == 0
    assert result.consent_ms is not None
    assert result.consent_ms >= 0
    assert article.call_count == 2
    body = parse_qs(post.calls.last.request.content.decode())
    assert body["reject"] == ["reject"]
    assert "agree" not in body
    assert {"guce.yahoo.com", "consent.yahoo.com"} <= set(resolver.calls)

    # The cookie jar now carries the answer, so the next article skips the wall.
    again = await fetcher.get(YAHOO_ARTICLE)
    assert again.consent_ms is None
    assert article.call_count == 3


async def test_wall_after_reject_all_is_consent_required(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    _mock_consent_flow(router)
    router.get(YAHOO_ARTICLE).respond(302, headers={"Location": CONSENT_ENTRY})
    with pytest.raises(ConsentRequired) as caught:
        await fetcher.get(YAHOO_ARTICLE)
    assert caught.value.hint
    assert "Yahoo" in caught.value.hint


async def test_unanswerable_consent_form_is_consent_required(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    router.get(YAHOO_ARTICLE).respond(302, headers={"Location": CONSENT_FORM})
    router.get(CONSENT_FORM).respond(200, html="<html><form class='consent-form'></form></html>")
    with pytest.raises(ConsentRequired):
        await fetcher.get(YAHOO_ARTICLE)


async def test_consent_page_served_directly_is_consent_required(
    fetcher: HttpFetcher, router: respx.MockRouter
) -> None:
    router.get(ARTICLE).respond(200, html=CONSENT_PAGE)
    with pytest.raises(ConsentRequired):
        await fetcher.get(ARTICLE)


# --- rate limiting -------------------------------------------------------------------------------------


async def test_sec_requests_are_spaced_to_ten_per_second(
    fetcher: HttpFetcher, router: respx.MockRouter, clock: FakeClock
) -> None:
    router.get(SEC_EXHIBIT).respond(200, html=ARTICLE_HTML)
    router.get(ARTICLE).respond(200, html=ARTICLE_HTML)
    for _ in range(3):
        await fetcher.get(SEC_EXHIBIT)
        await fetcher.get(ARTICLE)
    assert clock.sleeps == [0.1, 0.1]


async def test_rate_limiter_reserves_slots_for_concurrent_callers() -> None:
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(round(seconds, 6))

    limiter = HostRateLimiter({"sec.gov": 10.0}, clock=lambda: 50.0, sleep=record)
    urls = ["https://www.sec.gov/a", "https://efts.sec.gov/b", "https://SEC.GOV./c", "https://news.test/d"]
    await asyncio.gather(*(limiter.acquire(url) for url in urls))
    assert sleeps == [0.1, 0.2]


async def test_rate_limiter_does_not_match_lookalike_hosts() -> None:
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    limiter = HostRateLimiter({"sec.gov": 1.0}, clock=lambda: 0.0, sleep=record)
    for url in [
        "https://notsec.gov/",
        "https://sec.gov.example.com/",
        "https://sec.gov/",
        "https://sec.gov/",
    ]:
        await limiter.acquire(url)
    assert sleeps == [1.0]


async def test_aclose_closes_the_client(settings: Settings, resolver: FakeResolver) -> None:
    fetcher = HttpFetcher(settings, resolver=resolver)
    await fetcher.aclose()
    with pytest.raises(RuntimeError):
        await fetcher.get(ARTICLE)
