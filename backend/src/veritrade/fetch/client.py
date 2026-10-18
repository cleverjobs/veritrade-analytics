"""The serving fetcher: one HTTP/2 client, manual SSRF-checked redirects, capped bodies, typed failures."""

from __future__ import annotations

import asyncio
import codecs
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import NoReturn
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from veritrade.config import Settings
from veritrade.errors import (
    ConsentRequired,
    InvalidUrl,
    TooLarge,
    UnsupportedContent,
    UpstreamHttp,
    UpstreamTimeout,
)
from veritrade.fetch import detect
from veritrade.fetch.consent import REDIRECT_STATUSES, is_consent_url, reject_all
from veritrade.fetch.headers import browser_headers, is_sec_host
from veritrade.fetch.ssrf import GuardedBackend, Resolver, check_target, system_resolver
from veritrade.fetch.types import FetchResult

LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=60.0)
# EDGAR's fair-access policy allows 10 requests per second per client; every sec.gov host shares it.
RATE_LIMITS: Mapping[str, float] = {"sec.gov": 10.0}
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
SEC_DOCUMENT_SUFFIXES = (".htm", ".html", ".txt")
# Error pages are only read for detect.classify, which scans the first 64 KB.
ERROR_BODY_BYTES = 65_536
SNIFF_BYTES = 4096

_META_CHARSET = re.compile(rb"""<meta[^>]{0,512}?charset\s*=\s*["']?\s*([a-z0-9_.:-]+)""", re.IGNORECASE)
# WHATWG decodes these labels as windows-1252; the 0x80-0x9F smart quotes in old filings depend on it.
_WINDOWS_1252_ALIASES = frozenset({"iso8859-1", "ascii"})


class HostRateLimiter:
    """Spaces request starts evenly per rate-limited host group; other hosts pass straight through."""

    def __init__(
        self,
        rates: Mapping[str, float] = RATE_LIMITS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._intervals = {suffix: 1.0 / rate for suffix, rate in rates.items()}
        self._next: dict[str, float] = {}
        self._clock = clock
        self._sleep = sleep

    def _group(self, host: str) -> str | None:
        host = host.lower().rstrip(".")
        for suffix in self._intervals:
            if host == suffix or host.endswith("." + suffix):
                return suffix
        return None

    async def acquire(self, url: str) -> None:
        group = self._group(urlsplit(url).hostname or "")
        if group is None:
            return
        # Slots are reserved before sleeping, so concurrent callers queue without a lock.
        now = self._clock()
        slot = max(now, self._next.get(group, now))
        self._next[group] = slot + self._intervals[group]
        if slot > now:
            await self._sleep(slot - now)


class _GuardedTransport(httpx.AsyncHTTPTransport):
    """httpx takes no network backend argument, so the pool is rebuilt around the SSRF-guarded one."""

    def __init__(self, backend: httpcore.AsyncNetworkBackend) -> None:
        super().__init__(http2=True, limits=LIMITS)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=LIMITS.max_connections,
            max_keepalive_connections=LIMITS.max_keepalive_connections,
            keepalive_expiry=LIMITS.keepalive_expiry,
            http1=True,
            http2=True,
            network_backend=backend,
        )


class HttpFetcher:
    """Fetcher protocol over one shared client whose cookie jar keeps Yahoo's reject-all answer."""

    def __init__(
        self,
        settings: Settings,
        *,
        resolver: Resolver = system_resolver,
        limiter: HostRateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._allow_hosts = settings.ssrf_allow_hosts
        self._limiter = limiter or HostRateLimiter()
        self._client = httpx.AsyncClient(
            transport=_GuardedTransport(GuardedBackend(resolver=resolver, allow_hosts=self._allow_hosts)),
            follow_redirects=False,
            timeout=httpx.Timeout(settings.fetch_deadline_s, connect=settings.fetch_connect_timeout_s),
            # User-supplied URLs must never pick up .netrc credentials.
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str) -> FetchResult:
        started = time.perf_counter()
        host = _host(url)
        deadline = self._settings.fetch_deadline_s
        try:
            async with asyncio.timeout(deadline):
                return await self._fetch(url, started)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise UpstreamTimeout(
                f"{host} did not deliver the page within {deadline:g} s",
                hint="The publisher is slow or unreachable; retry, or try the story from another source.",
            ) from exc
        except httpx.InvalidURL as exc:
            raise InvalidUrl(f"malformed URL: {exc}") from exc
        except httpx.HTTPError as exc:
            raise UpstreamHttp(
                f"request to {host} failed: {type(exc).__name__}",
                hint="The publisher could not be reached; retry shortly.",
            ) from exc

    async def _check(self, url: str) -> None:
        await check_target(url, resolver=self._resolver, allow_hosts=self._allow_hosts)

    async def _fetch(self, url: str, started: float) -> FetchResult:
        current, redirects = url, 0
        consent_ms: float | None = None
        while True:
            await self._check(current)
            await self._limiter.acquire(current)
            async with self._client.stream("GET", current, headers=browser_headers(current)) as response:
                location = response.headers.get("location")
                if response.status_code not in REDIRECT_STATUSES or not location:
                    html, content_type = await self._read(response, current)
                    fetch_ms = (time.perf_counter() - started) * 1000 - (consent_ms or 0.0)
                    return FetchResult(
                        requested_url=url,
                        final_url=current,
                        status=response.status_code,
                        html=html,
                        content_type=content_type,
                        fetch_ms=fetch_ms,
                        consent_ms=consent_ms,
                        redirects=redirects,
                    )
                status, headers = response.status_code, response.headers
            target = urljoin(current, location)
            if is_consent_url(target):
                if consent_ms is not None:
                    _consent_wall(status, headers, target)
                # Reject-all once, then retry the hop that hit the wall with the consent cookie set.
                outcome = await reject_all(self._client, target, guard=self._check)
                consent_ms = outcome.elapsed_ms
                continue
            redirects += 1
            if redirects > (cap := self._settings.fetch_max_redirects):
                raise UpstreamHttp(
                    f"{_host(url)} redirected more than {cap} times",
                    hint="The URL redirects in a loop; open it in a browser and use the final address.",
                    retryable=False,
                    upstream_status=status,
                )
            current = target

    async def _read(self, response: httpx.Response, url: str) -> tuple[str, str]:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        host = _host(url)
        if not 200 <= status < 300:
            prefix = await _read_body(response, ERROR_BODY_BYTES, host=host, truncate=True)
            detect.classify(status, response.headers, decode_html(prefix, content_type), url)
            raise UpstreamHttp(
                f"{host} answered HTTP {status}",
                hint=_status_hint(status),
                retryable=status >= 500 or status == 408,
                upstream_status=status,
            )

        media = content_type.split(";", 1)[0].strip().lower()
        if media and not _accepted(media, url):
            raise _unsupported(host, media)
        limit = self._settings.fetch_max_bytes
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise _too_large(host, limit)
        body = await _read_body(response, limit, host=host, truncate=False)
        if not media:
            if not _looks_like_html(body):
                raise _unsupported(host, "an undeclared non-HTML type")
            content_type = "text/html"
        html = decode_html(body, content_type)
        detect.classify(status, response.headers, html, url)
        return html, content_type


def _host(url: str) -> str:
    """Host for messages; malformed input (e.g. an unclosed IPv6 bracket) is left to check_target."""
    try:
        return urlsplit(url).hostname or url
    except ValueError:
        return url


async def _read_body(response: httpx.Response, limit: int, *, host: str, truncate: bool) -> bytes:
    """Decoded bytes, so a small compressed body cannot inflate past the cap."""
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > limit:
            if truncate:
                return bytes(body[:limit])
            raise _too_large(host, limit)
    return bytes(body)


def decode_html(body: bytes, content_type: str) -> str:
    """BOM, then header charset, then <meta> charset, then UTF-8 if valid, else windows-1252."""
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", "replace")
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return body.decode("utf-16", "replace")
    for label in (_header_charset(content_type), _meta_charset(body[:SNIFF_BYTES])):
        if (encoding := _codec(label)) is None:
            continue
        try:
            return body.decode(encoding, "replace")
        except LookupError:
            # codecs also registers bytes-to-bytes transforms (base64, zlib) that str decoding refuses.
            continue
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", "replace")


def _header_charset(content_type: str) -> str | None:
    for param in content_type.split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip("'\"")
    return None


def _meta_charset(head: bytes) -> str | None:
    match = _META_CHARSET.search(head)
    return match.group(1).decode("ascii") if match else None


def _codec(label: str | None) -> str | None:
    if not label:
        return None
    try:
        name = codecs.lookup(label).name
    except LookupError:
        return None
    return "cp1252" if name in _WINDOWS_1252_ALIASES else name


def _accepted(media: str, url: str) -> bool:
    if media in HTML_TYPES:
        return True
    parts = urlsplit(url)
    # EDGAR serves .txt exhibits and some filer-agent .htm files as text/plain.
    return (
        media == "text/plain"
        and is_sec_host(parts.hostname or "")
        and parts.path.lower().endswith(SEC_DOCUMENT_SUFFIXES)
    )


def _looks_like_html(body: bytes) -> bool:
    head = body[:SNIFF_BYTES].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html")) or b"<html" in head


def _unsupported(host: str, media: str) -> UnsupportedContent:
    return UnsupportedContent(
        f"{host} returned {media}, not an HTML article",
        hint="Only HTML article pages are supported; link the article page rather than a PDF, image or feed.",
    )


def _too_large(host: str, limit: int) -> TooLarge:
    return TooLarge(
        f"{host} sent more than {limit / 1_000_000:g} MB",
        hint="Article pages are far smaller; the URL probably points at a file download.",
    )


def _status_hint(status: int) -> str:
    if status in (404, 410):
        return "The page does not exist; check the URL or whether the article was taken down."
    if status >= 500:
        return "The publisher had a server error; retry shortly."
    return "The publisher refused the request; try the story from another source."


def _consent_wall(status: int, headers: Mapping[str, str], url: str) -> NoReturn:
    """A second consent redirect after reject-all means the cookie did not stick."""
    detect.classify(status, headers, "", url)
    raise ConsentRequired(f"{urlsplit(url).hostname} demands cookie consent")
