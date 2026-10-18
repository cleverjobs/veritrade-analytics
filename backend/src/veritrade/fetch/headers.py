"""Request headers of a current desktop Chrome; plain identification, no fingerprint evasion."""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec
from urllib.parse import urlsplit

from veritrade.config import get_settings

CHROME_MAJOR = 155
CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{CHROME_MAJOR}.0.0.0 Safari/537.36"
)
# Brand order and GREASE entry follow Chrome's seeded algorithm for this major version.
CHROME_SEC_CH_UA = f'"Google Chrome";v="{CHROME_MAJOR}", "Chromium";v="{CHROME_MAJOR}", "Not(A:Brand";v="24"'
ACCEPT_HTML = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
    "application/signed-exchange;v=b3;q=0.7"
)
ACCEPT_LANGUAGE = "en-US,en;q=0.9"
SEC_HOST_SUFFIX = "sec.gov"


@cache
def accept_encoding() -> str:
    """Advertise only codings httpx can decode here, or a br/zstd body would arrive undecodable."""
    codings = ["gzip", "deflate"]
    if find_spec("brotli") or find_spec("brotlicffi"):
        codings.append("br")
    if find_spec("zstandard"):
        codings.append("zstd")
    return ", ".join(codings)


def is_sec_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host == SEC_HOST_SUFFIX or host.endswith("." + SEC_HOST_SUFFIX)


def browser_headers(url: str) -> dict[str, str]:
    """Top-level navigation headers; SEC hosts get the declared contact UA its fair-access policy asks for."""
    if is_sec_host(urlsplit(url).hostname or ""):
        # A contact UA with Chrome client hints would contradict itself, so EDGAR gets the plain set.
        return {
            "User-Agent": get_settings().sec_user_agent,
            "Accept": ACCEPT_HTML,
            "Accept-Language": ACCEPT_LANGUAGE,
            "Accept-Encoding": accept_encoding(),
        }
    return {
        "User-Agent": CHROME_UA,
        "Accept": ACCEPT_HTML,
        "Accept-Language": ACCEPT_LANGUAGE,
        "Accept-Encoding": accept_encoding(),
        "Sec-CH-UA": CHROME_SEC_CH_UA,
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"macOS"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Priority": "u=0, i",
    }
