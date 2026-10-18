from __future__ import annotations

import pytest

from veritrade.fetch import headers
from veritrade.fetch.headers import CHROME_MAJOR, browser_headers, is_sec_host


def test_publisher_hosts_get_the_chrome_navigation_set() -> None:
    sent = browser_headers("https://www.cnbc.com/2026/01/01/x.html")
    assert f"Chrome/{CHROME_MAJOR}.0.0.0" in sent["User-Agent"]
    assert sent["Accept-Language"].startswith("en-US")
    assert sent["Sec-Fetch-Mode"] == "navigate"
    assert sent["Sec-Fetch-Dest"] == "document"
    assert sent["Upgrade-Insecure-Requests"] == "1"
    assert f'"Google Chrome";v="{CHROME_MAJOR}"' in sent["Sec-CH-UA"]


@pytest.mark.usefixtures("fresh_settings")
def test_sec_hosts_get_the_declared_contact_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Example Org ops@example.com")
    sent = browser_headers("https://www.sec.gov/Archives/edgar/data/1/2/x.htm")
    assert sent["User-Agent"] == "Example Org ops@example.com"
    assert not any(name.startswith("Sec-") for name in sent)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("www.sec.gov", True),
        ("SEC.GOV.", True),
        ("efts.sec.gov", True),
        ("notsec.gov", False),
        ("sec.gov.example.com", False),
    ],
)
def test_is_sec_host(host: str, expected: bool) -> None:
    assert is_sec_host(host) is expected


@pytest.mark.usefixtures("fresh_settings")
def test_accept_encoding_only_advertises_installed_decoders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(headers, "find_spec", lambda name: None)
    assert headers.accept_encoding() == "gzip, deflate"
    headers.accept_encoding.cache_clear()
    monkeypatch.setattr(headers, "find_spec", lambda name: object() if name == "brotli" else None)
    assert headers.accept_encoding() == "gzip, deflate, br"
