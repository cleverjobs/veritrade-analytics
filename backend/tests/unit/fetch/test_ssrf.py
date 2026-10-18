from __future__ import annotations

import ipaddress
from typing import Any

import httpcore
import pytest

from veritrade.errors import BlockedDestination, InvalidUrl, UpstreamHttp
from veritrade.fetch.ssrf import GuardedBackend, blocked_reason, check_target, system_resolver

from .conftest import PUBLIC_IP, FakeResolver

BLOCKED_ADDRESSES = [
    ("127.0.0.1", "loopback"),
    ("127.8.9.10", "loopback"),
    ("10.1.2.3", "private"),
    ("172.16.0.1", "private"),
    ("192.168.1.1", "private"),
    ("169.254.169.254", "link-local"),
    ("100.64.0.1", "non-global"),
    ("100.127.255.254", "non-global"),
    ("0.0.0.0", "unspecified"),
    ("0.1.2.3", "private"),
    ("224.0.0.1", "multicast"),
    ("239.255.255.250", "multicast"),
    ("240.0.0.1", "reserved"),
    ("255.255.255.255", "reserved"),
    ("192.0.2.1", "private"),
    ("198.18.0.1", "private"),
    ("::1", "loopback"),
    ("::", "unspecified"),
    ("fe80::1", "link-local"),
    ("fc00::1", "private"),
    ("fd12:3456::1", "private"),
    ("fec0::1", "site-local"),
    ("ff02::1", "multicast"),
    ("ff0e::1", "multicast"),
    ("::ffff:127.0.0.1", "loopback"),
    ("::ffff:10.0.0.1", "private"),
    ("::ffff:169.254.169.254", "link-local"),
    ("::ffff:100.64.0.1", "non-global"),
    ("::127.0.0.1", "reserved"),
    ("64:ff9b::a00:1", "private"),
    ("64:ff9b::7f00:1", "loopback"),
    ("2002:7f00:1::1", "private"),
    ("2001::1", "private"),
    ("2001:db8::1", "private"),
    ("100::1", "reserved"),
]
PUBLIC_ADDRESSES = ["8.8.8.8", PUBLIC_IP, "2606:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"]


@pytest.mark.parametrize(("address", "reason"), BLOCKED_ADDRESSES)
def test_blocked_address_classes(address: str, reason: str) -> None:
    assert blocked_reason(ipaddress.ip_address(address)) == reason


@pytest.mark.parametrize("address", PUBLIC_ADDRESSES)
def test_public_addresses_pass(address: str) -> None:
    assert blocked_reason(ipaddress.ip_address(address)) is None


@pytest.mark.parametrize(("address", "_reason"), BLOCKED_ADDRESSES)
async def test_dns_answer_in_any_blocked_class_is_refused(address: str, _reason: str) -> None:
    resolver = FakeResolver({"news.test": [address]})
    with pytest.raises(BlockedDestination, match=r"news\.test resolves to a"):
        await check_target("https://news.test/a", resolver=resolver)


async def test_public_dns_answer_passes() -> None:
    resolver = FakeResolver()
    await check_target("https://news.test/a?b=1", resolver=resolver)
    assert resolver.calls == ["news.test"]


async def test_one_private_record_among_public_ones_refuses_the_host() -> None:
    resolver = FakeResolver({"mixed.test": [PUBLIC_IP, "2606:4700::1111", "10.0.0.7"]})
    with pytest.raises(BlockedDestination):
        await check_target("https://mixed.test/", resolver=resolver)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:7f00:1]/",
        "http://[fd00::1]:443/",
        "https://0.0.0.0/",
    ],
)
async def test_ip_literals_are_checked_without_dns(url: str) -> None:
    resolver = FakeResolver()
    with pytest.raises(BlockedDestination):
        await check_target(url, resolver=resolver)
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url",
    ["http://2130706433/", "http://0x7f.1/", "http://0177.0.0.1/", "http://127.1/", "http://999.1.1.1/"],
)
async def test_non_canonical_numeric_hosts_are_invalid(url: str) -> None:
    with pytest.raises(InvalidUrl, match="canonical"):
        await check_target(url, resolver=FakeResolver())


@pytest.mark.parametrize(
    "url",
    [
        "ftp://news.test/a",
        "file:///etc/passwd",
        "gopher://news.test/",
        "javascript:alert(1)",
        "news.test/a",
        "https://user:pw@news.test/",
        "https://user@news.test/",
        "https:///path-only",
        "https://news.test:99999/",
        "https://news.test:abc/",
        "https://[fe80::1%25en0]/",
    ],
)
async def test_malformed_or_unsupported_urls_are_invalid(url: str) -> None:
    with pytest.raises(InvalidUrl):
        await check_target(url, resolver=FakeResolver())


@pytest.mark.parametrize("url", ["https://news.test:8443/", "http://news.test:22/", "http://news.test:6379/"])
async def test_ports_other_than_80_and_443_are_blocked(url: str) -> None:
    with pytest.raises(BlockedDestination, match="port"):
        await check_target(url, resolver=FakeResolver())


@pytest.mark.parametrize("url", ["https://news.test:443/", "http://news.test:80/", "HTTPS://NEWS.TEST/"])
async def test_default_ports_and_case_pass(url: str) -> None:
    await check_target(url, resolver=FakeResolver())


async def test_unresolvable_host_is_invalid() -> None:
    with pytest.raises(InvalidUrl, match="does not resolve"):
        await check_target("https://gone.test/", resolver=FakeResolver({"gone.test": ["NXDOMAIN"]}))
    with pytest.raises(InvalidUrl, match="does not resolve"):
        await check_target("https://empty.test/", resolver=FakeResolver({"empty.test": []}))


async def test_temporary_dns_failure_is_retryable_upstream_error() -> None:
    with pytest.raises(UpstreamHttp) as caught:
        await check_target("https://flaky.test/", resolver=FakeResolver({"flaky.test": ["SERVFAIL"]}))
    assert caught.value.retryable


async def test_allow_hosts_bypass_ip_and_port_checks_for_exact_names_only() -> None:
    resolver = FakeResolver({"fixtures.local": ["172.18.0.5"], "sub.fixtures.local": ["172.18.0.6"]})
    allow = frozenset({"fixtures.local"})
    await check_target("http://fixtures.local:8080/yahoo/a.html", resolver=resolver, allow_hosts=allow)
    await check_target("http://FIXTURES.LOCAL./a", resolver=resolver, allow_hosts=allow)
    assert resolver.calls == []
    with pytest.raises(BlockedDestination):
        await check_target("http://sub.fixtures.local/", resolver=resolver, allow_hosts=allow)
    with pytest.raises(InvalidUrl):
        await check_target("ftp://fixtures.local/", resolver=resolver, allow_hosts=allow)


async def test_trailing_dot_host_is_still_resolved_and_checked() -> None:
    with pytest.raises(BlockedDestination):
        await check_target("http://internal.test./", resolver=FakeResolver({"internal.test.": ["10.0.0.1"]}))


async def test_system_resolver_reads_localhost_without_network() -> None:
    answers = await system_resolver("localhost", 80)
    assert answers
    assert {"127.0.0.1", "::1"} & set(answers)
    with pytest.raises(BlockedDestination, match="loopback"):
        await check_target("http://localhost/")


# --- connect-time guard ------------------------------------------------------------------------------


class RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.failing = failing
        self.connects: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connects.append((host, port))
        if host in self.failing:
            raise httpcore.ConnectError(f"unreachable {host}")
        return httpcore.AsyncMockStream([])

    async def sleep(self, seconds: float) -> None:
        return None


async def test_guarded_backend_connects_to_the_vetted_address_not_the_name() -> None:
    inner = RecordingBackend()
    backend = GuardedBackend(resolver=FakeResolver(), inner=inner)
    await backend.connect_tcp("news.test", 443)
    assert inner.connects == [(PUBLIC_IP, 443)]


async def test_guarded_backend_defeats_dns_rebinding_between_check_and_connect() -> None:
    answers = iter([[PUBLIC_IP], ["127.0.0.1"]])

    async def rebinding(host: str, port: int) -> list[str]:
        return next(answers)

    inner = RecordingBackend()
    await check_target("https://rebind.test/", resolver=rebinding)
    with pytest.raises(BlockedDestination, match="loopback"):
        await GuardedBackend(resolver=rebinding, inner=inner).connect_tcp("rebind.test", 443)
    assert inner.connects == []


async def test_guarded_backend_falls_back_to_the_next_address() -> None:
    inner = RecordingBackend(failing=frozenset({"2606:4700::1111"}))
    backend = GuardedBackend(
        resolver=FakeResolver({"dual.test": ["2606:4700::1111", PUBLIC_IP]}), inner=inner
    )
    await backend.connect_tcp("dual.test", 443)
    assert inner.connects == [("2606:4700::1111", 443), (PUBLIC_IP, 443)]


async def test_guarded_backend_raises_the_last_connect_error() -> None:
    inner = RecordingBackend(failing=frozenset({PUBLIC_IP}))
    with pytest.raises(httpcore.ConnectError):
        await GuardedBackend(resolver=FakeResolver(), inner=inner).connect_tcp("news.test", 443)


async def test_guarded_backend_passes_allow_hosts_by_name() -> None:
    inner = RecordingBackend()
    resolver = FakeResolver({"fixtures.local": ["172.18.0.5"]})
    backend = GuardedBackend(resolver=resolver, allow_hosts=frozenset({"fixtures.local"}), inner=inner)
    await backend.connect_tcp("fixtures.local", 8080)
    assert inner.connects == [("fixtures.local", 8080)]
    assert resolver.calls == []


async def test_guarded_backend_refuses_unix_sockets() -> None:
    with pytest.raises(BlockedDestination):
        await GuardedBackend(inner=RecordingBackend()).connect_unix_socket("/var/run/docker.sock")
