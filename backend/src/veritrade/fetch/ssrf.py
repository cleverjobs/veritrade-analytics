"""SSRF guard: only public http(s) destinations, vetted per redirect hop and again at connect time."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpcore

from veritrade.errors import BlockedDestination, InvalidUrl, UpstreamHttp

Resolver = Callable[[str, int], Awaitable[Sequence[str]]]
IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
# Dotted, decimal, octal or hex numerals that resolvers read as IPv4 (2130706433, 0x7f.1, 0177.0.0.1).
_NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+)){0,3}\.?")
_HINT = "Only public http(s) article URLs on ports 80 and 443 can be fetched."


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def blocked_reason(address: IpAddress) -> str | None:
    """Why an address is not a public unicast destination, or None when it is."""
    if isinstance(address, ipaddress.IPv6Address):
        if (mapped := address.ipv4_mapped) is not None:
            return blocked_reason(mapped)
        # DNS64 networks answer every IPv4-only host inside 64:ff9b::/96, so judge the embedded address.
        if address in NAT64:
            return blocked_reason(ipaddress.IPv4Address(int(address) & 0xFFFF_FFFF))
        if address.is_site_local:
            return "site-local"
    if address.is_unspecified:
        return "unspecified"
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link-local"
    if address.is_multicast:
        return "multicast"
    if address.is_reserved:
        return "reserved"
    if address.is_private:
        return "private"
    if not address.is_global:
        return "non-global"
    return None


def _host_key(host: str) -> str:
    return host.lower().rstrip(".")


def _literal(host: str) -> IpAddress | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        if _NUMERIC_HOST.fullmatch(host):
            raise InvalidUrl(f"'{host}' is not a canonical IP address or host name", hint=_HINT) from None
        return None


async def public_addresses(host: str, port: int, *, resolver: Resolver) -> list[str]:
    """Every address the host resolves to; one non-public answer refuses the whole host."""
    if (literal := _literal(host)) is not None:
        candidates: Sequence[str] = [str(literal)]
    else:
        try:
            candidates = await resolver(host, port)
        except socket.gaierror as exc:
            if exc.errno == socket.EAI_AGAIN:
                raise UpstreamHttp(f"DNS lookup for {host} failed for now", hint="Retry shortly.") from exc
            raise InvalidUrl(f"{host} does not resolve", hint="Check the domain name in the URL.") from exc
    if not candidates:
        raise InvalidUrl(f"{host} does not resolve", hint="Check the domain name in the URL.")
    for candidate in candidates:
        address = ipaddress.ip_address(candidate)
        if (reason := blocked_reason(address)) is not None:
            raise BlockedDestination(f"{host} resolves to a {reason} address", hint=_HINT)
    return list(candidates)


async def check_target(
    url: str, *, resolver: Resolver = system_resolver, allow_hosts: frozenset[str] = frozenset()
) -> None:
    """Raise InvalidUrl or BlockedDestination unless the URL may be fetched; allow_hosts is test-only."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise InvalidUrl(f"malformed URL: {exc}", hint=_HINT) from exc
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidUrl(f"unsupported URL scheme '{scheme or '(none)'}'", hint=_HINT)
    if parts.username is not None or parts.password is not None:
        raise InvalidUrl("URLs with embedded credentials are not fetched", hint=_HINT)
    host = parts.hostname
    if not host or "%" in host:
        raise InvalidUrl("URL has no usable host", hint=_HINT)
    if _host_key(host) in allow_hosts:
        return
    if port is not None and port not in ALLOWED_PORTS:
        raise BlockedDestination(f"port {port} is not allowed", hint=_HINT)
    await public_addresses(host, port or (443 if scheme == "https" else 80), resolver=resolver)


class GuardedBackend(httpcore.AsyncNetworkBackend):
    """Connects only to vetted addresses, pinning the IP so DNS cannot change between check and connect."""

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        allow_hosts: frozenset[str] = frozenset(),
        inner: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._resolver = resolver
        self._allow_hosts = allow_hosts
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's backend signature
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if _host_key(host) in self._allow_hosts:
            return await self._inner.connect_tcp(host, port, timeout, local_address, socket_options)
        addresses = await public_addresses(host, port, resolver=self._resolver)
        # TLS still verifies the certificate against `host`: httpcore passes it as SNI to start_tls.
        for address in addresses[:-1]:
            try:
                return await self._inner.connect_tcp(address, port, timeout, local_address, socket_options)
            except httpcore.ConnectError:
                continue
        return await self._inner.connect_tcp(addresses[-1], port, timeout, local_address, socket_options)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's backend signature
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise BlockedDestination("unix sockets are not a fetch destination", hint=_HINT)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)
