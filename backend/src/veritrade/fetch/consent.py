"""Yahoo EU consent wall: answer "reject all" so the article loads without granting tracking consent."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from veritrade.errors import ConsentRequired
from veritrade.fetch.detect import CONSENT_HOSTS
from veritrade.fetch.headers import browser_headers

MAX_CONSENT_HOPS = 6
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Lets the fetch layer apply its SSRF policy to every consent hop.
Guard = Callable[[str], Awaitable[None]]


def is_consent_url(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in CONSENT_HOSTS


@dataclass(frozen=True, slots=True)
class ConsentForm:
    action: str
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ConsentOutcome:
    done_url: str | None
    elapsed_ms: float


@dataclass
class _RawForm:
    action: str
    classes: frozenset[str]
    hidden: list[tuple[str, str]] = field(default_factory=list)
    submits: list[tuple[str, str, frozenset[str]]] = field(default_factory=list)


class _FormCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[_RawForm] = []
        self._open: _RawForm | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = frozenset(attr.get("class", "").split())
        if tag == "form":
            self._open = _RawForm(action=attr.get("action", ""), classes=classes)
            self.forms.append(self._open)
            return
        if self._open is None or not (name := attr.get("name", "")):
            return
        kind = attr.get("type", "submit" if tag == "button" else "text").lower()
        if tag == "input" and kind == "hidden":
            self._open.hidden.append((name, attr.get("value", "")))
        elif tag in ("button", "input") and kind == "submit":
            self._open.submits.append((name, attr.get("value", ""), classes))

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._open = None


def _reject_button(form: _RawForm) -> tuple[str, str] | None:
    for name, value, classes in form.submits:
        if name.lower() == "reject" or "reject-all" in classes:
            return name, value
    return None


def parse_consent_form(html: str, page_url: str) -> ConsentForm:
    """Hidden fields plus the reject-all button of the consent form; accept-all is never selected."""
    collector = _FormCollector()
    collector.feed(html)
    collector.close()
    candidates = sorted(collector.forms, key=lambda form: "consent-form" not in form.classes)
    for form in candidates:
        reject = _reject_button(form)
        if reject is None:
            continue
        action = urljoin(page_url, form.action) if form.action else page_url
        if not is_consent_url(action):
            break
        return ConsentForm(action=action, fields=(*form.hidden, reject))
    raise ConsentRequired(
        "consent page has no reject-all form we can submit",
        hint="Yahoo changed its consent page; open the article in a browser or try another source.",
    )


async def reject_all(
    client: httpx.AsyncClient, consent_url: str, *, guard: Guard | None = None
) -> ConsentOutcome:
    """Walk the consent redirects, POST reject-all and stop where the chain leaves the consent hosts.

    Cookies land in the client's jar so the caller retries the original URL once; transport errors propagate.
    """
    if not is_consent_url(consent_url):
        raise ValueError(f"not a consent URL: {consent_url}")
    started = time.perf_counter()
    response, exit_url = await _follow(client, "GET", consent_url, guard=guard)
    if exit_url is not None:
        return ConsentOutcome(exit_url, _ms(started))

    page_url = str(response.url)
    form = parse_consent_form(response.text, page_url)
    action = urlsplit(form.action)
    headers = browser_headers(form.action) | {
        "Origin": f"{action.scheme}://{action.netloc}",
        "Referer": page_url,
        "Sec-Fetch-Site": "same-origin",
    }
    response, exit_url = await _follow(
        client, "POST", form.action, guard=guard, data=dict(form.fields), headers=headers
    )
    if exit_url is None:
        raise ConsentRequired(
            f"consent submission was not accepted (HTTP {response.status_code})",
            hint="Yahoo kept showing its consent page; open the article in a browser or try another source.",
            upstream_status=response.status_code,
        )
    return ConsentOutcome(exit_url, _ms(started))


async def _follow(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    guard: Guard | None,
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[httpx.Response, str | None]:
    """Follow redirects while they stay on consent hosts; the URL is set once the chain leaves them."""
    for _ in range(MAX_CONSENT_HOPS):
        if guard is not None:
            await guard(url)
        response = await client.request(
            method,
            url,
            data=data,
            headers=headers or browser_headers(url),
            follow_redirects=False,
        )
        location = response.headers.get("location")
        if response.status_code not in REDIRECT_STATUSES or not location:
            return response, None
        next_url = urljoin(url, location)
        if not is_consent_url(next_url):
            return response, next_url
        method, url, data, headers = "GET", next_url, None, None
    raise ConsentRequired(
        "consent flow redirected too many times",
        hint="Yahoo's consent flow looped; open the article in a browser or try another source.",
    )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
