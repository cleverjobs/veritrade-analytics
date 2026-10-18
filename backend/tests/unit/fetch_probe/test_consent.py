from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest
import respx

from veritrade.errors import ConsentRequired
from veritrade.fetch.consent import is_consent_url, parse_consent_form, reject_all

ENTRY = "https://guce.yahoo.com/consent?brandType=eu&done=https%3A%2F%2Ffinance.yahoo.com%2Fnews%2Fx.html"
FORM_URL = "https://consent.yahoo.com/v2/collectConsent?sessionId=3_cc-session_abc"
COPY_URL = "https://guce.yahoo.com/copyConsent?sessionId=3_cc-session_abc"
DONE_URL = "https://finance.yahoo.com/news/x.html?guccounter=1"


def test_is_consent_url() -> None:
    assert is_consent_url(ENTRY)
    assert is_consent_url(FORM_URL)
    assert not is_consent_url("https://finance.yahoo.com/news/x.html")
    assert not is_consent_url("https://consent.yahoo.com.example.org/")


def test_form_parser_picks_reject_and_never_accept(consent_page: str) -> None:
    form = parse_consent_form(consent_page, FORM_URL)
    fields = dict(form.fields)
    assert form.action == FORM_URL
    assert fields["reject"] == "reject"
    assert "agree" not in fields
    assert fields["csrfToken"] == "tok-123"
    assert fields["originalDoneUrl"] == "https://finance.yahoo.com/?guccounter=1"
    assert "q" not in fields


def test_form_without_reject_option_is_consent_required(consent_page: str) -> None:
    accept_only = consent_page.replace('name="reject" value="reject"', 'name="more" value="more"').replace(
        "reject-all", "manage"
    )
    with pytest.raises(ConsentRequired):
        parse_consent_form(accept_only, FORM_URL)


def test_form_posting_off_the_consent_hosts_is_refused(consent_page: str) -> None:
    hostile = consent_page.replace('action=""', 'action="https://collector.example.com/submit"')
    with pytest.raises(ConsentRequired):
        parse_consent_form(hostile, FORM_URL)


def test_reject_button_found_by_class_when_named_differently(consent_page: str) -> None:
    renamed = consent_page.replace('name="reject" value="reject"', 'name="choice" value="necessary"')
    assert dict(parse_consent_form(renamed, FORM_URL).fields)["choice"] == "necessary"


async def test_reject_all_posts_reject_and_keeps_cookies(consent_page: str) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENTRY).respond(302, headers={"Location": FORM_URL})
        router.get(FORM_URL).respond(200, html=consent_page)
        post = router.post(FORM_URL).respond(302, headers={"Location": COPY_URL})
        router.get(COPY_URL).respond(
            302,
            headers={"Location": DONE_URL, "Set-Cookie": "GUC=AQAB; Domain=.yahoo.com; Path=/; Secure"},
        )
        seen: list[str] = []

        async def guard(url: str) -> None:
            seen.append(url)

        async with httpx.AsyncClient() as client:
            outcome = await reject_all(client, ENTRY, guard=guard)
            assert client.cookies.get("GUC", domain=".yahoo.com") == "AQAB"

    assert outcome.done_url == DONE_URL
    assert outcome.elapsed_ms >= 0
    body = parse_qs(post.calls.last.request.content.decode())
    assert body["reject"] == ["reject"]
    assert "agree" not in body
    assert body["csrfToken"] == ["tok-123"]
    assert post.calls.last.request.headers["Origin"] == "https://consent.yahoo.com"
    assert seen == [ENTRY, FORM_URL, FORM_URL, COPY_URL]


async def test_existing_consent_short_circuits_without_posting() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENTRY).respond(302, headers={"Location": DONE_URL})
        async with httpx.AsyncClient() as client:
            outcome = await reject_all(client, ENTRY)
    assert outcome.done_url == DONE_URL


async def test_rejected_submission_is_consent_required(consent_page: str) -> None:
    with respx.mock() as router:
        router.get(FORM_URL).respond(200, html=consent_page)
        router.post(FORM_URL).respond(200, html=consent_page)
        async with httpx.AsyncClient() as client:
            with pytest.raises(ConsentRequired) as caught:
                await reject_all(client, FORM_URL)
    assert caught.value.upstream_status == 200


async def test_redirect_loop_is_consent_required() -> None:
    with respx.mock() as router:
        router.get(ENTRY).respond(302, headers={"Location": ENTRY})
        async with httpx.AsyncClient() as client:
            with pytest.raises(ConsentRequired, match="too many"):
                await reject_all(client, ENTRY)


async def test_reject_all_refuses_non_consent_urls() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="not a consent URL"):
            await reject_all(client, "https://finance.yahoo.com/news/x.html")
