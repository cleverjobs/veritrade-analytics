from __future__ import annotations

from collections.abc import Iterator

import pytest

from veritrade.config import get_settings
from veritrade.fetch.headers import accept_encoding

# Synthetic copy of Yahoo's consent form structure (names, classes, entity-encoded value); no real text.
CONSENT_PAGE = """<!doctype html><html><body>
<form method="get" action="/search"><input type="hidden" name="q" value="x"><button>Go</button></form>
<form method="post" class="consent-form" action="">
  <input type="hidden" name="csrfToken" value="tok-123">
  <input type="hidden" name="sessionId" value="3_cc-session_abc">
  <input type="hidden" name="originalDoneUrl" value="https://finance.yahoo.com/?guccounter&#x3D;1">
  <input type="hidden" name="namespace" value="yahoo">
  <input type="hidden" data-impressionBeaconJs="/beacon?x=1" />
  <button type="submit" class="btn secondary accept-all" name="agree" value="agree">Accept all</button>
  <button type="submit" class="btn secondary reject-all" name="reject" value="reject">Reject all</button>
</form></body></html>"""


@pytest.fixture
def consent_page() -> str:
    return CONSENT_PAGE


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """Settings and the encoding probe are cached per process; reset them around env-dependent tests."""
    get_settings.cache_clear()
    accept_encoding.cache_clear()
    yield
    get_settings.cache_clear()
    accept_encoding.cache_clear()
