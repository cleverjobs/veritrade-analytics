"""Per-source profiles from sources.toml and the paragraph rules every own strategy applies."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from veritrade.extract.dom import Dom, Node

# Unambiguous press-release tails; they end the body on any page, on top of a profile's own markers.
GENERIC_STOPS = (
    "Forward-Looking Statement",
    "Forward Looking Statement",
    "Cautionary Statement",
    "Safe Harbor",
    "Media Contact",
    "Investor Contact",
    "Investors Contact",
    "Investor Relations Contact",
    "Contact:",
    "Contacts:",
    "Contact Information",
)
# Whole-line UI leftovers that sit between body paragraphs.
BOILERPLATE_LINES = frozenset(
    {
        "advertisement", "continue reading", "read more", "read next", "recommended stories",
        "related articles", "related content", "related stories", "scroll to continue", "share this article",
        "show less", "show more", "sign up", "skip advertisement", "story continues", "subscribe",
        "view comments",
    }
)  # fmt: skip
_BOILERPLATE_TRAIL = " .:»›>…"
# "About Acme" ends a release, "About 40% of" does not: name-style markers need a capital or a short line.
_HEADING_WORDS = 8
# A short line with an e-mail address is a contact line; dropped, not a stop, since "Send tips to ..."
# promos also sit mid-article.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CONTACT_MAX_WORDS = 15


class Fate(StrEnum):
    KEEP = "keep"
    EMPTY = "empty"
    NESTED = "nested"
    SHORT = "short"
    BOILERPLATE = "boilerplate"
    CONTACT = "contact"
    STOP = "stop"
    AFTER_STOP = "after_stop"


@dataclass(frozen=True, slots=True)
class Verdict:
    fate: Fate
    phrase: str | None = None


@dataclass(frozen=True, slots=True)
class Profile:
    source: str
    body: str
    drop: tuple[str, ...] = ()
    title: str | None = None
    lede_paragraphs: int = 2
    min_words: int = 1
    stop_markers: tuple[str, ...] = ()
    paywall: str | None = None

    @property
    def stops(self) -> tuple[str, ...]:
        return self.stop_markers + tuple(m for m in GENERIC_STOPS if m not in self.stop_markers)


@dataclass(frozen=True, slots=True)
class Source:
    key: str
    name: str
    domains: tuple[str, ...]
    status: str
    kind: str
    profile: Profile | None


def _profile(key: str, raw: dict[str, Any]) -> Profile:
    return Profile(
        source=key,
        body=str(raw["body"]),
        drop=tuple(str(s) for s in raw.get("drop", ())),
        title=raw.get("title"),
        lede_paragraphs=int(raw.get("lede_paragraphs", 2)),
        min_words=int(raw.get("min_words", 1)),
        stop_markers=tuple(str(s) for s in raw.get("stop_markers", ())),
        paywall=raw.get("paywall"),
    )


@cache
def sources() -> dict[str, Source]:
    text = resources.files("veritrade").joinpath("data/sources.toml").read_text(encoding="utf-8")
    out: dict[str, Source] = {}
    for key, raw in tomllib.loads(text).get("sources", {}).items():
        profile = raw.get("profile")
        out[key] = Source(
            key=key,
            name=str(raw.get("name", key)),
            domains=tuple(str(d).lower() for d in raw.get("domains", ())),
            status=str(raw.get("status", "")),
            kind=str(raw.get("kind", "")),
            profile=_profile(key, profile) if isinstance(profile, dict) else None,
        )
    return out


@cache
def _domain_index() -> dict[str, str]:
    return {domain: key for key, source in sources().items() for domain in source.domains}


def source_for_url(url: str) -> str | None:
    """Source key for the URL's host or its closest listed parent domain (uk.finance.yahoo.com -> yahoo)."""
    host = (urlsplit(url).hostname or "").rstrip(".").lower()
    index = _domain_index()
    while host:
        if (key := index.get(host)) is not None:
            return key
        _, _, host = host.partition(".")
    return None


def profile_for(source_key: str | None) -> Profile | None:
    source = sources().get(source_key) if source_key else None
    return source.profile if source else None


# --- paragraph rules --------------------------------------------------------------------------------


def stop_marker(text: str, markers: Sequence[str]) -> str | None:
    lowered = text.lower()
    for marker in markers:
        if not lowered.startswith(marker.lower()):
            continue
        rest = text[len(marker) :]
        if marker.endswith(" ") and rest and not rest[0].isupper() and len(text.split()) > _HEADING_WORDS:
            continue
        return marker
    return None


def boilerplate_line(text: str) -> str | None:
    line = text.lower().rstrip(_BOILERPLATE_TRAIL)
    return line if line in BOILERPLATE_LINES else None


def contact_line(text: str) -> bool:
    return len(text.split()) <= CONTACT_MAX_WORDS and _EMAIL.search(text) is not None


def triage(texts: Sequence[str], profile: Profile | None) -> list[Verdict]:
    """Fate of every candidate paragraph, in order; the skeleton capture replays these to stay in sync."""
    min_words = profile.min_words if profile else 1
    lede = profile.lede_paragraphs if profile else 2
    stops = profile.stops if profile else GENERIC_STOPS
    verdicts: list[Verdict] = []
    anchor = ""
    kept = 0
    stopped = False
    for text in texts:
        if stopped:
            verdicts.append(Verdict(Fate.AFTER_STOP))
            continue
        if not text:
            verdicts.append(Verdict(Fate.EMPTY))
            continue
        # A selector list can match an <li> and the <p> inside it; the outer text already holds the inner.
        if anchor and len(text) < len(anchor) and text in anchor:
            verdicts.append(Verdict(Fate.NESTED))
            continue
        anchor = text
        if phrase := boilerplate_line(text):
            verdicts.append(Verdict(Fate.BOILERPLATE, phrase))
        elif contact_line(text):
            verdicts.append(Verdict(Fate.CONTACT))
        # Before min_words: a two-word "About Acme" heading must still end the body.
        elif kept >= lede and (marker := stop_marker(text, stops)):
            verdicts.append(Verdict(Fate.STOP, marker))
            stopped = True
        elif len(text.split()) < min_words:
            verdicts.append(Verdict(Fate.SHORT))
        else:
            verdicts.append(Verdict(Fate.KEEP))
            kept += 1
    return verdicts


def refine(texts: Sequence[str], profile: Profile | None) -> list[str]:
    return [text for text, v in zip(texts, triage(texts, profile), strict=True) if v.fate is Fate.KEEP]


# --- strategy -------------------------------------------------------------------------------------


def select_body(dom: Dom, profile: Profile) -> list[Node]:
    """Body candidates in document order after the profile's drop selectors ran."""
    for css in profile.drop:
        dom.remove(css)
    return dom.select(profile.body)


def extract(dom: Dom, url: str, source_key: str | None) -> list[str]:
    profile = profile_for(source_key)
    if profile is None:
        return []
    return refine([node.text() for node in select_body(dom, profile)], profile)
