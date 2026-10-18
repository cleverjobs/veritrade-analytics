"""Cheapest trustworthy strategy first: JSON-LD, then the source profile, then text density."""

from __future__ import annotations

from veritrade.extract import density, jsonld, profiles
from veritrade.extract.dom import Dom

JSONLD_MIN_CHARS = 400
PROFILE_MIN_WORDS = 80


def _words(paragraphs: list[str]) -> int:
    return sum(len(p.split()) for p in paragraphs)


def run(dom: Dom, url: str, source_key: str | None) -> tuple[str, list[str]]:
    """(winning method, paragraphs); a JSON-LD hit returns before the DOM is ever built."""
    tried: list[tuple[str, list[str]]] = []
    body = jsonld.extract(dom, url, source_key)
    if sum(len(p) for p in body) >= JSONLD_MIN_CHARS:
        return "jsonld", body
    tried.append(("jsonld", body))
    if profiles.profile_for(source_key) is not None:
        body = profiles.extract(dom, url, source_key)
        if _words(body) >= PROFILE_MIN_WORDS:
            return "profile", body
        tried.append(("profile", body))
    tried.append(("density", density.extract(dom, url, source_key)))
    # A short release beats an empty density pass; ties go to the more specific method.
    return max(tried, key=lambda item: _words(item[1]))


def extract(dom: Dom, url: str, source_key: str | None) -> list[str]:
    return run(dom, url, source_key)[1]
