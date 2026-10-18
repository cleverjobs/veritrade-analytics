"""Skeletons must not carry publisher text: every string is built from the fixed synthetic vocabulary."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import lxml.html
import pytest

from ._skeletons import Skeleton, skeletons, vocabulary

SKELETONS = skeletons()
VOCAB = vocabulary()
WORDS = frozenset(VOCAB["words"])
PHRASES = sorted((p.strip() for p in VOCAB["phrases"]), key=len, reverse=True)
INVALID_URL = VOCAB["url"]
JSONLD_KEEP = frozenset(VOCAB["jsonld_keep_keys"])
MARKER = re.compile(r"VT[BX]-\d+")
TOKEN = re.compile(r"[^\W\d_]+|\d+")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][\d:.+\-Z]*)?$")
TEXT_ATTRS = ("alt", "title", "aria-label", "placeholder", "aria-description")
TEXT_METAS = frozenset({"description", "og:title", "og:description", "twitter:title", "twitter:description"})


def synthetic(text: str) -> bool:
    """Only vocabulary words and markers, optionally after one stop/boilerplate phrase at the start."""
    rest = text.strip()
    for phrase in PHRASES:
        if rest.lower().startswith(phrase.lower()):
            rest = rest[len(phrase) :]
            break
    rest = MARKER.sub(" ", rest).replace(VOCAB["contact_email"], " ")
    return all(token.lower() in WORDS for token in TOKEN.findall(rest))


def _ids(skeleton: Skeleton) -> str:
    return skeleton.source


def _strings(value: Any, key: str = "") -> Iterator[tuple[str, str]]:
    """(key, string) pairs; structural keys such as @context are skipped with everything under them."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in JSONLD_KEEP:
                yield from _strings(v, k)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, key)
    elif isinstance(value, str):
        yield key, value


def test_vocabulary_is_small_and_plain() -> None:
    assert 40 <= len(WORDS) <= 150
    assert all(re.fullmatch(r"[a-z]+", w) for w in WORDS)


def test_detector_rejects_real_prose() -> None:
    assert synthetic("VTB-3 The company said the plan was shared with the board.")
    assert synthetic("About VTX-9 the group")
    assert not synthetic("Shares of Acme jumped 12% after earnings beat estimates.")
    assert not synthetic("VTB-3 the company said revenue")


@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_text_nodes_are_synthetic(skeleton: Skeleton) -> None:
    root = lxml.html.document_fromstring(skeleton.html)
    offenders = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            offenders.append("comment or processing instruction")
            continue
        if el.tag in ("script", "style"):
            ld = el.tag == "script" and "ld+json" in (el.get("type") or "")
            if el.text and el.text.strip() and not ld:
                offenders.append(f"non-empty <{el.tag}>")
        elif el.text and not synthetic(el.text):
            offenders.append(f"<{el.tag}> text")
        if el.tail and not synthetic(el.tail):
            offenders.append(f"<{el.tag}> tail")
    assert offenders == []


@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_jsonld_strings_are_synthetic(skeleton: Skeleton) -> None:
    root = lxml.html.document_fromstring(skeleton.html)
    offenders = []
    for script in root.iter("script"):
        if "ld+json" not in (script.get("type") or ""):
            continue
        for key, value in _strings(json.loads(script.text or "null")):
            structural = key.startswith("@") and key != "@id"
            if structural or value == INVALID_URL or ISO_DATE.match(value) or synthetic(value):
                continue
            offenders.append(key)
    assert offenders == []


@pytest.mark.parametrize("skeleton", SKELETONS, ids=_ids)
def test_attributes_carry_no_publisher_text_or_links(skeleton: Skeleton) -> None:
    root = lxml.html.document_fromstring(skeleton.html)
    offenders = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for name in TEXT_ATTRS:
            if (value := el.get(name)) and not synthetic(value):
                offenders.append(f"<{el.tag} {name}>")
        for name in ("href", "src"):
            if (value := el.get(name)) and value != INVALID_URL:
                offenders.append(f"<{el.tag} {name}>")
        key = (el.get("property") or el.get("name") or "").lower()
        content = el.get("content") or ""
        if el.tag == "meta" and key in TEXT_METAS and content != INVALID_URL and not synthetic(content):
            offenders.append(f"<meta {key}>")
    assert offenders == []
