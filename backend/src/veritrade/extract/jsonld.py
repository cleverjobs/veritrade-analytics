"""schema.org Article metadata and body from JSON-LD, including @graph containers."""

from __future__ import annotations

import html as html_lib
import re
from collections.abc import Iterator
from typing import Any

from veritrade.extract.dom import Dom, normalize, split_sentences
from veritrade.extract.profiles import Profile, profile_for, refine

ARTICLE_TYPES = frozenset(
    {
        "AnalysisNewsArticle", "Article", "BackgroundNewsArticle", "BlogPosting", "NewsArticle",
        "OpinionNewsArticle", "Report", "ReportageNewsArticle", "ReviewNewsArticle", "TechArticle",
    }
)  # fmt: skip
# Bodies that arrive as one blob are regrouped so "first N paragraphs" still means an opening.
_GROUP_SENTENCES = 3
_GROUP_WORDS = 90
_BLOCK_TAG = re.compile(r"</?(?:p|div|li|h[1-6]|blockquote|ul|ol)\b[^>]*>|<br\s*/?>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")


def nodes(objects: list[Any]) -> Iterator[dict[str, Any]]:
    stack: list[Any] = list(reversed(objects))
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))


def types_of(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    values = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    # "https://schema.org/NewsArticle" and "schema:NewsArticle" name the same type.
    return {re.split(r"[/:#]", v)[-1] for v in values if isinstance(v, str)}


def article_nodes(objects: list[Any]) -> list[dict[str, Any]]:
    return [node for node in nodes(objects) if types_of(node) & ARTICLE_TYPES]


def _text_field(node: dict[str, Any], key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str) and (text := normalize(html_lib.unescape(value))):
        return text
    return None


def headline(objects: list[Any]) -> str | None:
    for node in article_nodes(objects):
        if text := _text_field(node, "headline") or _text_field(node, "name"):
            return text
    return None


def date_published(objects: list[Any]) -> str | None:
    for node in [*article_nodes(objects), *nodes(objects)]:
        if text := _text_field(node, "datePublished"):
            return text
    return None


def is_paywalled(objects: list[Any]) -> bool:
    """True when any node says isAccessibleForFree false; the body may still be complete (metered sites)."""
    for node in nodes(objects):
        value = node.get("isAccessibleForFree")
        if value is False or (isinstance(value, str) and value.strip().lower() == "false"):
            return True
    return False


def article_body(objects: list[Any]) -> str:
    bodies = [b for node in article_nodes(objects) if isinstance(b := node.get("articleBody"), str)]
    if not bodies:
        bodies = [b for node in nodes(objects) if isinstance(b := node.get("articleBody"), str)]
    return max(bodies, key=len, default="")


def split_body(body: str) -> list[str]:
    if "<" in body and _BLOCK_TAG.search(body):
        body = _ANY_TAG.sub("", _BLOCK_TAG.sub("\n\n", body))
    body = html_lib.unescape(body)
    chunks = re.split(r"\n\s*\n", body) if "\n\n" in body else body.split("\n")
    out: list[str] = []
    for chunk in chunks:
        if not (text := normalize(chunk)):
            continue
        if len(text.split()) <= _GROUP_WORDS:
            out.append(text)
        else:
            out.extend(_group(split_sentences(text)))
    return out


def _group(sentences: list[str]) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and (len(current) >= _GROUP_SENTENCES or words + n > _GROUP_WORDS):
            out.append(" ".join(current))
            current, words = [], 0
        current.append(sentence)
        words += n
    if current:
        out.append(" ".join(current))
    return out


def paragraphs(objects: list[Any], profile: Profile | None) -> list[str]:
    return refine(split_body(article_body(objects)), profile)


def extract(dom: Dom, url: str, source_key: str | None) -> list[str]:
    return paragraphs(dom.json_ld(), profile_for(source_key))
