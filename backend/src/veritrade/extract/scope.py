"""What the model reads: the headline plus the lede or the whole body, cut to a token budget."""

from __future__ import annotations

import math
from collections.abc import Sequence

from veritrade.extract.dom import split_sentences
from veritrade.extract.profiles import profile_for
from veritrade.extract.types import Article, Scope

CHARS_PER_TOKEN = 4
LEDE_PARAGRAPHS = 2
LEDE_MIN_TOKENS = 60  # a dateline-only first paragraph is topped up with the next one
LEDE_MAX_TOKENS = 200
_SEP = "\n\n"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def lede_paragraphs(article: Article) -> list[str]:
    profile = profile_for(article.source)
    count = profile.lede_paragraphs if profile else LEDE_PARAGRAPHS
    out = list(article.paragraphs[:count])
    for paragraph in article.paragraphs[count:]:
        if estimate_tokens(_SEP.join(out)) >= LEDE_MIN_TOKENS:
            break
        out.append(paragraph)
    return out


def _cut(paragraph: str, room: int, *, mid_sentence: bool) -> str:
    kept = ""
    for sentence in split_sentences(paragraph):
        candidate = f"{kept} {sentence}" if kept else sentence
        if len(candidate) > room:
            break
        kept = candidate
    if kept or not mid_sentence:
        return kept
    head = paragraph[: room + 1]
    return head[:room] if " " not in head else head.rsplit(" ", 1)[0]


def fit(paragraphs: Sequence[str], budget_chars: int) -> list[str]:
    """Whole paragraphs, then whole sentences; words are cut only when not even one sentence fits."""
    out: list[str] = []
    used = 0
    for paragraph in paragraphs:
        sep = len(_SEP) if out else 0
        if used + sep + len(paragraph) <= budget_chars:
            out.append(paragraph)
            used += sep + len(paragraph)
            continue
        if partial := _cut(paragraph, budget_chars - used - sep, mid_sentence=not out):
            out.append(partial)
        break
    return out


def build_input(article: Article, scope: Scope, max_tokens: int) -> str:
    """'TITLE: ...' then a blank line and the body; the estimate (chars / 4) never exceeds max_tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    budget = max_tokens * CHARS_PER_TOKEN
    head = f"TITLE: {article.title}" if article.title else ""
    if len(head) > budget:
        return _SEP.join(fit([head], budget))
    room = budget - len(head) - (len(_SEP) if head else 0)
    if scope == "lede":
        body = fit(lede_paragraphs(article), min(room, LEDE_MAX_TOKENS * CHARS_PER_TOKEN))
    else:
        body = fit(article.paragraphs, room)
    return _SEP.join([head, *body] if head else body)
