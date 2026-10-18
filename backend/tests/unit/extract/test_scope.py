from __future__ import annotations

import pytest

from veritrade.extract.scope import LEDE_MAX_TOKENS, build_input, estimate_tokens, lede_paragraphs
from veritrade.extract.types import Article

SENTENCE = "The regional office said the new plan was shared with the board."  # 64 chars


def article(paragraphs: list[str], *, title: str = "Acme plan", source: str | None = None) -> Article:
    return Article(
        url="https://example.com/a",
        title=title,
        paragraphs=tuple(paragraphs),
        published_at=None,
        source=source,
        method="profile",
        parser="selectolax",
        extractor="cascade",
        parse_ms=0.0,
        extract_ms=0.0,
    )


def para(sentences: int, tag: str) -> str:
    return " ".join(f"{tag}{i} {SENTENCE}" for i in range(sentences))


def test_lede_is_title_plus_first_two_paragraphs() -> None:
    body = [para(2, "A"), para(2, "B"), para(2, "C")]

    assert build_input(article(body), "lede", 4000) == f"TITLE: Acme plan\n\n{body[0]}\n\n{body[1]}"


def test_lede_uses_the_profile_paragraph_count() -> None:
    body = [para(2, "A"), para(2, "B"), para(2, "C"), para(2, "D")]

    assert lede_paragraphs(article(body, source="sec")) == body[:3]
    assert lede_paragraphs(article(body, source="cnbc")) == body[:2]


def test_tiny_opening_paragraphs_are_topped_up() -> None:
    body = ["CITY, Oct 1 /Wire/ --", "Acme Corp.", para(4, "C"), para(2, "D")]

    assert lede_paragraphs(article(body)) == body[:3]


def test_lede_is_capped_at_a_sentence_boundary() -> None:
    body = [para(20, "A"), para(20, "B")]

    text = build_input(article(body), "lede", 4000)
    lede = text.split("\n\n", 1)[1]

    assert estimate_tokens(lede) <= LEDE_MAX_TOKENS
    assert lede.endswith(".")
    assert body[0].startswith(lede)


def test_full_keeps_whole_paragraphs_under_the_cap() -> None:
    body = [para(3, "A"), para(3, "B"), para(3, "C")]

    assert build_input(article(body), "full", 4000) == "TITLE: Acme plan\n\n" + "\n\n".join(body)
    capped = build_input(article(body), "full", 120)
    assert estimate_tokens(capped) <= 120
    assert capped.endswith(".")
    assert "C0" not in capped


@pytest.mark.parametrize("max_tokens", [5, 17, 33, 64, 100, 257])
def test_estimate_never_exceeds_the_cap(max_tokens: int) -> None:
    body = [para(4, "A"), "x" * 900, para(4, "C")]

    assert estimate_tokens(build_input(article(body), "full", max_tokens)) <= max_tokens


def test_a_giant_first_sentence_is_cut_at_a_word() -> None:
    body = [" ".join(["word"] * 400)]

    text = build_input(article(body, title=""), "full", 50)

    assert not text.startswith("TITLE")
    assert text == " ".join(["word"] * 40)


def test_invalid_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_input(article(["x"]), "full", 0)
