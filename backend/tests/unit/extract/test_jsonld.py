from __future__ import annotations

import json
from typing import Any

import pytest

from veritrade.extract import jsonld
from veritrade.extract.dom import parse

GRAPH: list[Any] = [
    {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "WebPage", "isAccessibleForFree": "False"},
            {
                "@type": ["https://schema.org/NewsArticle"],
                "headline": "Acme &amp; Co beat",
                "articleBody": "short",
            },
            {
                "@type": "ReportageNewsArticle",
                "name": "Second",
                "datePublished": "2026-10-17T09:00:00Z",
                "articleBody": "Para one.\n\nPara two.",
            },
        ],
    }
]


def test_article_nodes_walk_graphs_and_type_urls() -> None:
    assert len(jsonld.article_nodes(GRAPH)) == 2
    assert jsonld.types_of({"@type": "schema:NewsArticle"}) == {"NewsArticle"}


def test_body_is_the_longest_article_body() -> None:
    assert jsonld.article_body(GRAPH) == "Para one.\n\nPara two."
    assert jsonld.paragraphs(GRAPH, None) == ["Para one.", "Para two."]


def test_body_falls_back_to_any_node() -> None:
    assert jsonld.article_body([{"@type": "WebPage", "articleBody": "x y"}]) == "x y"
    assert jsonld.article_body([{"@type": "WebPage"}]) == ""


def test_metadata_fields() -> None:
    assert jsonld.headline(GRAPH) == "Acme & Co beat"
    assert jsonld.date_published(GRAPH) == "2026-10-17T09:00:00Z"
    assert jsonld.headline([{"@type": "WebPage", "headline": "not an article"}]) is None


@pytest.mark.parametrize(
    ("value", "paywalled"),
    [(False, True), ("False", True), (" false ", True), (True, False), ("True", False)],
)
def test_paywall_flag(value: object, paywalled: bool) -> None:
    assert jsonld.is_paywalled([{"@type": "NewsArticle", "isAccessibleForFree": value}]) is paywalled
    assert jsonld.is_paywalled([{"@type": "NewsArticle"}]) is False


def test_split_body_handles_lines_and_html() -> None:
    assert jsonld.split_body("One.\nTwo.") == ["One.", "Two."]
    assert jsonld.split_body("A line\nwrapped.\n\nTwo.") == ["A line wrapped.", "Two."]
    assert jsonld.split_body("<p>One &amp; two.</p><p>Three<br>four.</p>") == ["One & two.", "Three", "four."]


def test_blob_bodies_are_regrouped_into_paragraphs() -> None:
    sentence = "The company said the new plan was shared with the board this week."  # 13 words
    paragraphs = jsonld.split_body(" ".join([sentence] * 10))

    assert len(paragraphs) == 4
    assert paragraphs[0] == " ".join([sentence] * 3)
    assert all(len(p.split()) <= 90 for p in paragraphs)


def test_strategy_applies_profile_rules() -> None:
    body = "Lede one.\n\nLede two.\n\nBody three.\n\nForward-Looking Statements\n\nLegal text."
    payload = json.dumps({"@type": "NewsArticle", "articleBody": body})
    page = f'<script type="application/ld+json">{payload}</script>'

    assert jsonld.extract(parse(page), "https://example.com/a", None) == [
        "Lede one.",
        "Lede two.",
        "Body three.",
    ]
