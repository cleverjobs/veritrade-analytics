from __future__ import annotations

import json

from veritrade.extract import cascade
from veritrade.extract.dom import parse

SENTENCE = "The regional office said the new plan was shared with the board during the week."
YAHOO = "https://finance.yahoo.com/news/acme-plan-123.html"


def page(*, ld_body: str | None = None, yahoo_paragraphs: int = 0, other_paragraphs: int = 0) -> str:
    ld = ""
    if ld_body is not None:
        payload = json.dumps({"@type": "NewsArticle", "headline": "H", "articleBody": ld_body})
        ld = f'<script type="application/ld+json">{payload}</script>'
    yahoo = "".join(f"<p>Y{i} {SENTENCE}</p>" for i in range(yahoo_paragraphs))
    other = "".join(f"<p>D{i} {SENTENCE}</p>" for i in range(other_paragraphs))
    return f"""<html><head>{ld}</head><body>
      <div data-testid="article-body">{yahoo}</div><div class="story">{other}</div>
    </body></html>"""


def test_jsonld_wins_without_building_the_tree() -> None:
    dom = parse(page(ld_body=" ".join([SENTENCE] * 6), yahoo_paragraphs=8))

    method, paragraphs = cascade.run(dom, YAHOO, "yahoo")

    assert method == "jsonld"
    assert paragraphs
    assert dom.parse_ms == 0.0


def test_short_jsonld_falls_through_to_the_profile() -> None:
    method, paragraphs = cascade.run(parse(page(ld_body=SENTENCE, yahoo_paragraphs=8)), YAHOO, "yahoo")

    assert method == "profile"
    assert len(paragraphs) == 8


def test_unknown_source_uses_density() -> None:
    method, paragraphs = cascade.run(parse(page(other_paragraphs=6)), "https://example.com/a", None)

    assert method == "density"
    assert len(paragraphs) == 6


def test_thin_profile_yields_to_a_richer_density_pass() -> None:
    method, paragraphs = cascade.run(parse(page(yahoo_paragraphs=1, other_paragraphs=8)), YAHOO, "yahoo")

    assert method == "density"
    assert len(paragraphs) == 8


def test_ties_go_to_the_more_specific_method() -> None:
    method, paragraphs = cascade.run(parse(page(yahoo_paragraphs=3)), YAHOO, "yahoo")

    assert (method, len(paragraphs)) == ("profile", 3)
