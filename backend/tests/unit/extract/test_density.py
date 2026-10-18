from __future__ import annotations

import pytest

from veritrade.extract import density
from veritrade.extract.dom import parse
from veritrade.extract.types import PARSERS, ParserName

SENTENCE = "The regional office said the new plan was shared with the board during the week"


def para(n: int) -> str:
    return f"<p>Paragraph {n}: {SENTENCE}, and staff noted the schedule.</p>"


def story_page() -> str:
    story = "".join(para(i) for i in range(1, 7))
    related = "".join(
        f'<li><a href="/r{i}">Related headline number {i} goes here today</a></li>' for i in range(8)
    )
    comments = "".join(f"<p>Reader comment {i}: {SENTENCE}, they wrote at length.</p>" for i in range(3))
    return f"""<html><body>
      <header><p>Site banner text that is long enough to count as prose here.</p></header>
      <main>
        <div class="xl:grid-[content]">
          <h1>Headline</h1>
          <div class="StoryBody-storyBody">{story}
            <div class="InlineImage-imageEmbedCaption">An image caption that runs on for several words.</div>
          </div>
          <div class="RelatedContent-list"><ul>{related}</ul></div>
          <section id="comments">{comments}</section>
        </div>
      </main>
      <footer><p>Copyright notice and other footer prose that is quite long.</p></footer>
    </body></html>"""


@pytest.mark.parametrize("parser", PARSERS)
def test_density_keeps_the_story_and_nothing_around_it(parser: ParserName) -> None:
    paragraphs = density.extract(parse(story_page(), parser), "https://example.com/a", None)

    assert [p.split(":")[0] for p in paragraphs] == [f"Paragraph {i}" for i in range(1, 7)]


@pytest.mark.parametrize("parser", PARSERS)
def test_data_tables_do_not_vote_but_layout_tables_still_work(parser: ParserName) -> None:
    rows = "".join(
        f"<tr><td>Row {i} has a long label for the metric</td><td>{i}</td></tr>" for i in range(40)
    )
    page = (
        f"<html><body><div class='post'>{para(1)}{para(2)}{para(3)}</div><table>{rows}</table></body></html>"
    )
    layout = f"<html><body><table><tr><td>{para(1)}{para(2)}{para(3)}</td></tr></table></body></html>"

    assert len(density.extract(parse(page, parser), "https://example.com/", None)) == 3
    assert len(density.extract(parse(layout, parser), "https://example.com/", None)) == 3


@pytest.mark.parametrize("parser", PARSERS)
def test_short_stop_heading_ends_a_release(parser: ParserName) -> None:
    body = "".join(para(i) for i in range(1, 4))
    about = f"<p><b>About Acme</b></p><p>Acme is a company, {SENTENCE}.</p>"
    page = f"<html><body><div>{body}{about}</div></body></html>"

    paragraphs = density.extract(
        parse(page, parser), "https://www.globenewswire.com/news-release/x", "globenewswire"
    )

    assert len(paragraphs) == 3


def test_empty_page() -> None:
    assert density.extract(parse("<html><body><p>Too short.</p></body></html>"), "https://e.com/", None) == []


@pytest.mark.parametrize(
    ("hint", "negative"),
    [
        ("RelatedContent-nonCollapsibleContent", True),
        ("InlineImage-imageEmbedCaption", True),
        ("ssrcss-1emzoi7-PromoHeadline", True),
        ("comments", True),
        ("foolcom-grid-content-sidebar mt-9", False),
        ("xl:group-[:has(#right-rail-recirc-active)]/article:max-w-full", False),
        ("ArticleBody-articleBody", False),
        ("article-body body-text", False),
        ("", False),
    ],
)
def test_negative_hints(hint: str, negative: bool) -> None:
    assert density.negative_hint(hint) is negative


def test_positive_hints() -> None:
    assert density.positive_hint("article-body")
    assert not density.positive_hint("related-article")
    assert not density.positive_hint("flex gap-4")
