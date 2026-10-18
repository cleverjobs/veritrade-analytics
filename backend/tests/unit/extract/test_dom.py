from __future__ import annotations

import pytest

from veritrade.extract.dom import head_title, normalize, parse, scan_json_ld, split_sentences
from veritrade.extract.types import PARSERS, ParserName

PAGE = """<!doctype html><html><head><title>Plain title | Site</title>
<meta property="og:title" content="Social &amp; title">
<script type="application/ld+json">{"@type": "NewsArticle", "headline": "LD headline"}</script>
<script type="application/ld+json"><!-- {"@type": "Organization", "name": "Org"} --></script>
<script type="application/ld+json">{not json</script>
</head><body>
<div id="main" class="story body">
  <p><span class="dropcap">R</span>ogue traders<br>were <b>caught</b>&nbsp;today\u200b.</p>
  <ul><li>First <a href="/x">point</a></li></ul>
  <p>Second<!-- hidden --> paragraph<script>var x = 1;</script> here.</p>
  <div class="ad"><div class="ad"><p>nested ad copy</p></div></div>
</div>
<p class="tail">Outside text</p>
</body></html>"""


@pytest.mark.parametrize("parser", PARSERS)
def test_text_joins_inline_runs_and_separates_breaks(parser: ParserName) -> None:
    dom = parse(PAGE, parser)
    texts = [node.text() for node in dom.select("div#main p, div#main li")]

    assert texts == [
        "Rogue traders were caught today.",
        "First point",
        "Second paragraph here.",
        "nested ad copy",
    ]


@pytest.mark.parametrize("parser", PARSERS)
def test_remove_handles_nested_matches(parser: ParserName) -> None:
    dom = parse(PAGE, parser)

    assert dom.remove("div.ad") == 2
    assert [n.text() for n in dom.select("p")][-2:] == ["Second paragraph here.", "Outside text"]


@pytest.mark.parametrize("parser", PARSERS)
def test_attr_and_tag(parser: ParserName) -> None:
    node = parse(PAGE, parser).select("div#main")[0]

    assert node.tag == "div"
    assert node.attr("class") == "story body"
    assert node.attr("missing") is None


@pytest.mark.parametrize("parser", PARSERS)
def test_blocks_carry_link_density_and_container_path(parser: ParserName) -> None:
    dom = parse(PAGE, parser)
    blocks = {b.text: b for b in dom.iter_blocks()}

    item = blocks["First point"]
    assert item.tag == "li"
    assert item.link_chars == len("point")
    assert 0.4 < item.link_density < 0.5
    assert [c.tag for c in item.path[:3]] == ["li", "ul", "div"]
    assert item.path[2].hint.strip() == "story body main"
    assert blocks["Rogue traders were caught today."].link_density == 0.0
    assert "var x = 1;" not in " ".join(blocks)


@pytest.mark.parametrize("parser", PARSERS)
def test_node_link_density_counts_link_text_like_blocks(parser: ParserName) -> None:
    page = """<html><body>
      <p id="mixed">Read the <a href="/a">full <b>report</b></a> today</p>
      <p id="links"><a href="/b">Related one</a> <a href="/c">Related two</a></p>
      <p id="plain">No links <script>var a = "x";</script>here</p>
      <p id="empty"> </p>
    </body></html>"""
    dom = parse(page, parser)
    density = {node.attr("id"): node.link_density() for node in dom.select("p")}

    assert density["mixed"] == pytest.approx(len("full report") / len("Read the full report today"), abs=0.05)
    assert density["links"] == pytest.approx(1.0, abs=0.05)
    assert density["plain"] == 0.0
    assert density["empty"] == 0.0


@pytest.mark.parametrize("parser", PARSERS)
def test_title_prefers_og_title(parser: ParserName) -> None:
    assert parse(PAGE, parser).title() == "Social & title"
    assert parse("<html><head><title> Only  title </title></head></html>", parser).title() == "Only title"


@pytest.mark.parametrize("parser", PARSERS)
def test_json_ld_is_parser_independent_and_skips_bad_blocks(parser: ParserName) -> None:
    dom = parse(PAGE, parser)

    assert dom.json_ld() == [
        {"@type": "NewsArticle", "headline": "LD headline"},
        {"@type": "Organization", "name": "Org"},
    ]
    assert dom.parse_ms == 0.0  # JSON-LD never builds the tree


@pytest.mark.parametrize("parser", PARSERS)
def test_tree_is_built_lazily_and_timed(parser: ParserName) -> None:
    dom = parse(PAGE, parser)
    assert dom.parser == parser
    assert dom.parse_ms == 0.0
    dom.select("p")
    assert dom.parse_ms > 0.0


@pytest.mark.parametrize("parser", PARSERS)
def test_xhtml_with_encoding_declaration(parser: ParserName) -> None:
    page = '<?xml version="1.0" encoding="UTF-8"?><html><body><p>Hello world</p></body></html>'

    assert [n.text() for n in parse(page, parser).select("p")] == ["Hello world"]


@pytest.mark.parametrize("parser", PARSERS)
def test_empty_document(parser: ParserName) -> None:
    dom = parse("", parser)

    assert dom.select("p") == []
    assert list(dom.iter_blocks()) == []
    assert dom.title() == ""


def test_normalize_collapses_whitespace_and_invisible_characters() -> None:
    assert normalize(" a\u00a0\n b\u200b\u00adc\t") == "a bc"


def test_scan_json_ld_handles_cdata_and_case() -> None:
    page = '<SCRIPT TYPE="application/ld+json">//<![CDATA[\n{"a": 1}\n//]]></SCRIPT>'
    assert scan_json_ld('<script type="application/ld+json"><![CDATA[{"a": 1}]]></script>') == [{"a": 1}]
    assert scan_json_ld(page) == [{"a": 1}]
    assert scan_json_ld('<script type="application/ld+json">{"a": </script><p>') == []


def test_head_title_reads_meta_then_title_tag() -> None:
    assert (
        head_title('<meta content="Meta &amp; title" property="og:title"><title>T</title>') == "Meta & title"
    )
    assert head_title("<title>Just <b>T</b></title>") == "Just T"
    assert head_title("<p>no title</p>") == ""


def test_split_sentences_respects_abbreviations_and_numbers() -> None:
    text = "Acme Inc. rose 3.5% in the U.S. market. J. Smith said so. Is it over? Yes!"

    assert split_sentences(text) == [
        "Acme Inc. rose 3.5% in the U.S. market.",
        "J. Smith said so.",
        "Is it over?",
        "Yes!",
    ]


@pytest.mark.parametrize("parser", PARSERS)
def test_deeply_nested_markup_does_not_recurse(parser: ParserName) -> None:
    page = "<html><body><p>" + "<font>" * 1500 + "deep text" + "</font>" * 1500 + "</p></body></html>"
    dom = parse(page, parser)

    assert [b.text for b in dom.iter_blocks()] == ["deep text"]
    assert dom.select("p")[0].text() == "deep text"
