from __future__ import annotations

from pathlib import Path

from veritrade_evals.articles import ArticlePaths, FetchRecord
from veritrade_evals.articles_packets import (
    LONG_LINE_WORDS,
    Row,
    build_gold_packet,
    cap_paragraphs,
    compact_rows,
    is_tiny,
    locate,
    page_lines,
    render_label,
    render_rows,
    split_long,
    write_gold_packet,
)

PARAGRAPHS = [
    f"Paragraph {n} of the synthetic story explains how Quillon Freight moved {n * 7} containers through "
    f"the harbour while rivals waited, according to figures the company shared with analysts this week."
    for n in range(1, 9)
]
NAV = "".join(f'<li><a href="/s{n}">Section {n}</a></li>' for n in range(30))
PROMOS = "".join(f"<p>Unrelated promo block number {n} with a few more words to read.</p>" for n in range(40))
FOOTER = "".join(f'<li><a href="/f{n}">Footer link {n}</a></li>' for n in range(30))
HTML = f"""<html><head><title>Quillon Freight story</title></head><body>
<nav><ul>{NAV}</ul></nav>
<main><article><h1>Quillon Freight clears the backlog</h1>
<div class="byline">By A. Writer</div>
{"".join(f"<p>{p}</p>" for p in PARAGRAPHS)}
</article></main>
<aside>{PROMOS}</aside>
<footer><ul>{FOOTER}</ul></footer>
</body></html>"""


def record() -> FetchRecord:
    return FetchRecord(
        id="art-0123456789ab",
        source="cnbc",
        url="https://news.example.com/2026/10/16/quillon.html",
        fetched_at="2026-10-18T10:00:00+00:00",
        status=200,
        html_sha256="0" * 64,
        published_date="2026-10-16",
    )


def covered(rows: list[Row]) -> list[int]:
    return [n for row in rows for n in range(row.start, row.end + 1)]


def test_split_long_keeps_text_and_caps_line_length() -> None:
    assert split_long("short line") == ["short line"]
    sentence = " ".join(["word"] * 29) + "."
    text = " ".join([sentence] * 40)
    parts = split_long(text)
    assert len(parts) > 1
    assert " ".join(parts) == text
    assert all(len(p.split()) <= LONG_LINE_WORDS for p in parts)
    run_on = " ".join(["cell"] * (LONG_LINE_WORDS * 2 + 5))
    assert [len(p.split()) for p in split_long(run_on)] == [LONG_LINE_WORDS, LONG_LINE_WORDS, 5]


def test_is_tiny() -> None:
    assert is_tiny("Sign in")
    assert is_tiny("Markets")
    assert not is_tiny("Shares fell 4% after the results")
    assert not is_tiny("Supercalifragilisticexpialidocious-and-more")


def test_compact_rows_fold_tiny_runs_and_keep_every_number() -> None:
    lines = [
        "Home",
        "Markets",
        "Tech",
        "A long headline about the company today",
        "Body one is long enough here",
        "Share",
        "Body two",
    ]
    rows = compact_rows(lines, None)
    assert covered(rows) == list(range(1, len(lines) + 1))
    assert rows[0] == Row(1, 3, note="short lines (menus, buttons, tickers)")
    # Runs shorter than three stay visible.
    assert Row(6, 6, "Share") in rows
    rendered = render_rows(rows, len(lines))
    assert rendered[0].startswith("L0001-L0003 [3 lines hidden")
    assert rendered[1].startswith("L0004 A long headline")


def test_compact_rows_fold_far_regions_around_the_span() -> None:
    lines = [f"line number {n} has several words in it" for n in range(100)]
    rows = compact_rows(lines, (40, 50))
    assert covered(rows) == list(range(1, 101))
    assert rows[0] == Row(1, 28, note="before the located article")
    assert rows[-1] == Row(64, 100, note="after the located article")
    assert rows[1].start == 29


def test_locate_finds_headline_first_and_last_paragraph() -> None:
    lines = ["Menu", "The headline", "By someone", *PARAGRAPHS, "Related stories"]
    assert locate(lines, "The headline", PARAGRAPHS) == (1, 2 + len(PARAGRAPHS))
    assert locate(lines, None, PARAGRAPHS) == (3, 2 + len(PARAGRAPHS))
    # A JSON-LD headline that differs from the visible h1 falls back to the nearest h1 above the body.
    assert locate(lines, "Different wording of the headline", PARAGRAPHS, headings=[1]) == (
        1,
        2 + len(PARAGRAPHS),
    )
    assert locate(lines, "x", ["not on the page at all, anywhere, in any form whatsoever"]) is None
    assert locate(lines, "x", []) is None


def test_compact_rows_never_fold_pinned_lines() -> None:
    lines = [
        "Home",
        "Markets",
        "Tech",
        "Body text that is long enough",
        *[f"filler line {n} with words" for n in range(40)],
    ]
    lines.append("Headline placed after the body")
    rows = compact_rows(lines, (3, 3), pinned=frozenset({1, len(lines) - 1}))
    assert covered(rows) == list(range(1, len(lines) + 1))
    assert Row(2, 2, "Markets") in rows
    assert rows[-1] == Row(len(lines), len(lines), "Headline placed after the body")
    assert rows[-2].note == "after the located article"


def test_page_lines_are_the_strip_tags_blocks() -> None:
    from veritrade.extract import third_party

    assert page_lines(HTML) == third_party.extract("strip-tags", HTML, "https://news.example.com/x")


def test_gold_packet_is_compact_addressable_and_names_its_files(tmp_path: Path) -> None:
    paths = ArticlePaths(private=tmp_path / "private", committed=tmp_path / "committed")
    packet = build_gold_packet(record(), HTML, paths)
    assert packet.located
    assert packet.body_words == sum(len(p.split()) for p in PARAGRAPHS)
    # Only the margin lines around the located article stay visible besides the body itself.
    assert packet.visible_words < 2 * packet.body_words
    assert covered(list(packet.rows)) == list(range(1, len(packet.lines) + 1))
    assert "Quillon Freight clears the backlog" in packet.text
    h1 = packet.lines.index("Quillon Freight clears the backlog") + 1
    assert f"- h1 lines (never folded): L{h1:04d}" in packet.text
    assert all(p in packet.text for p in PARAGRAPHS)
    assert "Footer link 29" not in packet.text
    assert "Footer link 29" in packet.full_text
    assert f"out/gold/{record().id}.json" in packet.text
    assert "opus_prompts/20-gold.md" in packet.text
    write_gold_packet(packet, paths)
    assert paths.packet("gold", record().id).exists()
    assert paths.full_packet(record().id).exists()
    assert paths.lines(record().id).exists()


def test_gold_packet_without_an_extractable_article_shows_every_line(tmp_path: Path) -> None:
    paths = ArticlePaths(private=tmp_path / "private", committed=tmp_path / "committed")
    packet = build_gold_packet(record(), "<html><body><p>Too short.</p></body></html>", paths)
    assert not packet.located
    assert packet.rows == (Row(1, 1, "Too short."),)


def test_cap_paragraphs_stops_at_a_boundary() -> None:
    paras = ["a " * 500, "b " * 500, "c " * 500]
    kept, omitted = cap_paragraphs([p.strip() for p in paras], 1200)
    assert len(kept) == 2
    assert omitted == 500
    kept, omitted = cap_paragraphs(["w " * 1500], 1200)
    assert len(kept[0].split()) == 1200
    assert omitted == 300
    assert cap_paragraphs(["x y"], 1200) == (["x y"], 0)


def test_label_packet_hides_source_url_and_other_labels(tmp_path: Path) -> None:
    paths = ArticlePaths(private=tmp_path / "private", committed=tmp_path / "committed")
    body = "\n\n".join(PARAGRAPHS)
    text = render_label("art-0123456789ab", "Quillon Freight clears the backlog", body, paths)
    assert "Quillon Freight clears the backlog" in text
    assert PARAGRAPHS[-1] in text
    assert "example.com" not in text
    assert "cnbc" not in text
    assert "provisional" not in text
    assert "out/label_a/art-0123456789ab.json" in text
    assert "out/label_b/art-0123456789ab.json" in text
