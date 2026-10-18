"""Agent packets: numbered page lines for gold selection, capped gold bodies for labelling and adjudication.

Gold packets list every visible text block of the page (strip-tags) as an addressable line; menus and the
page chrome far from the extracted article are folded into marker lines that keep their numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from veritrade.errors import VtError
from veritrade.extract import registry
from veritrade.extract.dom import normalize, parse, split_sentences
from veritrade_evals.articles import (
    LABEL_PACKET_WORDS,
    PROMPTS,
    ArticlePaths,
    FetchRecord,
    rel,
    write_json,
)
from veritrade_evals.articles_validate import (
    LabelCheck,
    adjudicate_requested,
    gold_body,
    gold_check,
    label_check,
)

TINY_WORDS = 3
TINY_CHARS = 32
TINY_RUN = 3
MARGIN_LINES = 12
TITLE_REACH = 200
LONG_LINE_WORDS = 250
CHUNK_WORDS = 120
ANCHOR_CHARS = 60
MIN_PARTIAL_CHARS = 40


def split_long(text: str) -> list[str]:
    """Very long blocks (unparagraphed filings) become sentence chunks so ranges stay fine-grained."""
    if len(text.split()) <= LONG_LINE_WORDS:
        return [text]
    chunks: list[list[str]] = [[]]
    count = 0
    for sentence in split_sentences(text):
        size = len(sentence.split())
        if chunks[-1] and count + size > CHUNK_WORDS:
            chunks.append([])
            count = 0
        chunks[-1].append(sentence)
        count += size
    out: list[str] = []
    for chunk in chunks:
        words = " ".join(chunk).split()
        # A run-on "sentence" (a flattened table) is cut by words.
        out += [" ".join(words[i : i + LONG_LINE_WORDS]) for i in range(0, len(words), LONG_LINE_WORDS)]
    return out


def page_blocks(html: str) -> list[tuple[str, str]]:
    """(text, tag) per line: the strip-tags blocks (selectolax), long ones split, each with its tag."""
    blocks = parse(html, "selectolax").iter_blocks()
    return [(line, block.tag) for block in blocks for line in split_long(block.text)]


def page_lines(html: str) -> list[str]:
    return [text for text, _ in page_blocks(html)]


def is_tiny(line: str) -> bool:
    return len(line.split()) <= TINY_WORDS and len(line) <= TINY_CHARS


def _key(text: str) -> str:
    return normalize(text).casefold()


def _matches(line: str, paragraph: str, anchor: str) -> bool:
    return anchor in line or (len(line) >= MIN_PARTIAL_CHARS and line in paragraph)


def locate(
    lines: Sequence[str], title: str | None, paragraphs: Sequence[str], headings: Sequence[int] = ()
) -> tuple[int, int] | None:
    """0-based (first, last) lines of the extracted article, widened up to its headline when it is near.

    The headline is the line matching the extracted title, else the nearest h1 line (`headings`) above.
    """
    if not paragraphs:
        return None
    keys = [_key(line) for line in lines]
    first = next(
        (
            i
            for p in paragraphs[:3]
            for i, k in enumerate(keys)
            if _matches(k, _key(p), _key(p)[:ANCHOR_CHARS])
        ),
        None,
    )
    if first is None:
        return None
    last = next(
        (
            i
            for p in reversed(paragraphs[-3:])
            for i in range(len(keys) - 1, first - 1, -1)
            if _matches(keys[i], _key(p), _key(p)[-ANCHOR_CHARS:])
        ),
        None,
    )
    if last is None:
        return None
    reach = range(first - 1, max(-1, first - TITLE_REACH), -1)
    wanted = _key(title) if title else ""
    for i in reach if wanted else ():
        if keys[i] == wanted or (len(keys[i]) >= 20 and (wanted in keys[i] or keys[i] in wanted)):
            return i, last
    h1 = set(headings)
    return next((i for i in reach if i in h1), first), last


@dataclass(frozen=True, slots=True)
class Row:
    """One packet row: a visible line (text set) or a folded run of lines (text None)."""

    start: int
    end: int
    text: str | None = None
    note: str = ""


BEFORE = "before the located article"
AFTER = "after the located article"
SHORT = "short lines (menus, buttons, tickers)"


def compact_rows(
    lines: Sequence[str], span: tuple[int, int] | None, pinned: frozenset[int] = frozenset()
) -> list[Row]:
    """Fold lines far from the span and tiny-line runs inside it; `pinned` (0-based) lines stay visible."""
    if not lines:
        return []
    last = len(lines) - 1
    lo, hi = (max(0, span[0] - MARGIN_LINES), min(last, span[1] + MARGIN_LINES)) if span else (0, last)
    notes: list[str | None] = [
        None if i in pinned else BEFORE if i < lo else AFTER if i > hi else None for i in range(len(lines))
    ]
    i = lo
    while i <= hi:
        j = i
        while j + 1 <= hi and {i, j + 1}.isdisjoint(pinned) and is_tiny(lines[i]) and is_tiny(lines[j + 1]):
            j += 1
        if j - i + 1 >= TINY_RUN:
            notes[i : j + 1] = [SHORT] * (j - i + 1)
        i = j + 1
    rows: list[Row] = []
    i = 0
    while i <= last:
        note = notes[i]
        j = i
        while note is not None and j + 1 <= last and notes[j + 1] == note:
            j += 1
        rows.append(Row(i + 1, i + 1, lines[i]) if note is None else Row(i + 1, j + 1, note=note))
        i = j + 1
    return rows


def _num(n: int, width: int) -> str:
    return f"L{n:0{width}d}"


def render_rows(rows: Sequence[Row], total: int) -> list[str]:
    width = max(4, len(str(total)))
    out = []
    for row in rows:
        if row.text is not None:
            out.append(f"{_num(row.start, width)} {row.text}")
        else:
            size = row.end - row.start + 1
            out.append(f"{_num(row.start, width)}-{_num(row.end, width)} [{size} lines hidden: {row.note}]")
    return out


@dataclass(frozen=True, slots=True)
class GoldPacket:
    id: str
    lines: tuple[str, ...]
    rows: tuple[Row, ...]
    text: str
    full_text: str
    body_words: int
    located: bool

    @property
    def visible_words(self) -> int:
        return sum(len(row.text.split()) for row in self.rows if row.text is not None)


def build_gold_packet(record: FetchRecord, html: str, paths: ArticlePaths) -> GoldPacket:
    url = record.final_url or record.url
    blocks = page_blocks(html)
    lines = [text for text, _ in blocks]
    headings = [i for i, (_, tag) in enumerate(blocks) if tag == "h1"]
    try:
        article = registry.extract(html, url, parser="selectolax", extractor="cascade")
    except VtError:
        article = None
    span = locate(lines, article.title, article.paragraphs, headings) if article else None
    rows = compact_rows(lines, span, frozenset(headings))
    width = max(4, len(str(len(lines))))
    full = paths.full_packet(record.id)
    header = [
        f"# Gold packet {record.id}",
        "",
        f"- id: {record.id}",
        f"- source: {record.source}",
        f"- url: {record.url}",
        f"- published_at: {record.published_date or 'unknown'}",
        f"- lines: {len(lines)}; folded runs keep their numbers and every line stays addressable",
        f"- h1 lines (never folded): {', '.join(_num(i + 1, width) for i in headings) or 'none'}",
        f"- every line, nothing folded: {rel(full)}",
        f"- follow: {rel(PROMPTS['gold'])}",
        f"- write your output to: {rel(paths.out('gold', record.id))}",
        "",
        "## Lines",
        "",
    ]
    full_lines = [f"{_num(n, width)} {line}" for n, line in enumerate(lines, start=1)]
    full_header = [
        f"# Full gold packet {record.id} (every line)",
        "",
        f"- compact packet: {rel(paths.packet('gold', record.id))}",
        "",
    ]
    return GoldPacket(
        id=record.id,
        lines=tuple(lines),
        rows=tuple(rows),
        text="\n".join(header + render_rows(rows, len(lines))) + "\n",
        full_text="\n".join(full_header + full_lines) + "\n",
        body_words=article.word_count if article else 0,
        located=span is not None,
    )


def write_gold_packet(packet: GoldPacket, paths: ArticlePaths) -> Path:
    path = paths.packet("gold", packet.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(packet.text, encoding="utf-8")
    paths.full_packet(packet.id).write_text(packet.full_text, encoding="utf-8")
    write_json(paths.lines(packet.id), {"id": packet.id, "lines": list(packet.lines)})
    return path


# --- label and adjudication packets -----------------------------------------------------------------------


def cap_paragraphs(paragraphs: Sequence[str], limit: int = LABEL_PACKET_WORDS) -> tuple[list[str], int]:
    """Whole paragraphs up to `limit` words (a longer first one is cut); also returns the omitted words."""
    kept: list[str] = []
    total = 0
    for paragraph in paragraphs:
        size = len(paragraph.split())
        if total + size > limit:
            if not kept:
                kept.append(" ".join(paragraph.split()[:limit]))
                total = limit
            break
        kept.append(paragraph)
        total += size
    all_words = sum(len(p.split()) for p in paragraphs)
    return kept, all_words - total


def label_body(title: str, body: str) -> list[str]:
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    kept, omitted = cap_paragraphs(paragraphs)
    lines = ["## Title", "", title, "", "## Article", ""]
    for paragraph in kept:
        lines += [paragraph, ""]
    if omitted:
        lines += [f"[{omitted} more words of the article are not shown]", ""]
    return lines


def render_label(article: str, title: str, body: str, paths: ArticlePaths) -> str:
    header = [
        f"# Label packet {article}",
        "",
        f"- id: {article}",
        f"- follow: {rel(PROMPTS['label'])}",
        f"- labeller A writes: {rel(paths.out('label_a', article))}",
        f"- labeller B writes: {rel(paths.out('label_b', article))}",
        "",
    ]
    return "\n".join(header + label_body(title, body))


def _label_lines(name: str, check: LabelCheck) -> list[str]:
    out = check.output
    if out is None:
        return [f"## Labeller {name}", "", "(no valid output)", ""]
    return [
        f"## Labeller {name}",
        "",
        f"- label: {out.label}",
        f"- score: {out.score}",
        f"- entity: {out.entity}",
        f"- ticker: {out.ticker or 'none'}",
        f"- ambiguous: {str(out.ambiguous).lower()}",
        f"- confidence: {out.confidence}",
        f"- rationale: {out.rationale}",
        "",
    ]


def render_adjudicate(
    article: str, title: str, body: str, a: LabelCheck, b: LabelCheck, paths: ArticlePaths
) -> str:
    header = [
        f"# Adjudication packet {article}",
        "",
        f"- id: {article}",
        f"- follow: {rel(PROMPTS['adjudicate'])}",
        f"- write your output to: {rel(paths.out('adjudicate', article))}",
        "",
    ]
    return "\n".join(header + _label_lines("A", a) + _label_lines("B", b) + label_body(title, body))


@dataclass(frozen=True, slots=True)
class Prepared:
    id: str
    path: Path | None
    reason: str = ""
    stats: str = ""


def prepare_label(paths: ArticlePaths, article: str) -> Prepared:
    check = gold_check(paths, article)
    body = gold_body(paths, article)
    if not check.usable or check.title is None or body is None or body != check.body:
        return Prepared(article, None, "gold_not_ingested")
    path = paths.packet("label", article)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_label(article, check.title, body, paths), encoding="utf-8")
    return Prepared(article, path, stats=f"body_words={len(body.split())}")


def prepare_adjudicate(paths: ArticlePaths, article: str) -> Prepared:
    if article not in adjudicate_requested(paths):
        return Prepared(article, None, "not_requested")
    check = gold_check(paths, article)
    body = gold_body(paths, article)
    if not check.usable or check.title is None or body is None:
        return Prepared(article, None, "gold_not_ingested")
    a, b = label_check(paths, "label_a", article), label_check(paths, "label_b", article)
    path = paths.packet("adjudicate", article)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_adjudicate(article, check.title, body, a, b, paths), encoding="utf-8")
    return Prepared(article, path)
