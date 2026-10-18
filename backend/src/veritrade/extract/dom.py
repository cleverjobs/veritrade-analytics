"""One small DOM protocol over four HTML parsers, so every own strategy runs unchanged on each backend."""

from __future__ import annotations

import html as html_lib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import lxml.etree
import lxml.html
from bs4 import BeautifulSoup
from bs4.element import NavigableString, PreformattedString, Tag
from lxml.cssselect import CSSSelector
from selectolax.lexbor import LexborHTMLParser, LexborNode

from veritrade.extract.types import ParserName

# Text inside these never reaches a reader.
SKIP_TAGS = frozenset(
    {
        "head", "iframe", "math", "noscript", "object", "script", "select", "style", "svg", "template",
        "textarea",
    }
)  # fmt: skip
# Phrasing content joins without a separator; every other element (block, custom, SGML) separates text.
INLINE_TAGS = frozenset(
    {
        "a", "abbr", "acronym", "b", "bdi", "bdo", "big", "cite", "code", "data", "del", "dfn", "em", "font",
        "i", "img", "ins", "kbd", "label", "mark", "nobr", "picture", "q", "s", "samp", "small", "source",
        "span", "strike", "strong", "sub", "sup", "time", "tt", "u", "var", "wbr",
    }
)  # fmt: skip

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\xad"), None)
_XML_DECL = re.compile(r"^\ufeff?\s*<\?xml[^>]*\?>", re.IGNORECASE)
_LD_OPEN = re.compile(r"<script\b[^>]*application/ld\+json[^>]*>", re.IGNORECASE)
_SCRIPT_CLOSE = re.compile(r"</script", re.IGNORECASE)
_LD_WRAPPER = re.compile(r"^\s*(?://\s*)?(?:<!--|<!\[CDATA\[)|(?://\s*)?(?:-->|\]\]>)\s*$")
_META_TITLE = re.compile(
    r"""<meta\b[^>]*?(?:property|name)\s*=\s*["'](?:og:title|twitter:title)["'][^>]*>""", re.IGNORECASE
)
_CONTENT_ATTR = re.compile(r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
_TITLE_TAG = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)


def strip_xml_declaration(html: str) -> str:
    """lxml refuses str input that still declares an encoding (XHTML pages)."""
    return _XML_DECL.sub("", html, count=1)


def normalize(text: str) -> str:
    """Collapse all whitespace (incl. nbsp) and drop zero-width characters."""
    return " ".join(text.translate(_INVISIBLE).split())


_SENTENCE_END = re.compile(r"""[.!?]["'”’)\]]*(?=\s+["'“‘(\[]?[A-Z0-9])""")
_LAST_TOKEN = re.compile(r"(\S+)$")
_ABBREVIATIONS = frozenset(
    {
        "approx", "apr", "aug", "co", "corp", "dec", "dr", "e.g", "est", "feb", "fig", "gen", "gov", "i.e",
        "inc", "jan", "jr", "jul", "jun", "llc", "ltd", "mar", "mr", "mrs", "ms", "no", "nov", "oct", "plc",
        "prof", "rep", "sen", "sep", "sept", "sr", "st", "u.k", "u.s", "vs",
    }
)  # fmt: skip


def split_sentences(text: str) -> list[str]:
    """Rule-based splitter: good enough to cut at a boundary, not a linguistic tool."""
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.group(0)[0] == "." and (token := _LAST_TOKEN.search(text, start, match.start())):
            word = token.group(1)
            if word.lower().rstrip(".") in _ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                continue
        out.append(text[start : match.end()].strip())
        start = match.end()
    if tail := text[start:].strip():
        out.append(tail)
    return out


def scan_json_ld(html: str) -> list[Any]:
    """Parse every JSON-LD script straight from the source string; no DOM needed, same on every parser."""
    out: list[Any] = []
    pos = 0
    while (opening := _LD_OPEN.search(html, pos)) is not None:
        closing = _SCRIPT_CLOSE.search(html, opening.end())
        if closing is None:
            break
        pos = closing.end()
        payload = _LD_WRAPPER.sub("", html[opening.end() : closing.start()]).strip()
        try:
            out.append(json.loads(payload, strict=False))
        except ValueError:
            continue
    return out


def head_title(html: str) -> str:
    """og:title, twitter:title or <title> read from the raw string, for paths that never build a DOM."""
    meta = _META_TITLE.search(html)
    content = _CONTENT_ATTR.search(meta.group(0)) if meta else None
    if content and (text := normalize(html_lib.unescape(content.group(1) or content.group(2) or ""))):
        return text
    if title := _TITLE_TAG.search(html):
        return normalize(html_lib.unescape(re.sub(r"<[^>]+>", " ", title.group(1))))
    return ""


class Node(Protocol):
    @property
    def tag(self) -> str: ...

    def text(self) -> str: ...

    def attr(self, name: str) -> str | None: ...

    def link_density(self) -> float: ...


class Container:
    """A non-inline element on the path of a text block; density scores these."""

    __slots__ = ("_hint", "_hint_of", "depth", "native", "tag")

    def __init__(self, tag: str, native: Any, depth: int, hint_of: Callable[[Any], str]) -> None:
        self.tag = tag
        self.native = native
        self.depth = depth
        self._hint_of = hint_of
        self._hint: str | None = None

    @property
    def hint(self) -> str:
        """Class and id as written, computed only for the containers density inspects."""
        if self._hint is None:
            self._hint = self._hint_of(self.native)
        return self._hint


@dataclass(frozen=True, slots=True)
class Block:
    text: str
    link_chars: int
    tag: str
    path: tuple[Container, ...]  # nearest container first

    @property
    def link_density(self) -> float:
        return self.link_chars / len(self.text) if self.text else 0.0


class Dom(Protocol):
    @property
    def parser(self) -> ParserName: ...

    @property
    def parse_ms(self) -> float: ...

    def select(self, css: str) -> list[Node]: ...

    def remove(self, css: str) -> int: ...

    def iter_blocks(self) -> Iterator[Block]: ...

    def json_ld(self) -> list[Any]: ...

    def title(self) -> str: ...


class _Node[E]:
    __slots__ = ("_dom", "native")

    def __init__(self, dom: _TreeDom[E], native: E) -> None:
        self._dom = dom
        self.native = native

    @property
    def tag(self) -> str:
        return self._dom._tag(self.native)

    def text(self) -> str:
        return self._dom._text(self.native)

    def attr(self, name: str) -> str | None:
        return self._dom._attr(self.native, name)

    def link_density(self) -> float:
        return self._dom._link_density(self.native)


class _TreeDom[E](ABC):
    """Shared walkers; a backend only supplies parse, select, drop and child iteration."""

    name: ParserName

    def __init__(self, html: str) -> None:
        self._html = html
        self._parse_ms = 0.0
        self._built = False
        self._json_ld: list[Any] | None = None

    # --- backend primitives ---------------------------------------------------------------------

    @abstractmethod
    def _build(self, html: str) -> None: ...

    @abstractmethod
    def _select(self, css: str) -> list[E]: ...

    @abstractmethod
    def _drop(self, el: E) -> None: ...

    @abstractmethod
    def _tag(self, el: E) -> str: ...

    @abstractmethod
    def _attr(self, el: E, name: str) -> str | None: ...

    @abstractmethod
    def _content(self, el: E) -> Iterator[E | str]:
        """Children in order: plain text as str, elements as E; comments and doctypes are skipped."""

    @abstractmethod
    def _body(self) -> E | None: ...

    # --- protocol -------------------------------------------------------------------------------

    @property
    def parser(self) -> ParserName:
        return self.name

    @property
    def parse_ms(self) -> float:
        return self._parse_ms

    def _ensure(self) -> None:
        # Lazy so a JSON-LD hit never pays for a parse; the build time is reported separately.
        if not self._built:
            start = time.perf_counter()
            self._build(self._html)
            self._parse_ms = (time.perf_counter() - start) * 1000
            self._built = True

    def select(self, css: str) -> list[Node]:
        self._ensure()
        return [_Node(self, el) for el in self._select(css)]

    def remove(self, css: str) -> int:
        self._ensure()
        matches = self._select(css)
        # Reverse document order detaches nested matches before their ancestors are destroyed.
        for el in reversed(matches):
            self._drop(el)
        return len(matches)

    def json_ld(self) -> list[Any]:
        if self._json_ld is None:
            self._json_ld = scan_json_ld(self._html)
        return self._json_ld

    def title(self) -> str:
        self._ensure()
        for css in ('meta[property="og:title"]', 'meta[name="twitter:title"]'):
            for el in self._select(css):
                if content := normalize(self._attr(el, "content") or ""):
                    return content
        for el in self._select("title"):
            return self._text(el)
        return ""

    def iter_blocks(self) -> Iterator[Block]:
        self._ensure()
        root = self._body()
        if root is None:
            return iter(())
        return iter(self._blocks(root))

    # --- shared walkers -------------------------------------------------------------------------

    def _hint(self, el: E) -> str:
        return f"{self._attr(el, 'class') or ''} {self._attr(el, 'id') or ''}"

    def _text(self, el: E) -> str:
        parts: list[str] = []
        # Explicit stacks, not recursion: unclosed tags in old filings can nest thousands deep.
        stack: list[tuple[Iterator[E | str], bool]] = [(self._content(el), False)]
        while stack:
            children, block = stack[-1]
            item = next(children, None)
            if item is None:
                stack.pop()
                if block:
                    parts.append(" ")
            elif isinstance(item, str):
                parts.append(item)
            elif (tag := self._tag(item)) not in SKIP_TAGS:
                block = tag not in INLINE_TAGS
                if block:
                    parts.append(" ")
                stack.append((self._content(item), block))
        return normalize("".join(parts))

    def _link_density(self, el: E) -> float:
        """Share of visible characters inside links, counted like Block.link_chars."""
        total = links = 0
        stack: list[tuple[Iterator[E | str], bool]] = [(self._content(el), False)]
        while stack:
            children, in_link = stack[-1]
            item = next(children, None)
            if item is None:
                stack.pop()
            elif isinstance(item, str):
                size = len(item.strip())
                total += size
                links += size if in_link else 0
            elif (tag := self._tag(item)) not in SKIP_TAGS:
                stack.append((self._content(item), in_link or tag == "a"))
        return links / total if total else 0.0

    def _blocks(self, root: E) -> list[Block]:
        """Split visible text at every non-inline boundary; each run remembers its container path."""
        out: list[Block] = []
        buffer: list[str] = []
        link_chars = 0
        containers: list[Container] = [Container(self._tag(root), root, 0, self._hint)]

        def flush() -> None:
            nonlocal link_chars
            if buffer:
                if text := normalize("".join(buffer)):
                    path = tuple(reversed(containers))
                    out.append(Block(text, min(link_chars, len(text)), containers[-1].tag, path))
                buffer.clear()
            link_chars = 0

        # (children, closes a container, inside a link)
        stack: list[tuple[Iterator[E | str], bool, bool]] = [(self._content(root), False, False)]
        while stack:
            children, block, in_link = stack[-1]
            item = next(children, None)
            if item is None:
                stack.pop()
                if block:
                    flush()
                    containers.pop()
            elif isinstance(item, str):
                buffer.append(item)
                if in_link:
                    link_chars += len(item.strip())
            elif (tag := self._tag(item)) in SKIP_TAGS:
                continue
            elif tag in INLINE_TAGS:
                stack.append((self._content(item), False, in_link or tag == "a"))
            elif tag == "br":
                buffer.append(" ")
            else:
                flush()
                containers.append(Container(tag, item, len(containers), self._hint))
                stack.append((self._content(item), True, in_link))
        flush()
        return out


# --- selectolax (Lexbor) --------------------------------------------------------------------------


class SelectolaxDom(_TreeDom[LexborNode]):
    name: ParserName = "selectolax"

    def _build(self, html: str) -> None:
        self._tree = LexborHTMLParser(html)

    def _select(self, css: str) -> list[LexborNode]:
        return list(self._tree.css(css))

    def _drop(self, el: LexborNode) -> None:
        el.decompose()

    def _tag(self, el: LexborNode) -> str:
        return el.tag or ""

    def _attr(self, el: LexborNode, name: str) -> str | None:
        value = el.attrs.get(name)
        return value if isinstance(value, str) else None

    def _content(self, el: LexborNode) -> Iterator[LexborNode | str]:
        for child in el.iter(include_text=True):
            tag = child.tag
            if tag == "-text":
                yield child.text_content or ""
            elif tag and tag[0] != "-" and tag[0] != "!":
                yield child

    def _body(self) -> LexborNode | None:
        return self._tree.body or self._tree.root


# --- lxml -----------------------------------------------------------------------------------------


_local = threading.local()


def _lxml_parser() -> lxml.html.HTMLParser:
    # libxml2 silently drops text nested deeper than 256 elements unless huge_tree is set; bodies are
    # already capped at 10 MB by the fetcher. Parser objects are per thread, like the selector cache.
    parser: lxml.html.HTMLParser | None = getattr(_local, "parser", None)
    if parser is None:
        parser = _local.parser = lxml.html.HTMLParser(huge_tree=True)
    return parser


def _lxml_selector(css: str) -> CSSSelector:
    # Compiled XPath objects are not safe to share across threads, so the cache is per thread.
    cache: dict[str, CSSSelector] | None = getattr(_local, "selectors", None)
    if cache is None:
        cache = _local.selectors = {}
    if (selector := cache.get(css)) is None:
        selector = cache[css] = CSSSelector(css, translator="html")
    return selector


class LxmlDom(_TreeDom[lxml.html.HtmlElement]):
    name: ParserName = "lxml"

    def _build(self, html: str) -> None:
        try:
            self._root = lxml.html.document_fromstring(strip_xml_declaration(html), parser=_lxml_parser())
        except lxml.etree.ParserError:
            self._root = lxml.html.document_fromstring("<html><body></body></html>")

    def _select(self, css: str) -> list[lxml.html.HtmlElement]:
        return list(_lxml_selector(css)(self._root))

    def _drop(self, el: lxml.html.HtmlElement) -> None:
        if el.getparent() is not None:
            el.drop_tree()

    def _tag(self, el: lxml.html.HtmlElement) -> str:
        tag = el.tag
        return tag if isinstance(tag, str) else ""

    def _attr(self, el: lxml.html.HtmlElement, name: str) -> str | None:
        value = el.get(name)
        return value if isinstance(value, str) else None

    def _content(self, el: lxml.html.HtmlElement) -> Iterator[lxml.html.HtmlElement | str]:
        if el.text:
            yield el.text
        for child in el:
            if isinstance(child.tag, str):
                yield child
            if child.tail:
                yield child.tail

    def _body(self) -> lxml.html.HtmlElement | None:
        body = self._root.find("body")
        return body if body is not None else self._root


# --- BeautifulSoup (html.parser or lxml tree builder) ---------------------------------------------


class _SoupDom(_TreeDom[Tag]):
    features: str

    def _build(self, html: str) -> None:
        self._soup = BeautifulSoup(html, self.features)

    def _select(self, css: str) -> list[Tag]:
        return list(self._soup.select(css))

    def _drop(self, el: Tag) -> None:
        el.decompose()

    def _tag(self, el: Tag) -> str:
        return el.name

    def _attr(self, el: Tag, name: str) -> str | None:
        value = el.attrs.get(name)
        if isinstance(value, list):
            return " ".join(value)
        return value if isinstance(value, str) else None

    def _content(self, el: Tag) -> Iterator[Tag | str]:
        for child in el.children:
            if isinstance(child, Tag):
                yield child
            # Comments, doctypes and CDATA are PreformattedString subclasses of NavigableString.
            elif isinstance(child, NavigableString) and not isinstance(child, PreformattedString):
                yield str(child)

    def _body(self) -> Tag | None:
        body = self._soup.body
        return body if body is not None else self._soup


class SoupDom(_SoupDom):
    name: ParserName = "bs4"
    features = "html.parser"


class SoupLxmlDom(_SoupDom):
    name: ParserName = "bs4-lxml"
    features = "lxml"


BACKENDS: dict[ParserName, type[_TreeDom[Any]]] = {
    "selectolax": SelectolaxDom,
    "lxml": LxmlDom,
    "bs4": SoupDom,
    "bs4-lxml": SoupLxmlDom,
}


def parse(html: str, parser: ParserName = "selectolax") -> Dom:
    """A Dom over `html`; the tree is built on first use and its build time lands in `parse_ms`."""
    return BACKENDS[parser](html)
