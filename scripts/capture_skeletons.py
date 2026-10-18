"""Turn one probe capture per source into a committable skeleton: the publisher's DOM, none of its text.

Text is rebuilt from a fixed vocabulary; body paragraphs carry VTB-<k> markers and all other text VTX-<k>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lxml.html
from lxml import etree

from veritrade.extract import jsonld, profiles
from veritrade.extract.dom import normalize, parse, strip_xml_declaration
from veritrade.extract.profiles import BOILERPLATE_LINES, GENERIC_STOPS, Fate, Profile

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "evals/data/private/probe"
PROBE_JSON = ROOT / "evals/data/source_probe.json"
OUT_DIR = ROOT / "backend/tests/fixtures/skeletons"
INVALID_URL = "https://example.invalid/"
EM_DASH = chr(0x2014)
CONTACT_EMAIL = "desk@example.invalid"  # keeps contact lines recognisable without a real address
KEY_ATTR = "data-vtk"

# Neutral, sentiment-free English with stopwords, so stopword-based extractors (justext, newspaper) still
# see prose. Small on purpose: a purity test asserts no other word survives in a fixture.
VOCABULARY = (
    "a", "added", "after", "also", "an", "and", "annual", "are", "as", "at", "be", "board", "by", "can",
    "city", "common", "company", "current", "design", "detail", "during", "early", "final", "first", "for",
    "from", "general", "group", "had", "has", "have", "in", "is", "it", "its", "later", "level", "listed",
    "local", "meeting", "member", "month", "network", "new", "noted", "of", "office", "on", "other",
    "outlined", "panel", "period", "plan", "planned", "process", "program", "project", "public", "recent",
    "region", "regional", "report", "resident", "review", "reviewed", "said", "says", "schedule", "season",
    "second", "series", "service", "several", "shared", "staff", "system", "team", "that", "the", "this",
    "to", "update", "visitor", "was", "week", "were", "while", "will", "with", "would", "year",
)  # fmt: skip

URL_ATTRS = frozenset(
    {
        "action", "background", "cite", "data", "data-href", "data-src", "data-srcset", "data-url",
        "formaction", "href", "longdesc", "poster", "src", "srcset",
    }
)  # fmt: skip
TEXT_ATTRS = frozenset(
    {
        "abbr", "alt", "aria-description", "aria-label", "aria-roledescription", "aria-valuetext", "label",
        "placeholder", "summary", "title", "value",
    }
)  # fmt: skip
DROP_ATTRS = frozenset({"style", "srcdoc", "integrity", "nonce"})
# Structural attributes whose values are markup, not prose; any other value with letters becomes "vt".
KEEP_VALUE_ATTRS = frozenset(
    {
        "aria-expanded", "aria-hidden", "aria-level", "aria-live", "async", "charset", "class", "colspan",
        "crossorigin", "data-component", "data-cy", "data-module", "data-test", "data-testid", "datetime",
        "decoding", "defer", "dir", "height", "hidden", "http-equiv", "itemprop", "itemscope", "lang",
        "loading", "media", "method", "open", "rel", "reversed", "role", "rowspan", "scope", "sizes", "start",
        "tabindex", "target", "type", "viewbox", "width",
    }
)  # fmt: skip
NAMESPACE_ATTRS = frozenset({"xmlns", "itemtype"})  # schema/namespace URIs, not publisher links
# Ids survive only when every word in them is layout vocabulary; heading-derived slugs are renamed.
STRUCTURAL_ID_WORDS = frozenset(
    {
        "ad", "ads", "app", "article", "body", "bottom", "comments", "container", "content", "footer",
        "gallery", "global", "header", "image", "inner", "layout", "left", "logo", "main", "media", "menu",
        "nav", "navigation", "outer", "page", "player", "primary", "related", "right", "root", "search",
        "secondary", "section", "sidebar", "site", "skip", "story", "text", "top", "video", "wrapper",
    }
)  # fmt: skip
TITLE_METAS = frozenset({"og:title", "twitter:title", "title", "sailthru.title", "parsely-title"})
STRUCTURAL_METAS = frozenset(
    {
        "viewport", "robots", "googlebot", "og:type", "og:locale", "twitter:card", "theme-color",
        "format-detection", "referrer", "color-scheme", "content-language", "content-type",
    }
)  # fmt: skip
# JSON-LD values that are structure, not prose: kept verbatim and exempt from the purity test.
JSONLD_KEEP_KEYS = frozenset(
    {
        "@context", "@type", "contentReferenceTime", "cssSelector", "dateCreated", "dateModified",
        "datePublished", "duration", "encodingFormat", "height", "inLanguage", "isAccessibleForFree",
        "position", "uploadDate", "width", "wordCount",
    }
)  # fmt: skip
SKIP_CONTENT = frozenset({"script", "style", "noscript", "template", "svg", "math", "iframe", "object"})

_URL_LIKE = re.compile(
    r"^(?:https?:)?//|^www\.|^/[\w./-]|^mailto:|^tel:|\.(?:jpe?g|png|gif|webp|svg|html?)\b"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][\d:.+\-Z]*)?$")
_LETTER_RUN = re.compile(r"[A-Za-z]{3,}")
_SELECTOR_ATTR = re.compile(r"\[([\w-]+)")
_ALNUM = re.compile(r"[^\W_]")
_WS_EDGES = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)


@dataclass
class Synth:
    """Deterministic text factory; markers are numbered in document order."""

    rng: random.Random
    vtb: int = 0
    vtx: int = 0

    def words(self, count: int, *, end: str = "") -> str:
        out: list[str] = []
        sentence = 0
        for _ in range(max(count, 1)):
            word = self.rng.choice(VOCABULARY)
            if sentence == 0:
                word = word.capitalize()
            sentence += 1
            if sentence >= self.rng.randint(9, 17):
                word += "."
                sentence = 0
            elif self.rng.random() < 0.08:
                word += ","
            out.append(word)
        text = " ".join(out).rstrip(".,")
        return text + end

    def text(self, original: str, marker: str | None) -> str:
        lead, core, trail = _WS_EDGES.match(original).groups()  # type: ignore[union-attr]
        count = len(core.split())
        end = "." if core[-1] in ".!?" else ":" if core[-1] == ":" else ""
        body = self.words(count, end=end)
        if marker:
            rest = body.split(" ", 1)
            body = marker if len(rest) == 1 else f"{marker} {rest[1]}"
            if count == 1 and end:
                body += end
        return f"{lead}{body}{trail}"

    def next_vtb(self) -> str:
        self.vtb += 1
        return f"VTB-{self.vtb}"

    def next_vtx(self) -> str:
        self.vtx += 1
        return f"VTX-{self.vtx}"


@dataclass
class Plan:
    keep: set[str]  # data-vtk keys of kept body paragraphs
    special: dict[str, tuple[str, str]]  # key -> (mode, phrase) for stop, boilerplate and contact lines
    dropped: set[str]
    title_key: str | None


def plan_body(annotated: str, profile: Profile) -> Plan:
    """Replay the profile strategy on the annotated page so the skeleton matches what it keeps."""
    dom = parse(annotated, "lxml")
    nodes = profiles.select_body(dom, profile)
    texts = [node.text() for node in nodes]
    keep: set[str] = set()
    special: dict[str, tuple[str, str]] = {}
    for node, verdict in zip(nodes, profiles.triage(texts, profile), strict=True):
        key = node.attr(KEY_ATTR) or ""
        if verdict.fate is Fate.KEEP:
            keep.add(key)
        elif verdict.fate is Fate.STOP and verdict.phrase:
            special[key] = ("stop", verdict.phrase)
        elif verdict.fate is Fate.BOILERPLATE and verdict.phrase:
            special[key] = ("boilerplate", verdict.phrase.capitalize())
        elif verdict.fate is Fate.CONTACT:
            special[key] = ("contact", CONTACT_EMAIL)
    fresh = parse(annotated, "lxml")
    dropped = {node.attr(KEY_ATTR) or "" for css in profile.drop for node in fresh.select(css)}
    titles = fresh.select(profile.title) if profile.title else []
    return Plan(keep, special, dropped, titles[0].attr(KEY_ATTR) if titles else None)


@dataclass
class Walker:
    synth: Synth
    plan: Plan
    body_seen: set[str] = field(default_factory=set)
    title_marker: str | None = None

    def run(self, root: Any) -> None:
        self._walk(root, kept=None, dropped=False, special=None, in_title=False)

    def _text(
        self, text: str | None, kept: str | None, dropped: bool, special: list[Any] | None
    ) -> str | None:
        if not text or not _ALNUM.search(text):
            return text
        if special is not None:
            # The first text of a stop, boilerplate or contact line keeps its trigger so the rules fire again.
            if not special:
                return " "
            mode, phrase = special.pop()
            lead = _WS_EDGES.match(text).group(1)  # type: ignore[union-attr]
            if mode == "boilerplate":
                return f"{lead}{phrase} "
            rest = self.synth.text(" ".join(text.split()[1:]) or "x", self.synth.next_vtx()).strip()
            return f"{lead}{rest} {phrase} " if mode == "contact" else f"{lead}{phrase.rstrip()} {rest} "
        if kept is not None and not dropped:
            marker = None
            if kept not in self.body_seen:
                self.body_seen.add(kept)
                marker = self.synth.next_vtb()
            return self.synth.text(text, marker)
        return self.synth.text(text, self.synth.next_vtx())

    def _walk(
        self, el: Any, kept: str | None, dropped: bool, special: list[Any] | None, in_title: bool
    ) -> None:
        key = el.get(KEY_ATTR)
        dropped = dropped or key in self.plan.dropped
        if key in self.plan.keep and kept is None:
            kept = key
        if key in self.plan.special:
            special = [self.plan.special[key]]
        in_title = in_title or (key is not None and key == self.plan.title_key)
        before = self.synth.vtx
        el.text = self._text(el.text, kept, dropped, special)
        if in_title and self.title_marker is None and self.synth.vtx > before:
            self.title_marker = f"VTX-{self.synth.vtx}"
        for child in el:
            if child.tag in SKIP_CONTENT:
                _empty(child)
            else:
                self._walk(child, kept, dropped, special, in_title)
            before = self.synth.vtx
            child.tail = self._text(child.tail, kept, dropped, special)
            if in_title and self.title_marker is None and self.synth.vtx > before:
                self.title_marker = f"VTX-{self.synth.vtx}"


def strip_comments(root: Any) -> None:
    """Remove comments and PIs first, merging their tails, so every text slot is walked exactly once."""
    for node in list(root.iter(etree.Comment, etree.ProcessingInstruction)):
        parent = node.getparent()
        if parent is None:
            continue
        tail, previous = node.tail, node.getprevious()
        parent.remove(node)
        if tail and previous is not None:
            previous.tail = (previous.tail or "") + tail
        elif tail:
            parent.text = (parent.text or "") + tail


def _empty(el: Any) -> None:
    if el.tag == "script" and "ld+json" in (el.get("type") or ""):
        return
    el.text = None
    for child in list(el):
        el.remove(child)


def _url_like(value: str) -> bool:
    return bool(_URL_LIKE.search(value.strip()))


def scrub_attributes(root: Any, synth: Synth, keep_ids: set[str], keep_attrs: set[str]) -> None:
    renamed: dict[str, str] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        meta_key = (el.get("property") or el.get("name") or "").lower() if el.tag == "meta" else ""
        for name, value in list(el.attrib.items()):
            lname = name.lower()
            if lname in DROP_ATTRS or lname.startswith("on") or lname == KEY_ATTR:
                del el.attrib[name]
            elif el.tag == "meta" and lname == "content":
                el.set(name, _meta_content(meta_key, value, el, synth))
            elif lname in NAMESPACE_ATTRS or (el.tag == "meta" and lname in ("name", "property")):
                continue
            elif lname in TEXT_ATTRS:
                if _ALNUM.search(value):
                    el.set(name, synth.words(len(value.split())))
            elif lname in URL_ATTRS or _url_like(value):
                el.set(name, f"{INVALID_URL} 1x" if "srcset" in lname else INVALID_URL)
            elif lname == "id":
                if value not in keep_ids and not _structural_id(value):
                    el.set(name, renamed.setdefault(value, f"vt-id-{len(renamed) + 1}"))
            elif lname in KEEP_VALUE_ATTRS or lname in keep_attrs:
                if lname != "class" and not _identifier(value):
                    el.set(name, "vt")
            elif _LETTER_RUN.search(value):
                el.set(name, "vt")


def _structural_id(value: str) -> bool:
    return all(word.lower() in STRUCTURAL_ID_WORDS for word in _LETTER_RUN.findall(value))


def _identifier(value: str) -> bool:
    return len(value) <= 64 and not any(c.isspace() for c in value)


def _meta_content(key: str, value: str, el: Any, synth: Synth) -> str:
    if (el.get("http-equiv") or "").lower() == "refresh":
        return "0"
    if key in STRUCTURAL_METAS or _ISO_DATE.match(value.strip()) or not _ALNUM.search(value):
        return value
    if _url_like(value):
        return INVALID_URL
    if key in TITLE_METAS:
        return "{title}"  # filled once the title text is known
    return synth.words(min(len(value.split()), 30))


# --- JSON-LD ----------------------------------------------------------------------------------------


def synth_jsonld(value: Any, synth: Synth, title: str, body: str, key: str = "") -> Any:
    """Rebuild a JSON-LD value; strings are rewritten at any depth, including inside lists."""
    if isinstance(value, list):
        return [synth_jsonld(v, synth, title, body, key) for v in value]
    if isinstance(value, str):
        return _jsonld_string(key, value, synth)
    if not isinstance(value, dict):
        return value
    title_keys = {"headline", "alternativeHeadline"}
    if jsonld.types_of(value) & jsonld.ARTICLE_TYPES:
        title_keys.add("name")
    out: dict[str, Any] = {}
    for k, item in value.items():
        if k in JSONLD_KEEP_KEYS:
            out[k] = item
        elif k == "articleBody" and isinstance(item, str):
            out[k] = body
        elif k in title_keys and isinstance(item, str):
            out[k] = title
        else:
            out[k] = synth_jsonld(item, synth, title, body, k)
    return out


def _jsonld_string(key: str, value: str, synth: Synth) -> str:
    if key == "@id" or _url_like(value):
        return INVALID_URL
    if _ISO_DATE.match(value.strip()) or not _ALNUM.search(value):
        return value
    return synth.text(normalize(value), synth.next_vtx())


def rewrite_jsonld(root: Any, synth: Synth, title: str, body_paragraphs: list[str]) -> bool:
    had_body = False
    for script in root.iter("script"):
        if "ld+json" not in (script.get("type") or ""):
            continue
        try:
            data = json.loads(script.text or "", strict=False)
        except ValueError:
            script.text = None
            continue
        original_body = jsonld.article_body([data])
        had_body = had_body or bool(original_body)
        joiner = "\n\n" if "\n" in original_body else " "
        script.text = json.dumps(
            synth_jsonld(data, synth, title, joiner.join(body_paragraphs)), ensure_ascii=False
        ).replace("</", "<\\/")
    return had_body


# --- driver -----------------------------------------------------------------------------------------


def capture(key: str, pick: int, probe: dict[str, Any]) -> dict[str, Any]:
    profile = profiles.profile_for(key)
    source = profiles.sources()[key]
    if profile is None:
        raise SystemExit(f"{key}: no [sources.{key}.profile] in sources.toml")
    articles = {a["n"]: a for a in probe["sources"][key]["articles"]}
    url = articles[pick]["fetch"]["final_url"]
    html = (PROBE_DIR / key / f"{pick}.html").read_text(encoding="utf-8", errors="replace")

    root = lxml.html.document_fromstring(strip_xml_declaration(html))
    strip_comments(root)
    for index, el in enumerate(e for e in root.iter() if isinstance(e.tag, str)):
        el.set(KEY_ATTR, str(index))
    annotated = lxml.html.tostring(root, encoding="unicode")
    plan = plan_body(annotated, profile)

    synth = Synth(random.Random(f"veritrade-skeleton-{key}"))
    walker = Walker(synth, plan)
    walker.run(root)

    slug = f"article-{pick}"
    fixture_url = f"https://{source.domains[0]}/vt-skeleton/{slug}"
    skeleton = lxml.html.tostring(root, encoding="unicode")
    paragraphs = profiles.extract(parse(skeleton, "lxml"), fixture_url, key)
    # The title as the registry reads it; pages without a profile title fall back to <title>.
    titled = parse(skeleton, "lxml").select(profile.title) if profile.title else []
    head_title = root.find(".//title")
    title = titled[0].text() if titled else ""
    if not title and head_title is not None:
        title = normalize(head_title.text or "")
        found = re.search(r"VTX-\d+", title)
        walker.title_marker = found.group(0) if found else None
    if head_title is not None and title:
        head_title.text = title
    selectors = (profile.body, *profile.drop, profile.title or "", profile.paywall or "")
    keep_ids = {m.group(1) for css in selectors for m in re.finditer(r"#([\w-]+)", css)}
    keep_attrs = {m.group(1).lower() for css in selectors for m in _SELECTOR_ATTR.finditer(css)}
    scrub_attributes(root, synth, keep_ids, keep_attrs)
    had_body = rewrite_jsonld(root, synth, title, paragraphs)
    for meta in root.iter("meta"):
        if meta.get("content") == "{title}":
            meta.set("content", title or synth.words(3))

    out = OUT_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    # Punctuation-only text nodes survive verbatim; the repo style bans the em-dash character outright.
    document = lxml.html.tostring(root, encoding="unicode", doctype="<!DOCTYPE html>").replace(EM_DASH, "-")
    (out / f"{slug}.html").write_text(document + "\n", encoding="utf-8")
    expected = {
        "source": key,
        "slug": slug,
        "captured_at": probe.get("generated_at"),
        "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
        "fixture_url": fixture_url,
        "title_marker": walker.title_marker,
        "body_markers": [f"VTB-{k}" for k in range(1, synth.vtb + 1)],
        "jsonld_body": had_body,
        "stop_marker": next((phrase for mode, phrase in plan.special.values() if mode == "stop"), None),
        "bytes": len(document.encode()),
    }
    (out / f"{slug}.expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return expected


def write_vocabulary() -> None:
    phrases = sorted(
        {m for s in profiles.sources().values() if s.profile for m in s.profile.stop_markers}
        | set(GENERIC_STOPS)
        | {line.capitalize() for line in BOILERPLATE_LINES}
    )
    payload = {
        "contact_email": CONTACT_EMAIL,
        "words": list(VOCABULARY),
        "phrases": phrases,
        "url": INVALID_URL,
        "jsonld_keep_keys": sorted(JSONLD_KEEP_KEYS),
    }
    (OUT_DIR / "vocabulary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: uv run python scripts/capture_skeletons.py --source sec --pick sec=2",
    )
    parser.add_argument(
        "--source", action="append", help="source key (repeatable); default: primary + backup"
    )
    parser.add_argument("--pick", action="append", default=[], help="KEY=N: use probe capture N for KEY")
    args = parser.parse_args()
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    picks = {k: int(v) for k, v in (p.split("=", 1) for p in args.pick)}
    keys = args.source or [k for k, s in profiles.sources().items() if s.status in ("primary", "backup")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_vocabulary()
    for key in keys:
        if not (PROBE_DIR / key).exists():
            print(
                f"{key}: no probe captures under {PROBE_DIR / key}; run scripts/probe_sources.py",
                file=sys.stderr,
            )
            continue
        expected = capture(key, picks.get(key, 1), probe)
        print(
            f"{key}: {len(expected['body_markers'])} body paragraphs, title {expected['title_marker']}, "
            f"jsonld_body={expected['jsonld_body']}, {expected['bytes'] // 1024} KB"
        )


if __name__ == "__main__":
    main()
