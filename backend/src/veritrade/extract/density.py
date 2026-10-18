"""Generic body finder for pages without a profile: the tightest container that holds most of the prose."""

from __future__ import annotations

import re
from functools import lru_cache

from veritrade.extract.dom import Block, Container, Dom
from veritrade.extract.profiles import GENERIC_STOPS, profile_for, refine, stop_marker

# Never article text. <form> stays: ASP.NET pages wrap the whole document in one.
JUNK = (
    "script, style, noscript, template, iframe, svg, nav, header, footer, aside, figure, figcaption, "
    "button, input, select, textarea, dialog"
)
HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
CELLS = frozenset({"td", "th"})
VOTE_MIN_WORDS = 6
VOTE_MAX_LINK_DENSITY = 0.25  # teaser cards and link lists must not pull the container towards them
KEEP_MAX_LINK_DENSITY = 0.5
SHORT_BLOCK_WORDS = 15  # below this a block is held to the stricter vote limit (teaser cards)
# Tightest container still holding this share of the votes; looser lets teasers in, tighter cuts ledes.
SHARE = 0.85
POSITIVE_SHARE = 0.6
HINT_DEPTH = 5  # hints only count close to the text, so a page-level "has-sidebar" class is harmless
NEGATIVE_TOKEN = re.compile(
    r"ads?|advert\w*|banner|breadcrumbs?|byline|caption|comments?|cookie\w*|credits?|disclaimer|footer|menu|"
    r"modal|nav\w*|newsletter\w*|outbrain|popular|promo\w*|recirc\w*|recommend\w*|related\w*|share|sharing|"
    r"sidebar|signup|social|sponsor\w*|subscri\w*|taboola|trending|video\w*"
)
POSITIVE_TOKEN = re.compile(r"(?:article|body|content|entry|main|post|prose|story|text)\w*")
# Tailwind variants such as "xl:group-[:has(#rail-recirc)]" describe layout, not the element's role.
_UTILITY_CLASS = re.compile(r"[:\[\]()/#&!@]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _names(hint: str) -> list[list[str]]:
    """Each class name and the id as lowercase word tokens; utility classes are skipped."""
    out: list[list[str]] = []
    for name in hint.split():
        if _UTILITY_CLASS.search(name):
            continue
        if tokens := [t for t in re.split(r"[-_]+", _CAMEL.sub("-", name).lower()) if t]:
            out.append(tokens)
    return out


@lru_cache(maxsize=4096)
def negative_hint(hint: str) -> bool:
    """A class or id names a non-article role: by its first word, or anywhere with no article-ish word."""
    for tokens in _names(hint):
        if NEGATIVE_TOKEN.fullmatch(tokens[0]):
            return True
        negative = any(NEGATIVE_TOKEN.fullmatch(t) for t in tokens)
        if negative and not any(POSITIVE_TOKEN.fullmatch(t) for t in tokens):
            return True
    return False


@lru_cache(maxsize=4096)
def positive_hint(hint: str) -> bool:
    return not negative_hint(hint) and any(
        POSITIVE_TOKEN.fullmatch(t) for tokens in _names(hint) for t in tokens
    )


def _below(block: Block, until: Container | None) -> list[Container]:
    """Containers between the text and `until` (exclusive), nearest first."""
    out: list[Container] = []
    for container in block.path:
        if container is until:
            break
        out.append(container)
    return out


def _negative(block: Block, until: Container | None) -> bool:
    for container in _below(block, until)[:HINT_DEPTH]:
        if container.tag in ("body", "html"):
            return False
        if negative_hint(container.hint):
            return True
    return False


def _in_cell(block: Block, until: Container | None) -> bool:
    return any(c.tag in CELLS for c in _below(block, until))


def _votes(block: Block, *, cells: bool) -> int:
    words = len(block.text.split())
    if words < VOTE_MIN_WORDS or block.tag in HEADINGS or block.link_density > VOTE_MAX_LINK_DENSITY:
        return 0
    if _negative(block, None) or (not cells and _in_cell(block, None)):
        return 0
    return words


def best_container(blocks: list[Block]) -> Container | None:
    # Data tables never vote, unless the whole page is laid out in one (old EDGAR filer agents).
    for cells in (False, True):
        totals: dict[Container, int] = {}
        voters: dict[Container, int] = {}
        for block in blocks:
            if votes := _votes(block, cells=cells):
                for container in block.path:
                    totals[container] = totals.get(container, 0) + votes
                    voters[container] = voters.get(container, 0) + 1
        if totals:
            top = max(totals.values())
            # Candidates above the share lie on one ancestor chain, so the deepest is the tightest.
            best = max((c for c, n in totals.items() if n >= SHARE * top), key=lambda c: c.depth)
            # Disclosures and pitches outside the story can hold more than 15%; a named body still wins.
            named = [
                c
                for c, n in totals.items()
                if c.depth > best.depth
                and voters[c] > 1
                and n >= POSITIVE_SHARE * totals[best]
                and positive_hint(c.hint)
            ]
            return max(named, key=lambda c: c.depth) if named else best
    return None


def _keep(block: Block, container: Container, stops: tuple[str, ...]) -> bool:
    words = len(block.text.split())
    max_links = KEEP_MAX_LINK_DENSITY if words >= SHORT_BLOCK_WORDS else VOTE_MAX_LINK_DENSITY
    if block.link_density > max_links or _negative(block, container) or _in_cell(block, container):
        return False
    if block.tag not in HEADINGS and (words >= 5 or (words >= 2 and block.text[-1] in '.!?:"”')):
        return True
    # Short "About Acme" headings are not prose, but they must reach the rules to end the body.
    return stop_marker(block.text, stops) is not None


def extract(dom: Dom, url: str, source_key: str | None) -> list[str]:
    dom.remove(JUNK)
    blocks = list(dom.iter_blocks())
    container = best_container(blocks)
    if container is None:
        return []
    profile = profile_for(source_key)
    stops = profile.stops if profile else GENERIC_STOPS
    texts = [b.text for b in blocks if container in b.path and _keep(b, container, stops)]
    return refine(texts, profile)
