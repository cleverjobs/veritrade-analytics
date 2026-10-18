from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

ParserName = Literal["selectolax", "lxml", "bs4", "bs4-lxml"]
ExtractorName = Literal[
    "cascade",
    "jsonld",
    "profile",
    "density",
    "trafilatura",
    "trafilatura-precision",
    "readability",
    "newspaper",
    "justext",
    "strip-tags",
]
Scope = Literal["lede", "full"]

PARSERS: tuple[ParserName, ...] = get_args(ParserName)
EXTRACTORS: tuple[ExtractorName, ...] = get_args(ExtractorName)
# Own strategies run on every parser backend; third-party extractors bring their own parser.
PARSER_AWARE_EXTRACTORS: frozenset[ExtractorName] = frozenset({"cascade", "jsonld", "profile", "density"})


@dataclass(frozen=True, slots=True)
class Article:
    url: str
    title: str
    paragraphs: tuple[str, ...]
    published_at: str | None
    source: str | None
    method: str
    parser: ParserName | None
    extractor: ExtractorName
    parse_ms: float
    extract_ms: float
    paywalled: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)
