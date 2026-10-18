"""Opus-labelled article set: ids, file layout, agent output schemas, committed rows and the eval loader.

Publisher text (HTML, titles, bodies) only ever lives under evals/data/private/articles; committed rows carry
URLs, hashes, labels and model-written facts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from veritrade.fetch import normalize_url
from veritrade.inference.types import Sentiment
from veritrade_evals.paths import ARTICLES_DIR, OPUS_PROMPTS_DIR, PRIVATE_ARTICLES_DIR, REPO_ROOT

REFERENCE_DATE = date(2026, 10, 18)
WINDOW_DAYS = 60
WINDOW_START = REFERENCE_DATE - timedelta(days=WINDOW_DAYS)
SEED = 20261018
PER_CLASS = 3
MIN_BODY_WORDS = 120
KEY_FACTS_MIN = 3
KEY_FACTS_MAX = 5
KEY_FACT_MAX_WORDS = 25
RATIONALE_MAX_WORDS = 50
LABEL_PACKET_WORDS = 1200
SHINGLE_WORDS = 10

Stage = Literal["gold", "label_a", "label_b", "adjudicate"]
STAGES: tuple[Stage, ...] = get_args(Stage)
PacketStage = Literal["gold", "label", "adjudicate"]
PACKET_STAGES: tuple[PacketStage, ...] = get_args(PacketStage)
ArticleSplit = Literal["dev", "test"]
RejectReason = Literal[
    "not_article",
    "live_blog",
    "paywalled_stub",
    "roundup",
    "video_only",
    "too_short",
    "non_english",
    "duplicate",
    "other",
]
REJECT_REASONS: tuple[RejectReason, ...] = get_args(RejectReason)
PROMPTS: dict[PacketStage, Path] = {
    "gold": OPUS_PROMPTS_DIR / "20-gold.md",
    "label": OPUS_PROMPTS_DIR / "30-label.md",
    "adjudicate": OPUS_PROMPTS_DIR / "40-adjudicate.md",
}


class ArticlesError(RuntimeError):
    pass


def article_id(url: str) -> str:
    return "art-" + hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:12]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    """Repo-relative path for messages and packets, absolute when outside the repo (tests)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True, slots=True)
class ArticlePaths:
    private: Path = PRIVATE_ARTICLES_DIR
    committed: Path = ARTICLES_DIR

    def candidates(self, source: str) -> Path:
        return self.private / "candidates" / f"{source}.jsonl"

    def view(self, source: str) -> Path:
        return self.private / "views" / f"{source}.tsv"

    @property
    def cache(self) -> Path:
        return self.private / "cache"

    @property
    def discovery_log(self) -> Path:
        return self.private / "discovery_log.jsonl"

    def html(self, article: str) -> Path:
        return self.private / "html" / f"{article}.html"

    def fetched(self, article: str) -> Path:
        return self.private / "fetched" / f"{article}.json"

    def packet(self, stage: PacketStage, article: str) -> Path:
        return self.private / "packets" / stage / f"{article}.md"

    def full_packet(self, article: str) -> Path:
        return self.private / "packets" / "gold" / f"{article}.full.md"

    def lines(self, article: str) -> Path:
        return self.private / "packets" / "gold" / f"{article}.lines.json"

    def out(self, stage: Stage, article: str) -> Path:
        return self.private / "out" / stage / f"{article}.json"

    def out_dir(self, stage: Stage) -> Path:
        return self.private / "out" / stage

    def gold(self, article: str) -> Path:
        return self.private / "gold" / f"{article}.txt"

    @property
    def adjudicate_ids(self) -> Path:
        return self.private / "adjudicate_ids.txt"

    @property
    def rows(self) -> Path:
        return self.committed / "articles.jsonl"

    @property
    def stats(self) -> Path:
        return self.committed / "stats.json"


# --- records ----------------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Candidate(_Model):
    id: str
    source: str
    url: str
    title: str | None = None
    published_at: str | None = None
    via: str
    via_url: str
    query: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def published(self) -> date | None:
        return date.fromisoformat(self.published_at) if self.published_at else None


class FetchRecord(_Model):
    id: str
    source: str
    url: str
    final_url: str | None = None
    fetched_at: str
    status: int | None = None
    error: str | None = None
    html_sha256: str | None = None
    words: int = 0
    method: str | None = None
    published_at: str | None = None
    published_date: str | None = None
    published_from: Literal["page", "candidate"] | None = None
    paywalled: bool = False
    extract_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.html_sha256 is not None

    @property
    def published(self) -> date | None:
        return date.fromisoformat(self.published_date) if self.published_date else None


# --- agent outputs (documented in evals/opus_prompts) ---------------------------------------------------


class GoldOutput(_Model):
    id: str
    usable: bool
    reject_reason: RejectReason | None = None
    title_line: int | None = None
    body_ranges: list[tuple[int, int]] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    entity: str | None = None
    ticker: str | None = None
    provisional_label: Sentiment | None = None


class LabelOutput(_Model):
    id: str
    label: Sentiment
    score: int = Field(ge=-2, le=2)
    entity: str = Field(min_length=1)
    ticker: str | None = None
    rationale: str = Field(min_length=1)
    ambiguous: bool
    confidence: int = Field(ge=1, le=5)


class AdjudicateOutput(_Model):
    id: str
    decision: Literal["keep", "replace"]
    label: Sentiment
    score: int = Field(ge=-2, le=2)
    entity: str = Field(min_length=1)
    ticker: str | None = None
    rationale: str = Field(min_length=1)


OUTPUT_MODELS: dict[Stage, type[GoldOutput] | type[LabelOutput] | type[AdjudicateOutput]] = {
    "gold": GoldOutput,
    "label_a": LabelOutput,
    "label_b": LabelOutput,
    "adjudicate": AdjudicateOutput,
}


def schema_codes(exc: ValidationError) -> list[str]:
    """Error locations and types only; messages can echo input text."""
    codes = []
    for error in exc.errors(include_url=False, include_input=False):
        where = ".".join(str(part) for part in error["loc"]) or "model"
        codes.append(f"schema:{where}:{error['type']}")
    return codes


# --- committed rows ---------------------------------------------------------------------------------------


class LabelSummary(_Model):
    label: Sentiment
    score: int
    ambiguous: bool
    confidence: int


class AdjudicationSummary(_Model):
    decision: Literal["keep", "replace"]
    label: Sentiment
    score: int


class ArticleLabels(_Model):
    a: LabelSummary
    b: LabelSummary
    adjudicated: AdjudicationSummary | None = None
    final: Sentiment


class ArticleRow(_Model):
    id: str
    url: str
    source: str
    published_at: str
    fetched_at: str
    html_sha256: str
    gold_sha256: str
    label: Sentiment
    score: float
    entity: str
    ticker: str | None = None
    rationale: str
    key_facts: list[str]
    split: ArticleSplit
    labels: ArticleLabels


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(row.model_dump_json(exclude_none=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_fetched(paths: ArticlePaths, article: str) -> FetchRecord | None:
    path = paths.fetched(article)
    return FetchRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def iter_fetched(paths: ArticlePaths, source: str | None = None) -> Iterator[FetchRecord]:
    folder = paths.private / "fetched"
    for path in sorted(folder.glob("art-*.json")) if folder.exists() else []:
        record = FetchRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if source is None or record.source == source:
            yield record


def in_window(day: date | None, since: date = WINDOW_START, until: date = REFERENCE_DATE) -> bool:
    return day is not None and since <= day <= until


# --- verify + loader --------------------------------------------------------------------------------------


def row_problems(row: ArticleRow, paths: ArticlePaths) -> list[str]:
    problems = []
    for kind, path, expected in (
        ("html", paths.html(row.id), row.html_sha256),
        ("gold", paths.gold(row.id), row.gold_sha256),
    ):
        if not path.exists():
            problems.append(f"{kind}_missing")
        elif sha256_file(path) != expected:
            problems.append(f"{kind}_drifted")
    return problems


@dataclass(frozen=True, slots=True)
class ArticleExample:
    id: str
    source: str
    url: str
    label: Sentiment
    score: float
    entity: str
    ticker: str | None
    key_facts: tuple[str, ...]
    gold_text: str
    html_path: Path
    split: ArticleSplit


def load_articles(
    split: ArticleSplit | Literal["all"], *, paths: ArticlePaths | None = None
) -> list[ArticleExample]:
    """Committed rows joined with the local gold body and HTML path; missing or drifted files fail loudly."""
    paths = paths or ArticlePaths()
    rows = read_jsonl(paths.rows, ArticleRow)
    if not rows:
        raise ArticlesError(f"no article rows in {rel(paths.rows)}; run `articles finalize` first")
    examples: list[ArticleExample] = []
    problems: list[str] = []
    for row in rows:
        if split != "all" and row.split != split:
            continue
        if found := row_problems(row, paths):
            problems.extend(f"{row.id}:{code}" for code in found)
            continue
        examples.append(
            ArticleExample(
                id=row.id,
                source=row.source,
                url=row.url,
                label=row.label,
                score=row.score,
                entity=row.entity,
                ticker=row.ticker,
                key_facts=tuple(row.key_facts),
                gold_text=paths.gold(row.id).read_text(encoding="utf-8"),
                html_path=paths.html(row.id),
                split=row.split,
            )
        )
    if problems:
        raise ArticlesError(
            f"{len(problems)} local article files missing or drifted, e.g. {', '.join(problems[:4])}; "
            "run `articles verify`"
        )
    return examples
