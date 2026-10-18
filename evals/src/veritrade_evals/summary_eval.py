"""Summary evaluation (R6) on the Opus article set: `uv run python -m veritrade_evals.summary_eval <cmd>`.

generate -> score -> spot-check -> judge-prepare -> (Opus judges) -> judge-check / judge-ingest -> report.
Summaries are produced by the serving path on the stored HTML. Generated text, judge packets and judge outputs
stay under evals/data/private/summary; evals/results/summary gets metrics only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, TextIO, get_args

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from veritrade.config import get_settings
from veritrade.errors import LlmUnreachable, ModelUnavailable, SchemaViolation, VtError
from veritrade.extract import registry
from veritrade.extract.dom import split_sentences
from veritrade.extract.scope import build_input
from veritrade.extract.types import ExtractorName, ParserName, Scope
from veritrade.inference.adapters import (
    SUMMARY_MAX_SENTENCES,
    SUMMARY_MAX_WORDS,
    analyze,
    cap_summary,
    chat_options,
    summarize,
)
from veritrade.inference.ollama import FIXED_OPTIONS, OllamaClient
from veritrade.inference.schemas import ANALYZE_MODELS, SummaryOut, ollama_format
from veritrade.inference.types import ChatResult
from veritrade.models_registry import ModelSpec, allowlist
from veritrade.prompts.registry import PromptRegistry, PromptSpec
from veritrade_evals.articles import (
    SEED,
    ArticleExample,
    ArticlePaths,
    ArticleSplit,
    load_articles,
    read_fetched,
    rel,
)
from veritrade_evals.articles_validate import gold_check
from veritrade_evals.metrics import bootstrap_ci
from veritrade_evals.paths import PRIVATE_DIR, RESULTS_DIR
from veritrade_evals.predcache import PredCache, PredError, canonical_json
from veritrade_evals.runner import model_digest
from veritrade_evals.summary_judge import (
    Blinding,
    JudgePaths,
    JudgeRow,
    check_output,
    deblind,
    make_blinding,
    render_packet,
)

SummaryEndpoint = Literal["summary", "analyze"]
SUMMARY_ENDPOINTS: tuple[SummaryEndpoint, ...] = get_args(SummaryEndpoint)
SPLITS: tuple[ArticleSplit, ...] = get_args(ArticleSplit)

# Part of every cache key: bump when generation changes what it records for an unchanged config.
HARNESS_VERSION = "1"
PARSER: ParserName = "selectolax"
EXTRACTOR: ExtractorName = "cascade"
SCOPE: Scope = "full"
MAX_WORDS = SUMMARY_MAX_WORDS
MAX_SENTENCES = SUMMARY_MAX_SENTENCES
EMBED_MODEL = "qwen3-embedding:0.6b"
# Calibrated on the dev spot check (hand labels in evals/results/summary/spotcheck.dev.json).
COVERAGE_THRESHOLD = 0.60
# Every fact flagged at 0.75 was really stated, so it is reported as a conservative lower bound.
STRICT_THRESHOLD = 0.75
SENSITIVITY = (0.65, 0.70, 0.80)
AGREEMENT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
PASS_GATE = 0.8
# 50-judge.md: faithfulness 3 or lower means at least one material problem, not a minor slip.
MATERIAL_MAX_FAITHFULNESS = 3
SPOT_CHECK_N = 10
# Ollama is shared with the grid run, so calls may queue; latency is not measured here.
TIMEOUT_S = 300.0

FloatArray = NDArray[np.float64]


class SummaryEvalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, order=True)
class Pair:
    endpoint: SummaryEndpoint
    model: str

    @property
    def key(self) -> str:
        return f"{self.endpoint}/{self.model}"


PAIRS: tuple[Pair, ...] = (
    Pair("summary", "qwen3.5:2b-q4_K_M"),
    Pair("summary", "qwen3.5:4b-q4_K_M"),
    Pair("analyze", "qwen3.5:4b-q4_K_M"),
    Pair("analyze", "qwen3:4b-instruct-2507-q4_K_M"),
    Pair("analyze", "ministral-3:3b"),
)


def parse_pair(text: str) -> Pair:
    endpoint, sep, model = text.partition("/")
    if not sep or not model or endpoint not in SUMMARY_ENDPOINTS:
        raise argparse.ArgumentTypeError(f"expected ENDPOINT/MODEL with ENDPOINT in {SUMMARY_ENDPOINTS}")
    return Pair("summary" if endpoint == "summary" else "analyze", model)


@dataclass(frozen=True, slots=True)
class SummaryPaths:
    private: Path = PRIVATE_DIR / "summary"
    results: Path = RESULTS_DIR / "summary"

    @property
    def preds(self) -> Path:
        return self.private / "preds"

    @property
    def manifest(self) -> Path:
        return self.private / "manifest.json"

    def scores(self, split: str) -> Path:
        return self.private / "scores" / f"{split}.jsonl"

    def spot_review(self, split: str) -> Path:
        return self.private / "spotcheck" / f"{split}.md"

    @property
    def judge(self) -> JudgePaths:
        return JudgePaths(self.private / "judge")

    @property
    def summary_json(self) -> Path:
        return self.results / "summary.json"

    @property
    def summary_md(self) -> Path:
        return self.results / "summary.md"

    def spot_check(self, split: str) -> Path:
        return self.results / f"spotcheck.{split}.json"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- articles -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalArticle:
    example: ArticleExample
    url: str
    title: str

    @property
    def id(self) -> str:
        return self.example.id

    @property
    def text(self) -> str:
        """Gold title and body: the reference every number and claim is checked against."""
        return f"{self.title}\n\n{self.example.gold_text}" if self.title else self.example.gold_text


def load_eval_articles(split: ArticleSplit, article_paths: ArticlePaths) -> list[EvalArticle]:
    out = []
    for example in load_articles(split, paths=article_paths):
        record = read_fetched(article_paths, example.id)
        url = (record.final_url or record.url) if record is not None else example.url
        out.append(EvalArticle(example, url, gold_check(article_paths, example.id).title or ""))
    return out


def model_input(article: EvalArticle, spec: ModelSpec) -> str:
    """Exactly what the serving path sends: default extraction, full scope, capped by the model."""
    html = article.example.html_path.read_text(encoding="utf-8")
    extracted = registry.extract(html, article.url, parser=PARSER, extractor=EXTRACTOR)
    return build_input(extracted, SCOPE, spec.max_input_tokens)


# --- generation ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenRun:
    pair: Pair
    spec: ModelSpec
    prompt: PromptSpec

    @classmethod
    def resolve(cls, pair: Pair, prompts: PromptRegistry | None = None) -> GenRun:
        spec = allowlist().get(pair.model)
        if spec is None or not spec.serves(pair.endpoint):
            raise SummaryEvalError(f"{pair.model} is not allowlisted for /{pair.endpoint} in models.toml")
        prompt = (prompts or PromptRegistry()).get(pair.endpoint, pair.model, kind=spec.kind)
        return cls(pair, spec, prompt)

    @property
    def variant(self) -> str:
        return "default" if self.pair.endpoint == "summary" else self.spec.field_order or "sentiment-first"

    def run_config(self, split: str) -> dict[str, Any]:
        return {
            "harness": f"summary-{HARNESS_VERSION}",
            "split": split,
            "endpoint": self.pair.endpoint,
            "variant": self.variant,
            "prompt_version": self.prompt.version,
            "max_input_tokens": self.spec.max_input_tokens,
            "parser": PARSER,
            "extractor": EXTRACTOR,
            "scope": SCOPE,
            "options": {**chat_options(self.spec, self.pair.endpoint), **FIXED_OPTIONS},
        }

    def prompt_sha(self) -> str:
        out_type = (
            SummaryOut
            if self.pair.endpoint == "summary"
            else ANALYZE_MODELS[self.spec.field_order or "sentiment-first"]
        )
        payload = {
            "messages": self.prompt.messages("{text}", self.variant),
            "format": ollama_format(out_type),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def cache(self, split: str, digest: str, root: Path) -> PredCache:
        return PredCache(
            self.pair.model,
            self.run_config(split),
            model_digest=digest,
            prompt_sha=self.prompt_sha(),
            root=root / self.pair.endpoint,
        )


def _timing(chat: ChatResult) -> dict[str, float]:
    timing = {"wall_ms": chat.wall_ms}
    timing.update({k: float(v) for k, v in asdict(chat.timings).items() if v is not None})
    return timing


async def generate_one(
    client: OllamaClient, run: GenRun, text: str
) -> tuple[dict[str, Any], dict[str, float]]:
    """One summary through the serving adapter; the record never carries the input."""
    pred: dict[str, Any]
    if run.pair.endpoint == "summary":
        summary = await summarize(client, run.spec, run.prompt, text)
        pred = {"summary": summary.output.summary, "truncated": summary.truncated}
        chat = summary.chat
    else:
        result = await analyze(client, run.spec, run.prompt, text)
        pred = {
            "summary": result.output.summary,
            "truncated": result.truncated,
            "score": result.output.score,
            "entity": result.output.entity,
        }
        chat = result.chat
    raw = _raw_summary(chat.content)
    pred["input_chars"] = len(text)
    pred["raw_words"] = word_count(raw) if raw is not None else None
    pred["capped"] = raw is not None and cap_summary(raw) != raw.strip()
    return pred, _timing(chat)


def _raw_summary(content: str) -> str | None:
    """The model's own summary before the serving cap; None for output cut at the token limit."""
    try:
        body = json.loads(content)
    except ValueError:
        return None
    value = body.get("summary") if isinstance(body, dict) else None
    return value if isinstance(value, str) else None


def _pred_error(exc: VtError) -> PredError:
    # Greedy seeded decoding repeats a schema failure, so it is final; transport failures retry on resume.
    retryable = not isinstance(exc, SchemaViolation) and (exc.retryable or isinstance(exc, ModelUnavailable))
    return PredError(exc.code.value, exc.message[:300], retryable)


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    model_digest: str
    prompt_version: str
    n: int
    done: int
    errors: int
    truncated: int
    updated_at: str


def read_manifest(paths: SummaryPaths) -> dict[str, dict[str, ManifestEntry]]:
    raw = read_json(paths.manifest) or {}
    return {
        split: {k: ManifestEntry.model_validate(v) for k, v in rows.items()} for split, rows in raw.items()
    }


def update_manifest(paths: SummaryPaths, split: str, pair: Pair, entry: ManifestEntry) -> None:
    manifest = read_manifest(paths)
    manifest.setdefault(split, {})[pair.key] = entry
    write_json(
        paths.manifest, {s: {k: e.model_dump() for k, e in rows.items()} for s, rows in manifest.items()}
    )


async def generate(
    client: OllamaClient,
    run: GenRun,
    articles: Sequence[EvalArticle],
    *,
    split: str,
    digest: str,
    paths: SummaryPaths,
    inputs: Callable[[EvalArticle, ModelSpec], str] = model_input,
    log: TextIO = sys.stderr,
) -> ManifestEntry:
    """Resumable and sequential; an unreachable Ollama aborts, other failures are recorded per article."""
    label = f"{run.pair.endpoint} {run.pair.model}"
    with run.cache(split, digest, paths.preds) as cache:
        done = cache.done_ids()
        todo = [a for a in articles if a.id not in done]
        print(f"[{label}] {len(articles) - len(todo)} cached, {len(todo)} to run", file=log, flush=True)
        for i, article in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                pred, timing = await generate_one(client, run, inputs(article, run.spec))
            except LlmUnreachable as exc:
                raise SummaryEvalError(f"{label}: {exc.message}") from exc
            except VtError as exc:
                wall = {"wall_ms": (time.perf_counter() - t0) * 1000}
                cache.append(article.id, article.example.label, error=_pred_error(exc), t=wall)
            else:
                cache.append(article.id, article.example.label, pred=pred, t=timing)
            if i % 10 == 0 or i == len(todo):
                print(f"[{label}] {i}/{len(todo)} new", file=log, flush=True)
        records = cache.records()
    wanted = [records[a.id] for a in articles if a.id in records]
    entry = ManifestEntry(
        key=cache.key,
        model_digest=digest,
        prompt_version=run.prompt.version,
        n=len(articles),
        done=sum(r["error"] is None for r in wanted),
        errors=sum(r["error"] is not None for r in wanted),
        truncated=sum(bool(r.get("truncated")) for r in wanted),
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    update_manifest(paths, split, run.pair, entry)
    return entry


def load_summaries(paths: SummaryPaths, split: str, pair: Pair) -> dict[str, dict[str, Any]]:
    """Cache records of the generate run in the manifest; a changed prompt or config fails loudly."""
    entry = read_manifest(paths).get(split, {}).get(pair.key)
    if entry is None:
        raise SummaryEvalError(f"no generate run for {pair.key} on {split}; run `generate` first")
    cache = GenRun.resolve(pair).cache(split, entry.model_digest, paths.preds)
    if cache.key != entry.key:
        raise SummaryEvalError(f"{pair.key}: prompt or config changed since `generate`; run it again")
    return cache.records()


def summary_text(record: Mapping[str, Any] | None) -> str | None:
    """The served summary: the serving cap applied again, so caches made before it match serving."""
    if record is None or record.get("error") is not None:
        return None
    text = record.get("summary")
    return (cap_summary(text) or None) if isinstance(text, str) else None


def was_capped(record: Mapping[str, Any]) -> bool:
    text = record.get("summary")
    return bool(record.get("capped")) or (isinstance(text, str) and cap_summary(text) != text.strip())


# --- deterministic checks -----------------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Fiscal periods: "Q3" and "third-quarter" are the same fact, and their digits are not amounts.
_PERIOD_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:Q([1-4])|([1-4])Q|H([12])|([12])H)(?![A-Za-z0-9])")
# Clock times: "5:00 p.m." and "5PM" are the same hour; the ":00" is not a number of its own.
_TIME = re.compile(r"(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])")
_FISCAL_YEAR = re.compile(r"(?<![A-Za-z0-9])FY\s?'?(\d{4}|\d{2})(?![A-Za-z0-9])")
# Calendar-quarter idioms in the article ("September quarter", "April to June") ground Q-tokens.
_MONTH_QUARTER = re.compile(r"\b(march|june|september|december)[\s-]+quarter\b", re.IGNORECASE)
_MONTH_RANGE = re.compile(
    r"\b(january|april|july|october)\s*(?:to|through|-|–)\s*(march|june|september|december)\b",
    re.IGNORECASE,
)
_QUARTER_END = {"march": "1", "june": "2", "september": "3", "december": "4"}
_QUARTER_START = {"january": "1", "april": "2", "july": "3", "october": "4"}
_SCALES = (Decimal(10**3), Decimal(10**6), Decimal(10**9))
_PERIOD_WORDS = re.compile(r"\b(first|second|third|fourth)[\s-]+(quarter|half)\b", re.IGNORECASE)
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
_NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty",
        ]
    )
}  # fmt: skip
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
_SUFFIX = re.compile(
    r",?\s+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|l\.p|holdings?|group|"
    r"sa|s\.a|ag|nv|n\.v|se)\.?$",
    re.IGNORECASE,
)
# Trailing words that news copy drops from a company name ("Six Flags Entertainment" -> "Six Flags").
DESCRIPTOR_WORDS = frozenset(
    {
        "entertainment", "technologies", "technology", "industries", "enterprises", "brands", "systems",
        "pharmaceuticals", "therapeutics", "resources", "partners", "bancorp", "financial", "worldwide",
        "solutions", "communications", "international",
    }
)  # fmt: skip
# A first word this generic would match unrelated text, so it never becomes an alias on its own.
GENERIC_FIRST_WORDS = frozenset(
    {
        "the", "american", "america", "united", "general", "first", "national", "international", "global",
        "new", "bank", "china", "chinese", "north", "south", "east", "west", "royal", "british", "federal",
        "central", "great", "western", "eastern", "pacific", "atlantic", "standard", "public", "world",
    }
)  # fmt: skip

Period = tuple[str, str]


def canonical_number(raw: str) -> str:
    """Thousands commas dropped and trailing zeros normalised, so 1,500.0 == 1500."""
    value = raw.replace(",", "")
    try:
        return format(Decimal(value).normalize(), "f")
    except InvalidOperation:
        return value


def _special_tokens(text: str) -> tuple[set[Period], set[str], str]:
    """Fiscal periods, plus clock times and fiscal years as numbers, and the text with all of them blanked."""
    periods: set[Period] = set()
    numbers: set[str] = set()

    def period(match: re.Match[str]) -> str:
        quarter = match.group(1) or match.group(2)
        periods.add(("Q", quarter) if quarter else ("H", match.group(3) or match.group(4)))
        return " "

    def clock(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), match.group(2)
        numbers.add(str(hour) if minute == "00" else f"{hour}:{minute}")
        return " "

    def fiscal(match: re.Match[str]) -> str:
        year = match.group(1)
        numbers.add(year if len(year) == 4 else f"20{year}")
        return " "

    rest = _FISCAL_YEAR.sub(fiscal, _TIME.sub(clock, _PERIOD_TOKEN.sub(period, text)))
    return periods, numbers, rest


def _scaled_down(value: str) -> set[str]:
    """80,000 also grounds `80K` and 2,400,000,000 grounds `2.4B`, the abbreviations summaries use."""
    try:
        number = Decimal(value)
    except InvalidOperation:
        return set()
    out = set()
    for scale in _SCALES:
        scaled = number / scale
        if number >= scale and scaled == scaled.quantize(Decimal("0.001")):
            out.add(format(scaled.normalize(), "f"))
    return out


@dataclass(frozen=True, slots=True)
class Grounding:
    numbers: frozenset[str]
    periods: frozenset[Period]


def grounding(text: str) -> Grounding:
    """Numbers a summary may use: digits (%, currency and units ignored), number words, periods, times."""
    periods, special, rest = _special_tokens(text)
    numbers = {canonical_number(m) for m in _NUMBER.findall(rest)}
    numbers |= {scaled for value in numbers for scaled in _scaled_down(value)} | special
    numbers |= {_NUMBER_WORDS[w.lower()] for w in _NUMBER_WORD.findall(text)}
    for ordinal, unit in _PERIOD_WORDS.findall(text):
        digit = _ORDINALS[ordinal.lower()]
        if unit.lower() == "quarter":
            periods.add(("Q", digit))
        elif digit in ("1", "2"):
            periods.add(("H", digit))
    periods |= {("Q", _QUARTER_END[m.lower()]) for m in _MONTH_QUARTER.findall(text)}
    for start, end in _MONTH_RANGE.findall(text):
        if _QUARTER_START[start.lower()] == _QUARTER_END[end.lower()]:
            periods.add(("Q", _QUARTER_END[end.lower()]))
    return Grounding(frozenset(numbers), frozenset(periods))


def summary_numbers(text: str) -> list[str]:
    """Distinct numbers, clock times and fiscal periods in a summary, in a stable order."""
    periods, special, rest = _special_tokens(text)
    numbers = {canonical_number(m) for m in _NUMBER.findall(rest)} | special
    return sorted(numbers) + sorted(f"{u}{d}" for u, d in periods)


def is_grounded(value: str, source: Grounding) -> bool:
    """`value` is a canonical number or a period token such as Q3 (numbers never start with a letter)."""
    if value[0] in "QH":
        return (value[0], value[1:]) in source.periods
    return value in source.numbers


def unsupported_numbers(text: str, source: Grounding) -> list[str]:
    return [value for value in summary_numbers(text) if not is_grounded(value, source)]


def _normalise(text: str) -> str:
    return " ".join(text.replace("’", "'").replace("‘", "'").split())


@dataclass(frozen=True, slots=True)
class EntityAliases:
    """Names match case-insensitively; tickers case-sensitively, since short ones collide with words."""

    names: frozenset[str]
    tickers: frozenset[str]
    words: tuple[str, ...] = ()


def short_name(name: str) -> str:
    """The name without legal suffixes and trailing descriptor words, never emptied."""
    current = name
    while True:
        shorter = _SUFFIX.sub("", current).strip()
        words = shorter.split()
        if len(words) > 1 and words[-1].lower().strip(".,") in DESCRIPTOR_WORDS:
            shorter = " ".join(words[:-1])
        if not shorter or shorter == current:
            return current
        current = shorter


def entity_aliases(entity: str, ticker: str | None) -> EntityAliases:
    """The gold entity, its short name as a phrase or as all its words anywhere (`UK ... economy`),
    its distinctive first word, and the ticker."""
    name = _normalise(re.sub(r"\([^)]*\)", " ", entity))
    names = {name.lower()} if name else set()
    stripped = short_name(name)
    if stripped:
        names.add(stripped.lower())
    words = tuple(w.lower().removesuffix("'s") for w in stripped.split())
    if len(words) >= 2 and len(words[0]) >= 4 and words[0] not in GENERIC_FIRST_WORDS:
        names.add(words[0])
    tickers: set[str] = set()
    for part in re.split(r"[\s,;/]+", ticker or ""):
        symbol = part.rpartition(":")[2].lstrip("$")
        if symbol:
            tickers.add(symbol)
            if (head := symbol.partition(".")[0]) and head != symbol:
                tickers.add(head)
    return EntityAliases(frozenset(names), frozenset(tickers), words if len(words) >= 2 else ())


def _has(haystack: str, needle: str, letters: str = "a-z") -> bool:
    return re.search(rf"(?<![{letters}0-9]){re.escape(needle)}(?![{letters}0-9])", haystack) is not None


def names_entity(text: str, aliases: EntityAliases) -> bool:
    plain = _normalise(text)
    lower = plain.lower()
    if any(_has(lower, name) for name in aliases.names):
        return True
    if aliases.words and all(_has(lower, word) for word in aliases.words):
        return True
    return any(_has(plain, ticker, "A-Za-z") for ticker in aliases.tickers)


def word_count(text: str) -> int:
    return len(text.split())


def sentences(text: str) -> list[str]:
    """The serving splitter, so abbreviations such as U.S. or Inc. do not end a sentence."""
    return split_sentences(text) or ([text.strip()] if text.strip() else [])


# --- key-fact coverage --------------------------------------------------------------------------------------


def unit_rows(vectors: Sequence[Sequence[float]]) -> FloatArray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 0, norms, 1.0)


def best_matches(facts: FloatArray, sentence_vecs: FloatArray) -> tuple[list[float], list[int]]:
    """Per fact: the best cosine over the summary sentences and that sentence's index (-1: no sentence)."""
    if len(facts) == 0:
        return [], []
    if len(sentence_vecs) == 0:
        return [0.0] * len(facts), [-1] * len(facts)
    sims = facts @ sentence_vecs.T
    best = sims.argmax(axis=1)
    return [round(float(sims[i, j]), 4) for i, j in enumerate(best)], [int(j) for j in best]


def coverage(cosines: Sequence[float], threshold: float = COVERAGE_THRESHOLD) -> float:
    return sum(c >= threshold for c in cosines) / len(cosines) if cosines else 0.0


# --- scoring ------------------------------------------------------------------------------------------------


class ScoreRow(BaseModel):
    """One (article, candidate) result; an error row means the candidate produced no summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    candidate: str
    error: str | None = None
    words: int = 0
    sentences: int = 0
    numbers: int = 0
    unsupported_numbers: list[str] = []
    unsupported_in_input: int = 0
    entity_named: bool = False
    truncated: bool = False
    capped: bool = False
    fact_cosines: list[float] = []
    fact_best_sentence: list[int] = []

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def words_ok(self) -> bool:
        return self.ok and self.words <= MAX_WORDS

    @property
    def sentences_ok(self) -> bool:
        return self.ok and self.sentences <= MAX_SENTENCES

    @property
    def numbers_ok(self) -> bool:
        return self.ok and not self.unsupported_numbers

    @property
    def passed(self) -> bool:
        checks = (self.words_ok, self.sentences_ok, self.numbers_ok, self.entity_named, not self.truncated)
        return self.ok and all(checks)


def check_summary(
    article: EvalArticle,
    candidate: str,
    text: str,
    *,
    truncated: bool,
    capped: bool,
    gold: Grounding,
    served_input: Grounding,
    cosines: Sequence[float],
    best: Sequence[int],
) -> ScoreRow:
    missing = unsupported_numbers(text, gold)
    return ScoreRow(
        id=article.id,
        candidate=candidate,
        words=word_count(text),
        sentences=len(sentences(text)),
        numbers=len(summary_numbers(text)),
        unsupported_numbers=missing,
        unsupported_in_input=sum(is_grounded(value, served_input) for value in missing),
        entity_named=names_entity(text, entity_aliases(article.example.entity, article.example.ticker)),
        truncated=truncated,
        capped=capped,
        fact_cosines=list(cosines),
        fact_best_sentence=list(best),
    )


async def score_split(
    client: OllamaClient,
    articles: Sequence[EvalArticle],
    summaries: Mapping[Pair, Mapping[str, Mapping[str, Any]]],
    *,
    inputs: Callable[[EvalArticle, ModelSpec], str] = model_input,
    embed_model: str = EMBED_MODEL,
) -> list[ScoreRow]:
    """One embedding call per article: its key facts plus every candidate's summary sentences."""
    specs = {pair: GenRun.resolve(pair).spec for pair in summaries}
    rows: list[ScoreRow] = []
    for article in articles:
        facts = list(article.example.key_facts)
        texts = list(facts)
        pending: list[tuple[Pair, str, Mapping[str, Any], int, int]] = []
        for pair, records in summaries.items():
            record = records.get(article.id)
            text = summary_text(record)
            if record is None or text is None:
                code = str(record["error"]["code"]) if record and record.get("error") else "missing"
                rows.append(ScoreRow(id=article.id, candidate=pair.key, error=code))
                continue
            parts = sentences(text)
            pending.append((pair, text, record, len(texts), len(parts)))
            texts += parts
        if not pending:
            continue
        vectors = unit_rows(await client.embed(embed_model, texts))
        gold = grounding(article.text)
        served: dict[int, Grounding] = {}
        for pair, text, record, start, count in pending:
            cosines, best = best_matches(vectors[: len(facts)], vectors[start : start + count])
            budget = specs[pair].max_input_tokens
            if budget not in served:
                served[budget] = grounding(inputs(article, specs[pair]))
            rows.append(
                check_summary(
                    article,
                    pair.key,
                    text,
                    truncated=bool(record.get("truncated")),
                    capped=was_capped(record),
                    gold=gold,
                    served_input=served[budget],
                    cosines=cosines,
                    best=best,
                )
            )
    return rows


def read_scores(paths: SummaryPaths, split: str) -> list[ScoreRow]:
    path = paths.scores(split)
    if not path.exists():
        return []
    return [
        ScoreRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def write_scores(paths: SummaryPaths, split: str, rows: Iterable[ScoreRow], replaced: set[str]) -> None:
    kept = [r for r in read_scores(paths, split) if r.candidate not in replaced]
    ordered = sorted([*kept, *rows], key=lambda r: (r.candidate, r.id))
    write_text(paths.scores(split), "".join(r.model_dump_json() + "\n" for r in ordered))


# --- aggregation --------------------------------------------------------------------------------------------


def _rate(flags: Sequence[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def mean_ci(values: Mapping[str, float], strata: Mapping[str, str] | None = None) -> list[float] | None:
    """95% percentile bootstrap CI of a per-article mean, resampled within article label strata."""
    if len(values) < 2:
        return None
    ids = sorted(values)
    arr = np.asarray([values[i] for i in ids], dtype=np.float64)
    gold = [(strata or {}).get(i, "all") for i in ids]
    ci = bootstrap_ci(lambda idx: float(arr[idx].mean()), gold)
    return [round(ci.low, 3), round(ci.high, 3)]


def paired_gap(
    a: Mapping[str, float], b: Mapping[str, float], strata: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """Mean a - b over articles both candidates were judged on, with a paired bootstrap CI."""
    shared = sorted(a.keys() & b.keys())
    if len(shared) < 2:
        return None
    gaps = {i: a[i] - b[i] for i in shared}
    return {"mean": round(statistics.fmean(gaps.values()), 3), "ci": mean_ci(gaps, strata), "n": len(shared)}


def _coverage(rows: Sequence[ScoreRow], threshold: float) -> float:
    return round(statistics.fmean(coverage(r.fact_cosines, threshold) for r in rows), 4) if rows else 0.0


def aggregate(
    pair: Pair,
    rows: Sequence[ScoreRow],
    ids: Sequence[str],
    judged: Sequence[JudgeRow] = (),
    entry: ManifestEntry | None = None,
    strata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rates over all articles in `ids`: a missing or failed summary fails every check and covers nothing."""
    by_id = {r.id: r for r in rows if r.candidate == pair.key}
    padded = [by_id.get(i) or ScoreRow(id=i, candidate=pair.key, error="missing") for i in ids]
    produced = [r for r in padded if r.ok]
    wanted = set(ids)
    mine = [j for j in judged if j.candidate == pair.key and j.id in wanted]
    row: dict[str, Any] = {
        "endpoint": pair.endpoint,
        "model": pair.model,
        "n": len(ids),
        "deterministicPassRate": _rate([r.passed for r in padded]),
        "keyFactCoverage": _coverage(padded, COVERAGE_THRESHOLD),
        "keyFactCoverageStrict": _coverage(padded, STRICT_THRESHOLD),
        "judgeFaithfulness": _mean([j.faithfulness for j in mine]),
        "judgeCoverage": _mean([j.coverage for j in mine]),
        "hallucinationRate": _rate([j.hallucinations > 0 for j in mine]) if mine else None,
        "materialErrorRate": _rate([j.faithfulness <= MATERIAL_MAX_FAITHFULNESS for j in mine])
        if mine
        else None,
        "wordsOkRate": _rate([r.words_ok for r in padded]),
        "sentencesOkRate": _rate([r.sentences_ok for r in padded]),
        "numbersGroundedRate": _rate([r.numbers_ok for r in padded]),
        "entityNamedRate": _rate([r.entity_named for r in padded]),
        "truncatedRate": _rate([r.truncated for r in padded]),
        "cappedRate": _rate([r.capped for r in padded]),
        "errorRate": _rate([not r.ok for r in padded]),
        "medianWords": statistics.median(r.words for r in produced) if produced else None,
        "unsupportedNumbers": sum(len(r.unsupported_numbers) for r in produced),
        "unsupportedNumbersInInput": sum(r.unsupported_in_input for r in produced),
        "keyFactCoverageAt": {f"{t:.2f}": _coverage(padded, t) for t in SENSITIVITY},
        "judgeN": len(mine),
        "judgeLeadsRate": _rate([j.leads for j in mine]) if mine else None,
        "judgeFaithfulnessCi": mean_ci({j.id: j.faithfulness for j in mine}, strata),
        "judgeCoverageCi": mean_ci({j.id: j.coverage for j in mine}, strata),
        "hallucinationRateCi": mean_ci({j.id: float(j.hallucinations > 0) for j in mine}, strata),
        "materialErrorRateCi": mean_ci(
            {j.id: float(j.faithfulness <= MATERIAL_MAX_FAITHFULNESS) for j in mine}, strata
        ),
    }
    row["passesGate"] = row["deterministicPassRate"] >= PASS_GATE
    if entry is not None:
        row["promptVersion"] = entry.prompt_version
        row["modelDigest"] = entry.model_digest.removeprefix("sha256:")[:12]
    return row


def quality_ranking(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Per endpoint: gated candidates by judge faithfulness, then judge coverage; latency gates come later."""
    out: dict[str, list[str]] = {}
    for endpoint in SUMMARY_ENDPOINTS:
        gated = [r for r in rows if r["endpoint"] == endpoint and r["passesGate"]]
        gated.sort(key=lambda r: (-(r["judgeFaithfulness"] or 0), -(r["judgeCoverage"] or 0), r["model"]))
        out[endpoint] = [r["model"] for r in gated]
    return out


# --- spot check ---------------------------------------------------------------------------------------------


def spot_sample(rows: Sequence[ScoreRow], pairs: Sequence[Pair], n: int, seed: int = SEED) -> list[ScoreRow]:
    """n seeded articles, rotating the candidate so every model is looked at."""
    by_key = {(r.id, r.candidate): r for r in rows if r.ok and r.fact_cosines}
    ids = sorted({r.id for r in by_key.values()})
    picked = random.Random(seed).sample(ids, min(n, len(ids)))
    out = []
    for i, article in enumerate(sorted(picked)):
        for step in range(len(pairs)):
            if (row := by_key.get((article, pairs[(i + step) % len(pairs)].key))) is not None:
                out.append(row)
                break
    return out


def parse_labels(text: str, sample: Sequence[Mapping[str, Any]]) -> list[list[bool]]:
    """`1011,0111`: one group per sampled summary, one digit per key fact (1 = the summary states it)."""
    groups = [g.strip() for g in text.split(",")]
    if len(groups) != len(sample):
        raise SummaryEvalError(f"expected {len(sample)} label groups, got {len(groups)}")
    out = []
    for group, entry in zip(groups, sample, strict=True):
        if len(group) != len(entry["cosines"]) or set(group) - {"0", "1"}:
            raise SummaryEvalError(f"{entry['id']}: expected {len(entry['cosines'])} digits of 0/1")
        out.append([c == "1" for c in group])
    return out


def label_agreement(
    sample: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = AGREEMENT_THRESHOLDS
) -> dict[str, dict[str, int]]:
    """Per threshold: coverage flags against hand labels of whether each fact is really stated."""
    pairs = [(c, s) for entry in sample for c, s in zip(entry["cosines"], entry["stated"], strict=True)]
    out = {}
    for t in thresholds:
        covered_right = sum(c >= t and s for c, s in pairs)
        missed_right = sum(c < t and not s for c, s in pairs)
        out[f"{t:.2f}"] = {
            "right": covered_right + missed_right,
            "falseCovered": sum(c >= t and not s for c, s in pairs),
            "falseNotCovered": sum(c < t and s for c, s in pairs),
        }
    return out


def render_spot_review(
    sample: Sequence[ScoreRow], articles: Mapping[str, EvalArticle], texts: Mapping[tuple[str, str], str]
) -> str:
    lines = [
        "# Key-fact coverage spot check (private)",
        "",
        f"Covered when the best cosine is >= {COVERAGE_THRESHOLD}. For each fact, judge whether the summary",
        "really states it, then compare with the flag. Do not copy text from this file anywhere.",
        "",
    ]
    for row in sample:
        parts = sentences(texts[(row.id, row.candidate)])
        lines += [f"## {row.id} ({row.candidate})", "", "Summary sentences:", ""]
        lines += [f"- [{i}] {s}" for i, s in enumerate(parts)]
        lines += ["", "Facts:", ""]
        facts = articles[row.id].example.key_facts
        for i, (fact, cos, best) in enumerate(
            zip(facts, row.fact_cosines, row.fact_best_sentence, strict=True)
        ):
            flag = "covered" if cos >= COVERAGE_THRESHOLD else "not covered"
            lines.append(f"- F{i} cos={cos:.3f} best=[{best}] {flag}: {fact}")
        lines.append("")
    return "\n".join(lines)


# --- judge packets ------------------------------------------------------------------------------------------


def candidate_summaries(
    summaries: Mapping[Pair, Mapping[str, Mapping[str, Any]]], article: str
) -> dict[str, str]:
    out = {}
    for pair, records in summaries.items():
        if (text := summary_text(records.get(article))) is not None:
            out[pair.key] = text
    return out


def prepare_packets(
    split: str,
    articles: Sequence[EvalArticle],
    summaries: Mapping[Pair, Mapping[str, Mapping[str, Any]]],
    judge: JudgePaths,
) -> list[Path]:
    written = []
    for article in articles:
        texts = candidate_summaries(summaries, article.id)
        if not texts:
            continue
        blinding = make_blinding(article.id, split, texts)
        packet = render_packet(
            blinding, article.title, article.example.gold_text, texts, judge.out(article.id)
        )
        write_text(judge.packet(article.id), packet)
        write_json(judge.blinding(article.id), blinding.model_dump())
        written.append(judge.packet(article.id))
    write_text(judge.packet_list(split), "".join(f"{rel(p)}\n" for p in written))
    return written


def check_judgements(
    articles: Sequence[EvalArticle],
    summaries: Mapping[Pair, Mapping[str, Mapping[str, Any]]],
    judge: JudgePaths,
) -> list[tuple[str, tuple[str, ...], list[JudgeRow]]]:
    """Per packet: (id, failure codes, de-blinded rows when valid)."""
    results = []
    for article in articles:
        raw_blinding = read_json(judge.blinding(article.id))
        if raw_blinding is None:
            continue
        blinding = Blinding.model_validate(raw_blinding)
        out = judge.out(article.id)
        raw = out.read_text(encoding="utf-8") if out.exists() else None
        output, codes = check_output(
            raw,
            blinding,
            article_text=article.text,
            summaries=candidate_summaries(summaries, article.id),
        )
        rows = deblind(output, blinding) if output is not None and not codes else []
        results.append((article.id, codes, rows))
    return results


# --- report -------------------------------------------------------------------------------------------------


def read_judged(judge: JudgePaths, split: str) -> list[JudgeRow]:
    path = judge.ingested(split)
    if not path.exists():
        return []
    return [
        JudgeRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def _fmt(value: Any, kind: str = "pct") -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "num":
        return f"{value:.2f}"
    return str(value)


INTRO = f"""# Summary evaluation

Decision R6: deterministic checks, key-fact coverage by a local embedding model, and Claude Opus as a
blind judge. Summaries come from the serving adapters (`/api/summary`, `/api/analyze`) with the default
`{PARSER}` + `{EXTRACTOR}` extraction and `{SCOPE}` scope, run on the stored HTML of the Opus article set.
Written by `uv run python -m veritrade_evals.summary_eval report`; no article or summary text is committed.
Every check scores the served summary: serving caps it at {MAX_SENTENCES} sentences and {MAX_WORDS} words
(`cap_summary` in `veritrade.inference.adapters`), and `score` applies the same cap to cached generations.

- **Halluc.** (`hallucinationRate`): share of summaries with any unsupported detail, minor slips included.
- **Material** (`materialErrorRate`): share at faithfulness <= {MATERIAL_MAX_FAITHFULNESS}: a material error.
"""
DEFINITIONS = f"""## Definitions

- **Pass**: <= {MAX_WORDS} words, <= {MAX_SENTENCES} sentences (serving sentence splitter), every number
  grounded, entity named, not truncated. Rates are over all articles; a candidate that produced no summary
  fails every check. Length holds by construction for served summaries; **Capped** is the share the cap
  had to shorten (whole leading sentences kept; an overlong first sentence ends in an ellipsis).
- **Numbers**: every number in the summary appears in the gold title or body. Thousands separators, `%`,
  currency symbols and units are ignored and 1.50 equals 1.5; number words up to twenty in the article
  ground their digits; Q1-Q4 / H1-H2 match the same token, `third quarter` / `first half`, `September
  quarter` or `July to September` (calendar quarters); `FY27` matches 2027; `80K` matches 80,000 (K/M/B
  abbreviations); a clock time on the hour (`5:00 p.m.`) matches the bare hour (`5PM`). Derived figures
  (a percentage change the model computed) count as unsupported: the prompt asks for verbatim numbers.
- **Entity**: the gold entity appears (case-insensitive, whole words); or its short name, without legal
  suffixes (Inc, Corp, Ltd, plc, Holdings, Group, ...) and trailing descriptors (Entertainment,
  Technologies, Industries, ...), as a phrase or as all its words anywhere (`UK ... economy`); or its first
  word when the name has several words and that word has 4+ letters and is not generic (American, United,
  Bank, ...); or the gold ticker (case-sensitive, also without an exchange suffix such as `.HK`).
- **Truncated**: the output hit `num_predict` and the serving path cut it to its last full sentence.
- **Key facts**: each of the 3-5 Opus key facts and each summary sentence is embedded with
  `{EMBED_MODEL}`; a fact is covered when its best cosine is >= {COVERAGE_THRESHOLD:.2f}, the threshold
  that agreed best with hand labels in the spot check. Mean share per article. **Key facts (strict)** uses
  {STRICT_THRESHOLD:.2f}, where every flagged fact was really stated: a conservative lower bound.
- **Faithful / Cover**: Opus judge scores 1-5 (`evals/opus_prompts/50-judge.md`), candidates anonymised and
  shuffled per article. **Halluc.** and **Material** are defined at the top: a minor slip such as a
  loosely worded detail counts towards the first only. **Leads**: share the judge found leading with the
  market-moving fact.
"""
TABLE_HEAD = (
    "| Endpoint | Model | Pass | <=60 words | <=3 sent. | Numbers | Entity | Truncated | Capped | Errors "
    "| Median words | Key facts | Key facts (strict) | Faithful | Cover | Halluc. | Material | Leads |\n"
    "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
)
TABLE_COLUMNS = (
    ("deterministicPassRate", "pct"),
    ("wordsOkRate", "pct"),
    ("sentencesOkRate", "pct"),
    ("numbersGroundedRate", "pct"),
    ("entityNamedRate", "pct"),
    ("truncatedRate", "pct"),
    ("cappedRate", "pct"),
    ("errorRate", "pct"),
    ("medianWords", "raw"),
    ("keyFactCoverage", "pct"),
    ("keyFactCoverageStrict", "pct"),
    ("judgeFaithfulness", "num"),
    ("judgeCoverage", "num"),
    ("hallucinationRate", "pct"),
    ("materialErrorRate", "pct"),
    ("judgeLeadsRate", "pct"),
)


def _spot_lines(spot: Mapping[str, Any] | None) -> list[str]:
    if spot is None:
        return ["- not run"]
    head = f"- {spot['articles']} summaries, {spot['facts']} key facts"
    agreement = spot.get("agreement")
    if not agreement:
        return [head, "- hand labels pending"]
    lines = [
        f"{head}, {spot['stated']} of them stated in the summary by hand label (read in the private files)"
    ]
    for threshold, counts in agreement.items():
        lines.append(
            f"- at {threshold}: {counts['right']} of {spot['facts']} flags right, "
            f"{counts['falseCovered']} wrongly covered, {counts['falseNotCovered']} wrongly not covered"
        )
    return [*lines, f"- {spot['note']}"] if spot.get("note") else lines


def _ci(value: float | None, ci: Sequence[float] | None, kind: str = "num") -> str:
    return f"{_fmt(value, kind)} [{_fmt(ci[0], kind)}, {_fmt(ci[1], kind)}]" if ci else _fmt(value, kind)


def _judge_lines(rows: Sequence[Mapping[str, Any]], gaps: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["", "Judge means with 95% bootstrap CIs (articles resampled within their sentiment label):", ""]
    for r in rows:
        faithful = _ci(r["judgeFaithfulness"], r["judgeFaithfulnessCi"])
        lines.append(
            f"- {r['endpoint']} `{r['model']}`: faithfulness {faithful}"
            f", coverage {_ci(r['judgeCoverage'], r['judgeCoverageCi'])}, hallucination rate "
            f"{_ci(r['hallucinationRate'], r['hallucinationRateCi'], 'pct')}, material error rate "
            f"{_ci(r['materialErrorRate'], r['materialErrorRateCi'], 'pct')} (n={r['judgeN']})"
        )
    if gaps:
        lines += ["", "Leader minus runner-up, paired by article:", ""]
    for endpoint, gap in gaps.items():
        parts = [
            f"{name} {_ci(gap[name]['mean'], gap[name]['ci'])}"
            for name in ("faithfulness", "coverage")
            if gap.get(name)
        ]
        lines.append(f"- /{endpoint}: `{gap['leader']}` vs `{gap['runnerUp']}`: {', '.join(parts)}")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [INTRO]
    for split, block in report["splits"].items():
        rows = block["rows"]
        lines += [f"## {split} split (n={block['n']} articles)", "", TABLE_HEAD]
        for r in rows:
            cells = " | ".join(_fmt(r[name], kind) for name, kind in TABLE_COLUMNS)
            lines.append(f"| {r['endpoint']} | `{r['model']}` | {cells} |")
        lines += [
            "",
            "Unsupported numbers (distinct per summary, summed) and how many of them the model input had:",
            "",
        ]
        lines += [
            f"- {r['endpoint']} `{r['model']}`: {r['unsupportedNumbers']} unsupported, "
            f"{r['unsupportedNumbersInInput']} present in the extracted input but not in the gold article"
            for r in rows
        ]
        lines += ["", f"Key-fact coverage at other thresholds (main threshold {COVERAGE_THRESHOLD:.2f}):", ""]
        for r in rows:
            at = ", ".join(f"{t}: {_fmt(v)}" for t, v in r["keyFactCoverageAt"].items())
            lines.append(f"- {r['endpoint']} `{r['model']}`: {at}")
        lines += [
            "",
            f"Quality order (deterministic pass rate >= {PASS_GATE}, then judge faithfulness, then judge",
            "coverage; latency budgets are applied later by `select`):",
            "",
        ]
        pending = "" if block["judged"] else " (judge pending: this order is not meaningful yet)"
        for endpoint, models in block["qualityRanking"].items():
            ranked = ", ".join(f"`{m}`" for m in models) or "no candidate passes the gate"
            lines.append(f"- /{endpoint}: {ranked}{pending}")
        if block["judged"]:
            lines += _judge_lines(rows, block.get("leaderGaps", {}))
        lines += ["", "Coverage threshold spot check:", "", *_spot_lines(block.get("spotCheck")), ""]
    lines.append(DEFINITIONS)
    return "\n".join(lines)


def leader_gaps(
    ranking: Mapping[str, Sequence[str]], judged: Sequence[JudgeRow], strata: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Per endpoint: the ranked leader minus the runner-up on judge faithfulness and coverage, paired."""
    out = {}
    for endpoint, models in ranking.items():
        if len(models) < 2:
            continue
        leader, runner_up = (f"{endpoint}/{m}" for m in models[:2])
        by = {key: [j for j in judged if j.candidate == key] for key in (leader, runner_up)}
        out[endpoint] = {
            "leader": models[0],
            "runnerUp": models[1],
            **{
                name: paired_gap(
                    {j.id: getattr(j, name) for j in by[leader]},
                    {j.id: getattr(j, name) for j in by[runner_up]},
                    strata,
                )
                for name in ("faithfulness", "coverage")
            },
        }
    return out


def build_report(
    split: str,
    pairs: Sequence[Pair],
    ids: Sequence[str],
    paths: SummaryPaths,
    strata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rows = read_scores(paths, split)
    judged = read_judged(paths.judge, split)
    manifest = read_manifest(paths).get(split, {})
    table = [aggregate(pair, rows, ids, judged, manifest.get(pair.key), strata) for pair in pairs]
    ranking = quality_ranking(table)
    spot = read_json(paths.spot_check(split))
    spot_block = None
    if spot is not None:
        labelled = "agreement" in spot
        shown = {f"{t:.2f}" for t in (COVERAGE_THRESHOLD, STRICT_THRESHOLD)}
        spot_block = {
            "articles": len(spot["sample"]),
            "facts": sum(len(s["cosines"]) for s in spot["sample"]),
            "stated": sum(sum(s["stated"]) for s in spot["sample"]) if labelled else None,
            "agreement": {t: c for t, c in spot["agreement"].items() if t in shown} if labelled else None,
            "note": spot.get("note"),
        }
    return {
        "n": len(ids),
        "articleIds": sorted(ids),
        "rows": table,
        "judged": bool(judged),
        "qualityRanking": ranking,
        "leaderGaps": leader_gaps(ranking, judged, strata) if judged else {},
        "spotCheck": spot_block,
    }


# --- CLI ----------------------------------------------------------------------------------------------------


def _client(base_url: str | None) -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        base_url or settings.ollama_base_url, timeout=TIMEOUT_S, keep_alive=settings.ollama_keep_alive
    )


def _gen_pairs(models: Sequence[str] | None, endpoint: SummaryEndpoint | None) -> list[Pair]:
    if not models:
        return [p for p in PAIRS if endpoint is None or p.endpoint == endpoint]
    pairs = []
    for model in models:
        spec = allowlist().get(model)
        if spec is None:
            raise SummaryEvalError(f"{model} is not in models.toml")
        served = [e for e in SUMMARY_ENDPOINTS if spec.serves(e) and (endpoint is None or e == endpoint)]
        if not served:
            raise SummaryEvalError(f"{model} serves none of the requested endpoints")
        pairs += [Pair(e, model) for e in served]
    return pairs


async def _generate(args: argparse.Namespace, paths: SummaryPaths) -> int:
    articles = load_eval_articles(args.split, ArticlePaths())
    if args.limit:
        articles = articles[: args.limit]
    pairs = _gen_pairs(args.model, args.endpoint)
    async with _client(args.base_url) as client:
        for pair in pairs:
            run = GenRun.resolve(pair)
            entry = await generate(
                client,
                run,
                articles,
                split=args.split,
                digest=await model_digest(client, pair.model),
                paths=paths,
            )
            print(
                f"{pair.key}: n={entry.n} done={entry.done} errors={entry.errors} "
                f"truncated={entry.truncated} prompt={entry.prompt_version}"
            )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    return asyncio.run(_generate(args, SummaryPaths()))


async def _score(args: argparse.Namespace, paths: SummaryPaths) -> int:
    articles = load_eval_articles(args.split, ArticlePaths())
    summaries = {pair: load_summaries(paths, args.split, pair) for pair in args.models}
    async with _client(args.base_url) as client:
        rows = await score_split(client, articles, summaries)
    write_scores(paths, args.split, rows, {p.key for p in args.models})
    print(f"wrote {len(rows)} rows to {rel(paths.scores(args.split))}")
    for pair in args.models:
        row = aggregate(pair, rows, [a.id for a in articles])
        print(
            f"{pair.key}: pass={row['deterministicPassRate']:.3f} words={row['wordsOkRate']:.3f} "
            f"sentences={row['sentencesOkRate']:.3f} numbers={row['numbersGroundedRate']:.3f} "
            f"entity={row['entityNamedRate']:.3f} truncated={row['truncatedRate']:.3f} "
            f"errors={row['errorRate']:.3f} median_words={row['medianWords']} "
            f"coverage={row['keyFactCoverage']:.3f}"
        )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    return asyncio.run(_score(args, SummaryPaths()))


def _record_labels(committed: Path, labels: str, note: str | None) -> None:
    spot = read_json(committed)
    if spot is None:
        raise SummaryEvalError("run spot-check first, then record --labels")
    stated = parse_labels(labels, spot["sample"])
    for entry, flags in zip(spot["sample"], stated, strict=True):
        entry["stated"] = flags
    spot["agreement"] = label_agreement(spot["sample"])
    spot["note"] = note
    write_json(committed, spot)
    for threshold, counts in spot["agreement"].items():
        print(f"t={threshold} " + " ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"recorded labels in {rel(committed)}")


def cmd_spot_check(args: argparse.Namespace) -> int:
    paths = SummaryPaths()
    committed = paths.spot_check(args.split)
    if args.labels:
        _record_labels(committed, args.labels, args.note)
        return 0
    existing = read_json(committed)
    if existing is not None and "agreement" in existing and not args.force:
        raise SummaryEvalError(f"{rel(committed)} holds hand labels; pass --force to draw a new sample")
    rows = read_scores(paths, args.split)
    articles = {a.id: a for a in load_eval_articles(args.split, ArticlePaths())}
    sample = spot_sample(rows, args.models, args.n)
    texts = {}
    for pair in args.models:
        for article, record in load_summaries(paths, args.split, pair).items():
            if (text := summary_text(record)) is not None:
                texts[(article, pair.key)] = text
    write_text(paths.spot_review(args.split), render_spot_review(sample, articles, texts))
    payload = {
        "split": args.split,
        "embeddingModel": EMBED_MODEL,
        "threshold": COVERAGE_THRESHOLD,
        "sample": [{"id": r.id, "candidate": r.candidate, "cosines": r.fact_cosines} for r in sample],
    }
    write_json(committed, payload)
    for r in sample:
        flags = " ".join(
            f"F{i}={c:.3f}{'+' if c >= COVERAGE_THRESHOLD else '-'}" for i, c in enumerate(r.fact_cosines)
        )
        print(f"{r.id} {r.candidate} {flags}")
    print(f"review file (Read tool only): {rel(paths.spot_review(args.split))}")
    return 0


def cmd_judge_prepare(args: argparse.Namespace) -> int:
    paths = SummaryPaths()
    articles = load_eval_articles(args.split, ArticlePaths())
    summaries = {pair: load_summaries(paths, args.split, pair) for pair in args.models}
    written = prepare_packets(args.split, articles, summaries, paths.judge)
    counts = [len(candidate_summaries(summaries, a.id)) for a in articles]
    print(
        f"{len(written)} packets in {rel(paths.judge.root / 'packets')}; "
        f"candidates per packet: min={min(counts, default=0)} max={max(counts, default=0)}"
    )
    print(f"packet list: {rel(paths.judge.packet_list(args.split))}")
    return 0


def _judge_results(
    args: argparse.Namespace, paths: SummaryPaths
) -> list[tuple[str, tuple[str, ...], list[JudgeRow]]]:
    articles = load_eval_articles(args.split, ArticlePaths())
    if getattr(args, "id", None):
        articles = [a for a in articles if a.id in set(args.id)]
    summaries = {pair: load_summaries(paths, args.split, pair) for pair in args.models}
    return check_judgements(articles, summaries, paths.judge)


def _print_checks(results: Sequence[tuple[str, tuple[str, ...], list[JudgeRow]]]) -> int:
    failed = 0
    for article, codes, _ in results:
        if codes:
            failed += 1
            print(f"FAIL {article} {' '.join(codes)}")
        else:
            print(f"ok {article}")
    print(f"{len(results) - failed} valid, {failed} failing or missing")
    return failed


def cmd_judge_check(args: argparse.Namespace) -> int:
    """Validation only: never reveals which candidate is behind a label."""
    return 1 if _print_checks(_judge_results(args, SummaryPaths())) else 0


def cmd_judge_ingest(args: argparse.Namespace) -> int:
    paths = SummaryPaths()
    results = _judge_results(args, paths)
    failed = _print_checks(results)
    rows = [row for _, codes, judged in results if not codes for row in judged]
    write_text(paths.judge.ingested(args.split), "".join(r.model_dump_json() + "\n" for r in rows))
    print(f"ingested {len(rows)} judged summaries into {rel(paths.judge.ingested(args.split))}")
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    paths = SummaryPaths()
    articles = load_eval_articles(args.split, ArticlePaths())
    existing = read_json(paths.summary_json) or {}
    splits = dict(existing.get("splits", {}))
    strata = {a.id: a.example.label for a in articles}
    splits[args.split] = build_report(args.split, args.models, [a.id for a in articles], paths, strata)
    report = {
        "version": 1,
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "embeddingModel": EMBED_MODEL,
            "coverageThreshold": COVERAGE_THRESHOLD,
            "strictCoverageThreshold": STRICT_THRESHOLD,
            "servedCap": "cap_summary in veritrade.inference.adapters, applied to cached generations",
            "maxWords": MAX_WORDS,
            "maxSentences": MAX_SENTENCES,
            "extraction": {"parser": PARSER, "extractor": EXTRACTOR, "scope": SCOPE},
            "passGate": PASS_GATE,
            "selection": "judge faithfulness, then judge coverage, among candidates with "
            "deterministicPassRate >= passGate; latency budgets are applied by `select`",
        },
        "splits": {s: splits[s] for s in sorted(splits)},
    }
    write_json(paths.summary_json, report)
    write_text(paths.summary_md, render_markdown(report))
    print(f"wrote {rel(paths.summary_json)} and {rel(paths.summary_md)}")
    return 0


def _models_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_pair,
        default=list(PAIRS),
        metavar="ENDPOINT/MODEL",
        help="candidates, e.g. summary/qwen3.5:2b-q4_K_M (default: the five R6 candidates)",
    )


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `summary` on a parent CLI; dispatch with `run(args)`."""
    summary: argparse.ArgumentParser = subparsers.add_parser("summary", help="summary evaluation (R6)")
    actions = summary.add_subparsers(dest="summary_action", required=True)

    gen = actions.add_parser("generate", help="summaries via the serving path (resumable)")
    gen.add_argument("--split", choices=SPLITS, required=True)
    gen.add_argument("--model", action="append", help="model (repeatable; default: the five R6 candidates)")
    gen.add_argument("--endpoint", choices=SUMMARY_ENDPOINTS, default=None)
    gen.add_argument("--limit", type=int, default=0, help="first N articles only (smoke runs)")
    gen.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    gen.set_defaults(summary_func=cmd_generate)

    score = actions.add_parser("score", help="deterministic checks and key-fact coverage")
    score.add_argument("--split", choices=SPLITS, required=True)
    _models_arg(score)
    score.add_argument("--base-url", default=None)
    score.set_defaults(summary_func=cmd_score)

    spot = actions.add_parser("spot-check", help="coverage threshold spot check (ids and cosines only)")
    spot.add_argument("--split", choices=SPLITS, default="dev")
    spot.add_argument("--n", type=int, default=SPOT_CHECK_N)
    _models_arg(spot)
    spot.add_argument(
        "--labels", default=None, help="record hand labels, e.g. 1011,0111: per summary, 1 per stated fact"
    )
    spot.add_argument("--note", default=None, help="one line for the results, no article text")
    spot.add_argument("--force", action="store_true", help="replace a hand-labelled sample")
    spot.set_defaults(summary_func=cmd_spot_check)

    prepare = actions.add_parser("judge-prepare", help="blind judge packets per article")
    prepare.add_argument("--split", choices=SPLITS, required=True)
    _models_arg(prepare)
    prepare.set_defaults(summary_func=cmd_judge_prepare)

    check = actions.add_parser("judge-check", help="validate judge outputs without de-blinding")
    check.add_argument("--split", choices=SPLITS, required=True)
    check.add_argument("--id", action="append", help="only this article (repeatable)")
    _models_arg(check)
    check.set_defaults(summary_func=cmd_judge_check)

    ingest = actions.add_parser("judge-ingest", help="validate and de-blind judge outputs")
    ingest.add_argument("--split", choices=SPLITS, required=True)
    _models_arg(ingest)
    ingest.set_defaults(summary_func=cmd_judge_ingest)

    report = actions.add_parser("report", help="write evals/results/summary/summary.{json,md}")
    report.add_argument("--split", choices=SPLITS, required=True)
    _models_arg(report)
    report.set_defaults(summary_func=cmd_report)
    return summary


def run(args: argparse.Namespace) -> int:
    try:
        code: int = args.summary_func(args)
    except SummaryEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.summary_eval", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["summary", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
