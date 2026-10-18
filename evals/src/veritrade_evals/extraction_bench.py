"""Parser x extractor matrix: `uv run python -m veritrade_evals.extraction_bench <cmd>`.

`measure` times and scores every row on the stored HTML (CPU only), `signal` runs the /signal decision model
on each row's lede and on the gold lede, `report` writes evals/results/extraction. Article text stays in
memory and under evals/data/private; outputs carry ids, sources and numbers only.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

import httpx
import numpy as np
from packaging.requirements import Requirement

from veritrade.config import get_settings
from veritrade.errors import LlmUnreachable, ModelUnavailable, SchemaViolation, VtError
from veritrade.extract import registry
from veritrade.extract.dom import normalize
from veritrade.extract.scope import build_input, estimate_tokens
from veritrade.extract.types import (
    EXTRACTORS,
    PARSER_AWARE_EXTRACTORS,
    PARSERS,
    Article,
    ExtractorName,
    ParserName,
)
from veritrade.inference.confidence import ProbabilityPolicy
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Sentiment
from veritrade.models_registry import allowlist
from veritrade_evals.analysis import percentile, rounded
from veritrade_evals.articles import ArticleExample, ArticlePaths, load_articles, read_fetched, rel
from veritrade_evals.articles_validate import gold_check
from veritrade_evals.grid import GRID_DIR, grid_cells
from veritrade_evals.paths import PREDS_DIR, PRIVATE_DIR, RESULTS_DIR
from veritrade_evals.predcache import PredCache, PredError, model_slug
from veritrade_evals.runner import RunSpec, call_once, model_digest

OUT_DIR = RESULTS_DIR / "extraction"
PRIVATE_BENCH_DIR = PRIVATE_DIR / "extraction"
MEASURE_PATH = PRIVATE_BENCH_DIR / "measure.jsonl"
SESSIONS_SUFFIX = ".sessions.json"
SIGNAL_MANIFEST = PRIVATE_BENCH_DIR / "signal.json"

# Part of the signal cache key: bump when the lede construction or record shape changes.
BENCH_VERSION = "1"
SIGNAL_MODEL = "clef-flash:9b-q8_0"
SIGNAL_SPLIT = "articles"
GOLD_ROW = "gold"
DEFAULT_RUNS = 5
SHINGLE = 5
# build_input needs a positive budget; this one never cuts, so the full size is measured uncapped.
UNCAPPED_TOKENS = 10**9
N_BOOT = 1000
SEED = 20261018

OWN_EXTRACTORS: tuple[ExtractorName, ...] = tuple(e for e in EXTRACTORS if e in PARSER_AWARE_EXTRACTORS)
THIRD_PARTY: tuple[ExtractorName, ...] = tuple(e for e in EXTRACTORS if e not in PARSER_AWARE_EXTRACTORS)
DEFAULT_PARSER: ParserName = "selectolax"
SERVED_ROW = f"{DEFAULT_PARSER}/cascade"

# Distributions each row needs on top of the app core; strip-tags walks the selectolax tree.
PARSER_DISTS: dict[ParserName, tuple[str, ...]] = {
    "selectolax": ("selectolax",),
    "lxml": ("lxml", "cssselect"),
    "bs4": ("beautifulsoup4",),
    "bs4-lxml": ("beautifulsoup4", "lxml"),
}
EXTRACTOR_DISTS: dict[ExtractorName, tuple[str, ...]] = {
    "trafilatura": ("trafilatura",),
    "trafilatura-precision": ("trafilatura",),
    "readability": ("readability-lxml", "lxml-html-clean"),
    "newspaper": ("newspaper4k",),
    "justext": ("justext",),
    "strip-tags": ("selectolax",),
}
BACKEND_DIST = "veritrade-backend"

Record = dict[str, Any]
Timing = dict[str, float]
Caller = Callable[[str], Awaitable[tuple[dict[str, Any], Timing]]]


class BenchError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- rows ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    parser: ParserName | None
    extractor: ExtractorName

    @property
    def name(self) -> str:
        return f"{self.parser}/{self.extractor}" if self.parser else self.extractor


def matrix_rows() -> list[Row]:
    """Own strategies on every parser, then the third-party extractors that bring their own parser."""
    own = [Row(p, e) for p in PARSERS for e in OWN_EXTRACTORS]
    return own + [Row(None, e) for e in THIRD_PARTY]


def select_rows(names: Sequence[str] | None) -> list[Row]:
    rows = matrix_rows()
    if not names:
        return rows
    known = {row.name: row for row in rows}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise BenchError(f"unknown rows {unknown}; valid: {', '.join(known)}")
    return [known[n] for n in names]


# --- quality metrics -------------------------------------------------------------------------

_TOKEN = re.compile(r"\w+")


def word_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Quality:
    extracted: int
    gold: int
    common: int
    leaked: int

    @property
    def precision(self) -> float:
        return self.common / self.extracted if self.extracted else 0.0

    @property
    def recall(self) -> float:
        return self.common / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def leakage(self) -> float:
        return self.leaked / self.extracted if self.extracted else 0.0


def unsupported_tokens(extracted: Sequence[str], gold: Sequence[str], k: int = SHINGLE) -> int:
    """Extracted tokens outside every k-token window that also occurs in the gold token sequence."""
    n = len(extracted)
    if n == 0:
        return 0
    k = min(k, n)
    windows = {tuple(gold[i : i + k]) for i in range(len(gold) - k + 1)}
    covered = [False] * n
    for j in range(n - k + 1):
        if tuple(extracted[j : j + k]) in windows:
            covered[j : j + k] = [True] * k
    return n - sum(covered)


def quality(extracted_text: str, gold_text: str, k: int = SHINGLE) -> Quality:
    ext, gold = word_tokens(extracted_text), word_tokens(gold_text)
    common = sum((Counter(ext) & Counter(gold)).values())
    return Quality(len(ext), len(gold), common, unsupported_tokens(ext, gold, k))


# --- pages and extraction --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Page:
    """One article as the bench sees it; html and gold text are only ever held in memory."""

    id: str
    source: str
    split: str
    url: str
    html: str
    gold_text: str
    gold_title: str
    label: str = ""

    @property
    def gold_article(self) -> Article:
        # extractor is a required field; build_input reads only title, paragraphs and source.
        paragraphs = tuple(p for p in self.gold_text.split("\n\n") if p.strip())
        return Article(
            url=self.url,
            title=self.gold_title,
            paragraphs=paragraphs,
            published_at=None,
            source=registry.source_for_url(self.url),
            method=GOLD_ROW,
            parser=None,
            extractor="cascade",
            parse_ms=0.0,
            extract_ms=0.0,
        )


def page_for(example: ArticleExample, paths: ArticlePaths) -> Page:
    """The serving extractor receives the final URL, so the bench does too."""
    fetched = read_fetched(paths, example.id)
    check = gold_check(paths, example.id)
    if check.title is None or check.body != example.gold_text:
        raise BenchError(
            f"{example.id}: gold output does not match the committed gold body; run `articles verify`"
        )
    return Page(
        id=example.id,
        source=example.source,
        split=example.split,
        url=(fetched.final_url if fetched and fetched.final_url else example.url),
        html=example.html_path.read_text(encoding="utf-8"),
        gold_text=example.gold_text,
        gold_title=check.title,
        label=example.label,
    )


def iter_pages(limit: int = 0, paths: ArticlePaths | None = None) -> Iterator[Page]:
    paths = paths or ArticlePaths()
    examples = load_articles("all", paths=paths)
    for example in examples[:limit] if limit else examples:
        yield page_for(example, paths)


@dataclass(frozen=True, slots=True)
class Outcome:
    article: Article | None
    error: str | None

    @property
    def signature(self) -> tuple[str | None, tuple[str, ...]]:
        return self.error, self.article.paragraphs if self.article else ()


def extract_row(row: Row, page: Page) -> Outcome:
    """The serving entry point; any exception is a result of the row, recorded by type name only."""
    try:
        article = registry.extract(
            page.html, page.url, parser=row.parser or DEFAULT_PARSER, extractor=row.extractor
        )
    except Exception as exc:
        return Outcome(None, type(exc).__name__)
    return Outcome(article, None)


def same_title(a: str, b: str) -> bool:
    return normalize(a).casefold() == normalize(b).casefold()


def signal_text(article: Article, title: str, max_tokens: int) -> str:
    """The /signal lede with the title held at the gold headline, so only the body varies across rows."""
    return build_input(replace(article, title=title), "lede", max_tokens)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def signal_budget() -> int:
    return allowlist()[SIGNAL_MODEL].max_input_tokens


# --- measure ---------------------------------------------------------------------------------


def _ms(value: float) -> float:
    return round(value, 3)


def measure_row(row: Row, page: Page, *, runs: int, lede_budget: int) -> Record:
    """One warm-up (the scored output), then `runs` timed calls; their median is the article's time."""
    gc.collect()
    first = extract_row(row, page)
    walls: list[float] = []
    parses: list[float] = []
    stable = True
    for _ in range(runs):
        t0 = time.perf_counter()
        again = extract_row(row, page)
        walls.append((time.perf_counter() - t0) * 1000)
        # Third-party extractors parse internally, so only own strategies report a separate parse time.
        if again.article is not None and again.article.parser is not None:
            parses.append(again.article.parse_ms)
        stable = stable and again.signature == first.signature
    record: Record = {
        "row": row.name,
        "id": page.id,
        "source": page.source,
        "split": page.split,
        "error": first.error,
        "ms": _ms(statistics.median(walls)) if walls else None,
        "ms_runs": [_ms(w) for w in walls],
        "parse_ms": _ms(statistics.median(parses)) if parses else None,
        "stable": stable,
        "measured_at": _now(),
    }
    if first.article is not None:
        article = first.article
        q = quality(article.text, page.gold_text)
        record |= {
            "method": article.method,
            "words": article.word_count,
            "tokens_extracted": q.extracted,
            "tokens_gold": q.gold,
            "tokens_common": q.common,
            "tokens_leaked": q.leaked,
            "full_tokens": estimate_tokens(build_input(article, "full", UNCAPPED_TOKENS)),
            "lede_tokens": estimate_tokens(build_input(article, "lede", lede_budget)),
            "title_matches_gold": same_title(article.title, page.gold_title),
        }
    return record


def gold_record(page: Page, *, lede_budget: int) -> Record:
    gold = page.gold_article
    return {
        "row": GOLD_ROW,
        "id": page.id,
        "source": page.source,
        "split": page.split,
        "error": None,
        "tokens_gold": len(word_tokens(page.gold_text)),
        "full_tokens": estimate_tokens(build_input(gold, "full", UNCAPPED_TOKENS)),
        "lede_tokens": estimate_tokens(build_input(gold, "lede", lede_budget)),
        "measured_at": _now(),
    }


def read_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def latest_by_key(records: Iterable[Record]) -> dict[tuple[str, str], Record]:
    return {(r["row"], r["id"]): r for r in records}


def _sysctl(key: str) -> str | None:
    try:
        done = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False, timeout=5)
    except OSError:
        return None
    return done.stdout.strip() or None


def machine() -> dict[str, Any]:
    memsize = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
        "cpus": os.cpu_count(),
        "ramGb": round(int(memsize) / 2**30) if memsize else None,
        "os": platform.platform(terse=True),
        "python": platform.python_version(),
    }


def ollama_resident(base_url: str) -> list[str] | None:
    """Model names Ollama has loaded right now; None when it does not answer."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/ps", timeout=2.0)
        response.raise_for_status()
        return sorted(str(m.get("name") or m.get("model")) for m in response.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return None


@dataclass(slots=True)
class Session:
    """Machine load while timing: 1-minute load average and Ollama residency, sampled once per article."""

    started_at: str = field(default_factory=_now)
    loadavg: list[float] = field(default_factory=list)
    ollama_samples: int = 0
    ollama_busy: int = 0
    models: set[str] = field(default_factory=set)
    records: int = 0

    def sample(self, base_url: str) -> None:
        self.loadavg.append(os.getloadavg()[0])
        resident = ollama_resident(base_url)
        if resident is not None:
            self.ollama_samples += 1
            self.ollama_busy += bool(resident)
            self.models.update(resident)

    def as_json(self) -> Record:
        return {
            "started_at": self.started_at,
            "finished_at": _now(),
            "records": self.records,
            "loadavg_1m_mean": round(statistics.fmean(self.loadavg), 2) if self.loadavg else None,
            "loadavg_1m_max": round(max(self.loadavg), 2) if self.loadavg else None,
            "ollama_samples": self.ollama_samples,
            "ollama_model_resident": self.ollama_busy,
            "ollama_models": sorted(self.models),
        }


def run_measure(
    pages: Iterable[Page],
    rows: Sequence[Row],
    *,
    runs: int,
    out: Path = MEASURE_PATH,
    sessions_path: Path | None = None,
    base_url: str | None = None,
    log: TextIO = sys.stderr,
) -> int:
    """Resumable: (row, article) pairs already in `out` are skipped. Returns records written."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(latest_by_key(read_records(out)))
    budget = signal_budget()
    url = base_url or get_settings().ollama_base_url
    session = Session()
    with out.open("a", encoding="utf-8") as fh:
        for i, page in enumerate(pages, 1):
            session.sample(url)
            todo = [row for row in rows if (row.name, page.id) not in done]
            fresh = [gold_record(page, lede_budget=budget)] if (GOLD_ROW, page.id) not in done else []
            t0 = time.perf_counter()
            for record in fresh:
                fh.write(json.dumps(record) + "\n")
            for row in todo:
                fh.write(json.dumps(measure_row(row, page, runs=runs, lede_budget=budget)) + "\n")
                fh.flush()
            session.records += len(fresh) + len(todo)
            print(
                f"[measure] {i} {page.id} {page.source}: {len(todo)} rows in {time.perf_counter() - t0:.1f}s",
                file=log,
                flush=True,
            )
    sessions_path = sessions_path or sessions_for(out)
    sessions = json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    sessions.append(session.as_json() | {"runs": runs})
    write_json(sessions_path, sessions)
    return session.records


def sessions_for(measure: Path) -> Path:
    return measure.with_name(measure.stem + SESSIONS_SUFFIX)


# --- signal ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalConfig:
    run: RunSpec
    threshold: float

    def run_config(self) -> Record:
        return {
            "bench": "extraction",
            "bench_version": BENCH_VERSION,
            **self.run.run_config(SIGNAL_SPLIT),
            "scope": "lede",
            "title": "gold",
        }

    def cache(self, digest: str, root: Path = PREDS_DIR) -> PredCache:
        return PredCache(
            self.run.model,
            self.run_config(),
            model_digest=digest,
            prompt_sha=self.run.prompt_sha(),
            root=root,
        )


def grid_result_path(model: str = SIGNAL_MODEL) -> Path:
    return GRID_DIR / f"signal__{model_slug(model)}.json"


def signal_config(result_path: Path | None = None) -> SignalConfig:
    """The grid's best config and operating threshold for the /signal decision model."""
    path = result_path or grid_result_path()
    if not path.exists():
        raise BenchError(f"{rel(path)} is missing; run `veritrade_evals.grid analyze` first")
    best = json.loads(path.read_text(encoding="utf-8"))["best"]
    cells = [c for c in grid_cells([SIGNAL_MODEL]) if c.name == best["config"]]
    if not cells:
        raise BenchError(f"grid config {best['config']!r} is not in the grid for {SIGNAL_MODEL}")
    if cells[0].prompt.version != best["prompt_version"]:
        raise BenchError(f"{rel(path)} was scored under another prompt version; re-run the grid analysis")
    return SignalConfig(cells[0], float(best["threshold"] or 0.0))


def record_key(row: str, article: str) -> str:
    return f"{row}|{article}"


@dataclass(frozen=True, slots=True)
class SignalJob:
    key: str
    gold: str
    input_sha: str | None
    input_tokens: int | None
    text: str | None
    extraction_error: str | None


def signal_jobs(pages: Iterable[Page], rows: Sequence[Row], max_tokens: int) -> Iterator[SignalJob]:
    for page in pages:
        gold = signal_text(page.gold_article, page.gold_title, max_tokens)
        yield SignalJob(
            record_key(GOLD_ROW, page.id), page.label, text_sha(gold), estimate_tokens(gold), gold, None
        )
        for row in rows:
            outcome = extract_row(row, page)
            key = record_key(row.name, page.id)
            if outcome.article is None:
                yield SignalJob(key, page.label, None, None, None, outcome.error)
                continue
            text = signal_text(outcome.article, page.gold_title, max_tokens)
            yield SignalJob(key, page.label, text_sha(text), estimate_tokens(text), text, None)


_META_FIELDS = frozenset(
    {"id", "gold", "error", "t", "input_sha", "input_tokens", "reused_from", "extraction_error"}
)


def _terminal(record: Record) -> bool:
    error = record.get("error")
    return error is None or not error.get("retryable", False)


def model_error(exc: VtError) -> PredError:
    """Same retry rule as the grid runner; the message is the type name so no model output is stored."""
    # Greedy seeded decoding repeats a schema failure, so it is final.
    retryable = not isinstance(exc, SchemaViolation) and (exc.retryable or isinstance(exc, ModelUnavailable))
    return PredError(exc.code.value, type(exc).__name__, retryable)


def _fresh(record: Record | None, job: SignalJob) -> bool:
    return (
        record is not None
        and _terminal(record)
        and record.get("input_sha") == job.input_sha
        and record.get("extraction_error") == job.extraction_error
    )


@dataclass(slots=True)
class FillStats:
    jobs: int = 0
    cached: int = 0
    calls: int = 0
    reused: int = 0
    extraction_failed: int = 0
    errors: int = 0


async def fill(
    cache: PredCache,
    jobs: Sequence[SignalJob],
    call: Caller,
    *,
    log: TextIO = sys.stderr,
    every_s: float = 30.0,
) -> FillStats:
    """One model call per distinct input, one record per (row, article); identical ledes reuse the answer."""
    stats = FillStats(jobs=len(jobs))
    records = cache.records()
    answered: dict[str, Record] = {}
    for record in records.values():
        if record.get("error") is None and record.get("input_sha"):
            answered.setdefault(record["input_sha"], record)
    pending: dict[str, list[SignalJob]] = {}
    for job in jobs:
        if _fresh(records.get(job.key), job):
            stats.cached += 1
        elif job.input_sha is None:
            cache.append(
                job.key,
                job.gold,
                pred={
                    "input_sha": None,
                    "input_tokens": None,
                    "reused_from": None,
                    "extraction_error": job.extraction_error,
                },
                error=PredError("extraction_failed", job.extraction_error or "unknown", retryable=False),
            )
            stats.extraction_failed += 1
        else:
            pending.setdefault(job.input_sha, []).append(job)
    todo = sum(sha not in answered for sha in pending)
    last = time.monotonic()
    for sha, group in pending.items():
        if sha not in answered:
            head = group[0]
            assert head.text is not None
            try:
                pred, timing = await call(head.text)
            except LlmUnreachable:
                raise
            except VtError as exc:
                for job in group:
                    cache.append(
                        job.key,
                        job.gold,
                        pred={"input_sha": sha, "input_tokens": job.input_tokens, "reused_from": None},
                        error=model_error(exc),
                    )
                stats.errors += len(group)
                continue
            answered[sha] = cache.append(
                head.key,
                head.gold,
                pred={**pred, "input_sha": sha, "input_tokens": head.input_tokens, "reused_from": None},
                t=timing,
            )
            stats.calls += 1
            group = group[1:]
            if time.monotonic() - last >= every_s:
                last = time.monotonic()
                print(f"[signal] {stats.calls}/{todo} model calls", file=log, flush=True)
        source = answered[sha]
        pred = {k: v for k, v in source.items() if k not in _META_FIELDS}
        for job in group:
            cache.append(
                job.key,
                job.gold,
                pred={
                    **pred,
                    "input_sha": sha,
                    "input_tokens": job.input_tokens,
                    "reused_from": source["id"],
                },
            )
            stats.reused += 1
    return stats


async def run_signal(
    pages: Iterable[Page],
    rows: Sequence[Row],
    cfg: SignalConfig,
    client: OllamaClient,
    *,
    preds_root: Path = PREDS_DIR,
    manifest_path: Path = SIGNAL_MANIFEST,
    log: TextIO = sys.stderr,
) -> FillStats:
    digest = await model_digest(client, cfg.run.model)
    jobs = list(signal_jobs(pages, rows, cfg.run.spec.max_input_tokens))
    distinct = len({j.input_sha for j in jobs if j.input_sha})
    print(f"[signal] {len(jobs)} (row, article) inputs, {distinct} distinct", file=log, flush=True)
    with cfg.cache(digest, preds_root) as cache:
        stats = await fill(cache, jobs, lambda text: call_once(client, cfg.run, text), log=log)
        key = cache.key
    manifest = {
        "model": cfg.run.model,
        "model_digest": digest,
        "config": cfg.run.name,
        "prompt_version": cfg.run.prompt.version,
        "threshold": cfg.threshold,
        "cache_key": key,
        "run_config": cfg.run_config(),
        "ollama_version": await client.version(),
        "updated_at": _now(),
    }
    write_json(manifest_path, manifest)
    return stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# --- install footprint -----------------------------------------------------------------------


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(requirements: Iterable[str]) -> set[str]:
    """Installed distributions reachable from requirement strings, with extras and markers honoured."""
    seen: set[tuple[str, str]] = set()
    names: set[str] = set()
    stack = list(requirements)
    while stack:
        req = Requirement(stack.pop())
        name = _canon(req.name)
        for extra in ("", *sorted(req.extras)):
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            try:
                dist = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
            names.add(name)
            for spec in dist.requires or []:
                dep = Requirement(spec)
                if dep.marker is None or dep.marker.evaluate({"extra": extra}):
                    stack.append(spec)
    return names


def dist_bytes(name: str) -> int:
    """Bytes of the files a distribution's RECORD lists (runtime __pycache__ is not counted)."""
    dist = metadata.distribution(name)
    total = 0
    for file in dist.files or []:
        path = Path(str(dist.locate_file(file)))
        if path.is_file():
            total += path.stat().st_size
    return total


def app_core() -> set[str]:
    """The backend's dependency closure without any parser or extractor package."""
    extraction = {_canon(d) for dists in (*PARSER_DISTS.values(), *EXTRACTOR_DISTS.values()) for d in dists}
    roots = [
        r for r in metadata.requires(BACKEND_DIST) or [] if _canon(Requirement(r).name) not in extraction
    ]
    return closure(roots)


def footprint_mb(dists: Iterable[str], core: set[str]) -> float:
    return sum(dist_bytes(d) for d in closure(dists) - core) / 1e6


def install_sizes(rows: Sequence[Row]) -> dict[str, dict[str, float]]:
    """Per row: what it adds to the app core (decimal MB), split into its parser and its extractor."""
    core = app_core()
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.parser is not None:
            parser = footprint_mb(PARSER_DISTS[row.parser], core)
            out[row.name] = {"installMb": parser, "parserMb": parser, "extractorMb": 0.0}
        else:
            total = footprint_mb(EXTRACTOR_DISTS[row.extractor], core)
            out[row.name] = {"installMb": total, "parserMb": 0.0, "extractorMb": total}
    return out


def package_versions() -> dict[str, str]:
    dists = sorted({d for ds in (*PARSER_DISTS.values(), *EXTRACTOR_DISTS.values()) for d in ds})
    return {d: metadata.version(d) for d in dists}


# --- report ----------------------------------------------------------------------------------


DIRECTIONAL = frozenset({"bullish", "bearish"})


def verdicts(records: Mapping[str, Record], threshold: float) -> dict[str, Sentiment | None]:
    """Served sentiment per record key after the operating threshold; None when no answer exists."""
    policy = ProbabilityPolicy(threshold)
    out: dict[str, Sentiment | None] = {}
    for key, record in records.items():
        probs = record.get("probs")
        out[key] = policy.decide(probs).sentiment if record.get("error") is None and probs else None
    return out


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT) -> list[float] | None:
    """95% percentile CI of the mean; a fixed seed resamples the same articles for every row (paired)."""
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(SEED).integers(0, len(data), size=(n_boot, len(data)))
    means = data[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def quality_scores(rec: Record) -> tuple[float, float, float]:
    """(precision, recall, F1); a failed extraction scores 0 on all three."""
    if rec.get("error") is not None:
        return 0.0, 0.0, 0.0
    q = Quality(rec["tokens_extracted"], rec["tokens_gold"], rec["tokens_common"], rec["tokens_leaked"])
    return q.precision, q.recall, q.f1


def _leakage(rec: Record) -> float:
    return float(rec["tokens_leaked"]) / rec["tokens_extracted"] if rec["tokens_extracted"] else 0.0


def signal_stats(
    row: str, ids: Sequence[str], verdict: Mapping[str, Sentiment | None] | None
) -> tuple[Record, list[float | None]]:
    """Agreement with the gold-lede verdict; a missing row verdict (failed extraction) counts as a miss."""
    if verdict is None:
        return {"signalAgreement": None}, [None] * len(ids)
    matches: list[float | None] = []
    flips = lost = new = missing = 0
    for article in ids:
        gold = verdict.get(record_key(GOLD_ROW, article))
        if gold is None:
            matches.append(None)
            continue
        got = verdict.get(record_key(row, article))
        matches.append(float(got == gold))
        if got is None:
            missing += 1
        elif gold in DIRECTIONAL and got in DIRECTIONAL and got != gold:
            flips += 1
        elif gold in DIRECTIONAL and got == "neutral":
            lost += 1
        elif gold == "neutral" and got in DIRECTIONAL:
            new += 1
    scored = [m for m in matches if m is not None]
    answered = len(scored) - missing
    stats: Record = {
        "signalAgreement": _mean(scored),
        "signalAgreementOk": sum(scored) / answered if answered else None,
        "signalN": len(scored),
        "signalMissing": missing,
        "signalFlips": flips,
        "signalLost": lost,
        "signalNew": new,
    }
    return stats, matches


def identical_share(row: str, ids: Sequence[str], lede_shas: Mapping[str, str | None] | None) -> float | None:
    """Share of articles whose /signal input from this row is byte-identical to the gold one."""
    if lede_shas is None:
        return None
    pairs = [(lede_shas.get(record_key(row, i)), lede_shas.get(record_key(GOLD_ROW, i))) for i in ids]
    scored = [float(mine == gold) for mine, gold in pairs if gold is not None]
    return _mean(scored)


def source_stats(recs: Sequence[Record], matches: Sequence[float | None]) -> Record:
    ok = [r for r in recs if r.get("error") is None]
    agree = [m for m in matches if m is not None]
    return {
        "n": len(recs),
        "failures": len(recs) - len(ok),
        "f1": _mean([quality_scores(r)[2] for r in recs]),
        "recall": _mean([quality_scores(r)[1] for r in recs]),
        "leakage": _mean([_leakage(r) for r in ok]),
        "p50Ms": percentile([r["ms"] for r in recs if r.get("ms") is not None], 50),
        "signalAgreement": _mean(agree),
        "methods": dict(Counter(r["method"] for r in ok)),
    }


def row_stats(
    row: Row,
    recs: Sequence[Record],
    verdict: Mapping[str, Sentiment | None] | None,
    install: Mapping[str, float],
    n_boot: int = N_BOOT,
    lede_shas: Mapping[str, str | None] | None = None,
) -> Record:
    """One matrix row in the frontend `ExtractionBenchmark` shape plus the fields the report needs."""
    ok = [r for r in recs if r.get("error") is None]
    scores = [quality_scores(r) for r in recs]
    ok_scores = [s for s, r in zip(scores, recs, strict=True) if r.get("error") is None]
    ms = [r["ms"] for r in recs if r.get("ms") is not None]
    signal, matches = signal_stats(row.name, [r["id"] for r in recs], verdict)
    agree = [m for m in matches if m is not None]
    by_source: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        by_source.setdefault(r["source"], []).append(i)
    return {
        "parser": row.parser,
        "extractor": row.extractor,
        "n": len(recs),
        "p50Ms": percentile(ms, 50),
        "p95Ms": percentile(ms, 95),
        "precision": _mean([s[0] for s in scores]),
        "recall": _mean([s[1] for s in scores]),
        "f1": _mean([s[2] for s in scores]),
        "leakage": _mean([_leakage(r) for r in ok]),
        "tokensToLlm": percentile([r["full_tokens"] for r in ok], 50),
        "failureRate": (len(recs) - len(ok)) / len(recs) if recs else None,
        "installMb": install["installMb"],
        "name": row.name,
        "f1Ci": bootstrap_ci([s[2] for s in scores], n_boot),
        "precisionOk": _mean([s[0] for s in ok_scores]),
        "recallOk": _mean([s[1] for s in ok_scores]),
        "f1Ok": _mean([s[2] for s in ok_scores]),
        "tokenLeakage": _mean([1 - s[0] for s in ok_scores]),
        "failures": dict(Counter(str(r["error"]) for r in recs if r.get("error") is not None)),
        "tokensToLlmP95": percentile([r["full_tokens"] for r in ok], 95),
        "ledeTokens": percentile([r["lede_tokens"] for r in ok], 50),
        "parseMsP50": percentile([r["parse_ms"] for r in recs if r.get("parse_ms") is not None], 50),
        "parserMb": install["parserMb"],
        "extractorMb": install["extractorMb"],
        "methods": dict(Counter(r["method"] for r in ok)),
        "titleMatchesGold": _mean([float(r["title_matches_gold"]) for r in ok]),
        "unstable": sum(not r.get("stable", True) for r in recs),
        **signal,
        "signalAgreementCi": bootstrap_ci(agree, n_boot),
        "ledeIdentical": identical_share(row.name, [r["id"] for r in recs], lede_shas),
        "perSource": {
            source: source_stats([recs[i] for i in members], [matches[i] for i in members])
            for source, members in sorted(by_source.items())
        },
    }


def paired_gap(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT) -> Record:
    diff = [x - y for x, y in zip(a, b, strict=True)]
    return {"point": _mean(diff), "ci": bootstrap_ci(diff, n_boot)}


def build_matrix(
    measured: Mapping[tuple[str, str], Record],
    rows: Sequence[Row],
    *,
    verdict: Mapping[str, Sentiment | None] | None,
    signal_meta: Record | None,
    install: Mapping[str, Mapping[str, float]],
    environment: Record,
    n_boot: int = N_BOOT,
    lede_shas: Mapping[str, str | None] | None = None,
) -> Record:
    ids = sorted({article for (row, article) in measured if row == GOLD_ROW})
    if not ids:
        raise BenchError("no measurements; run `extraction_bench measure` first")
    out_rows = []
    missing: list[str] = []
    for row in rows:
        recs = [measured[(row.name, i)] for i in ids if (row.name, i) in measured]
        if len(recs) < len(ids):
            missing.append(f"{row.name} ({len(ids) - len(recs)})")
        if recs:
            out_rows.append(row_stats(row, recs, verdict, install[row.name], n_boot, lede_shas))
    out_rows.sort(key=lambda r: (-(r["f1"] or 0.0), r["p50Ms"] or 0.0))
    gold = [measured[(GOLD_ROW, i)] for i in ids]
    gaps: Record = {}
    # Other parsers give the cascade identical bodies, so the challenger is the best other extractor.
    others = [r["name"] for r in out_rows if r["extractor"] != "cascade"]
    if others and not missing and any(r["name"] == SERVED_ROW for r in out_rows):
        best = others[0]
        f1_of = {name: [quality_scores(measured[(name, i)])[2] for i in ids] for name in (best, SERVED_ROW)}
        gaps = {"best": best, "served": SERVED_ROW, "f1": paired_gap(f1_of[best], f1_of[SERVED_ROW], n_boot)}
    gold_counts = Counter(str(verdict.get(record_key(GOLD_ROW, i))) for i in ids) if verdict else None
    return {
        "version": 1,
        "generatedBy": "uv run python -m veritrade_evals.extraction_bench report",
        "generatedAt": _now(),
        "articles": {
            "n": len(ids),
            "splits": dict(Counter(r["split"] for r in gold)),
            "perSource": dict(sorted(Counter(r["source"] for r in gold).items())),
        },
        "incomplete": missing,
        "environment": environment,
        "gold": {
            "tokensP50": percentile([r["tokens_gold"] for r in gold], 50),
            "fullTokensP50": percentile([r["full_tokens"] for r in gold], 50),
            "fullTokensP95": percentile([r["full_tokens"] for r in gold], 95),
            "ledeTokensP50": percentile([r["lede_tokens"] for r in gold], 50),
        },
        "signal": None if signal_meta is None else signal_meta | {"goldVerdicts": dict(gold_counts or {})},
        "servedGap": gaps,
        "rows": out_rows,
    }


# --- report rendering ------------------------------------------------------------------------


def _f(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _ms_cell(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def _ci_cell(point: float | None, ci: Sequence[float] | None) -> str:
    if point is None:
        return "-"
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{point:.3f}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"


def _row_label(row: Record) -> tuple[str, str]:
    return (f"`{row['parser']}`" if row["parser"] else "(built-in)", f"`{row['extractor']}`")


def render_summary(matrix: Record) -> str:
    rows: list[Record] = matrix["rows"]
    env = matrix["environment"]
    signal = matrix.get("signal")
    lines = [
        "# Extraction matrix",
        "",
        f"Generated by `{matrix['generatedBy']}` from local measurements; no article text is stored here.",
        "",
        f"- Data: the {matrix['articles']['n']} committed articles (all splits) from "
        f"{len(matrix['articles']['perSource'])} sources; stored HTML through the serving "
        "`veritrade.extract.registry.extract`; third-party extractors bring their own parser "
        "(parser column `(built-in)`).",
        "- Quality: macro means over articles against the Opus gold body. Tokens are lowercased `\\w+` "
        "runs; P/R/F1 are bag-of-words with multiplicity; a failed extraction scores 0. Leakage is the "
        f"share of extracted tokens outside every {SHINGLE}-token window that also occurs in the gold body "
        "(successful extractions only). 95% CIs: bootstrap over articles (B=1000).",
        "- Time: wall ms of one `extract` call (parse + extract), median of 5 runs per article after a "
        "warm-up, then p50/p95 over articles, failures included.",
        "- Tokens: chars / 4 of the uncapped `full` input (title + body) at p50; lede = the `/signal` "
        "lede input at the decision model's budget.",
        "- Install: decimal MB of the RECORD files a row adds to the backend core (its closure without "
        "any parser or extractor); own strategies add 0 beyond their parser.",
    ]
    if signal:
        lines.append(
            f"- Signal: `{signal['model']}` {signal['config']} (grid best), served sentiment at p >= "
            f"{signal['threshold']}; agreement = share of articles whose verdict on the row's lede equals "
            "the verdict on the gold lede, title held at the gold headline; a failed extraction is a miss."
        )
    lines += [
        f"- Machine: {env['chip']}, {env.get('ramGb')} GB, Python {env['python']}; "
        f"{env.get('sharedNote', 'machine load not recorded')}.",
        "",
        "## Matrix (sorted by F1)",
        "",
        "| # | parser | extractor | F1 [95% CI] | P | R | leakage | fail | p50 ms | p95 ms | tokens full | "
        "tokens lede | signal agree | install MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(rows, 1):
        parser, extractor = _row_label(row)
        lines.append(
            f"| {rank} | {parser} | {extractor} | {_ci_cell(row['f1'], row['f1Ci'])} | "
            f"{_f(row['precision'])} | {_f(row['recall'])} | {_pct(row['leakage'])} | "
            f"{_pct(row['failureRate'])} | {_ms_cell(row['p50Ms'])} | {_ms_cell(row['p95Ms'])} | "
            f"{_f(row['tokensToLlm'], 0)} | {_f(row['ledeTokens'], 0)} | "
            f"{_pct(row.get('signalAgreement'))} | {_f(row['installMb'], 1)} |"
        )
    gap = matrix.get("servedGap") or {}
    if gap:
        g = gap["f1"]
        ci = g["ci"]
        lines += [
            "",
            f"Paired bootstrap, F1 of the best other extractor `{gap['best']}` minus the served "
            f"`{gap['served']}`: "
            f"{g['point']:+.3f} "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}].",
        ]
    lines += ["", "## Failures and successful-only quality", ""]
    lines += [
        "| row | failures by type | F1 (ok only) | recall (ok only) | 1 - precision | unstable runs |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {_counts(row['failures'])} | {_f(row['f1Ok'])} | {_f(row['recallOk'])} | "
            f"{_pct(row['tokenLeakage'])} | {row['unstable']} |"
        )
    if signal:
        lines += [
            "",
            "## Downstream signal",
            "",
            f"Gold-lede verdicts: {_counts(signal['goldVerdicts'])}. {signal.get('inputs', '?')} (row, "
            "article) records including the gold ledes; failed extractions need no call and identical "
            f"ledes were sent once, so the model answered {signal.get('distinctInputs', '?')} distinct "
            "ledes.",
            "",
            "| row | agree [95% CI] | agree (ok only) | identical lede | bull<->bear flips "
            "| directional lost | directional added | no verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(rows, key=lambda r: -(r.get("signalAgreement") or 0.0)):
            agree = _ci_cell(row.get("signalAgreement"), row.get("signalAgreementCi"))
            lines.append(
                f"| `{row['name']}` | {agree} | {_f(row.get('signalAgreementOk'))} | "
                f"{_pct(row.get('ledeIdentical'))} | {row.get('signalFlips', '-')} | "
                f"{row.get('signalLost', '-')} | {row.get('signalNew', '-')} | "
                f"{row.get('signalMissing', '-')} |"
            )
    third = [r for r in rows if r["parser"] is None]
    picks = [r for r in rows if r["name"] == SERVED_ROW] + third[:1]
    for row in picks:
        lines += [
            "",
            f"## Per source: `{row['name']}`",
            "",
            "| source | n | fail | F1 | recall | leakage | p50 ms | signal agree | method |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for source, s in row["perSource"].items():
            lines.append(
                f"| {source} | {s['n']} | {s['failures']} | {_f(s['f1'])} | {_f(s['recall'])} | "
                f"{_pct(s['leakage'])} | {_ms_cell(s['p50Ms'])} | {_pct(s['signalAgreement'])} | "
                f"{_counts(s['methods'])} |"
            )
    sources = sorted({s for r in rows for s in r["perSource"]})
    lines += [
        "",
        "## F1 by source, every row",
        "",
        "| row | " + " | ".join(sources) + " |",
        "|---|" + "---|" * len(sources),
    ]
    for row in rows:
        cells = [_f(row["perSource"].get(s, {}).get("f1"), 2) for s in sources]
        lines.append(f"| `{row['name']}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Notes",
        "",
        "- The gold body is a subset of the `strip-tags` lines (selectolax blocks) chosen by Opus, so "
        "`strip-tags` recall is about 1 by construction and own strategies share the gold's text joining; "
        "scores are bag-of-words and the leakage only needs 5 tokens in order, so segmentation is moot.",
        "- The gold packets folded the page outside the span the served `selectolax/cascade` found (Opus "
        "could open the full page), so the gold leans toward the cascade; read own vs third-party gaps as "
        "upper bounds.",
        "- The globenewswire, prnewswire and yahoo profiles were tightened after a first measurement with "
        "these articles in view, so their scores, and the overall means they feed, are optimistic; "
        "`.context/docs/design/03-extraction.md` keeps the first-run numbers.",
        "- Timings come from one process on a shared workstation; see the environment block in "
        "matrix.json for load average and Ollama residency during the run.",
        "- The `jsonld` strategy reads JSON-LD from the raw HTML and never builds the tree, so its 4 rows "
        "differ only by timing noise.",
    ]
    if matrix.get("incomplete"):
        lines += ["", f"Incomplete rows (articles missing): {', '.join(matrix['incomplete'])}."]
    return "\n".join(lines) + "\n"


def environment(sessions_path: Path) -> Record:
    sessions = json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    samples = sum(s["ollama_samples"] for s in sessions)
    busy = sum(s["ollama_model_resident"] for s in sessions)
    loads = [s["loadavg_1m_mean"] for s in sessions if s.get("loadavg_1m_mean") is not None]
    note = (
        f"Ollama had a model resident in {busy}/{samples} per-article samples (another eval was running), "
        f"mean 1-min load average {statistics.fmean(loads):.1f}"
        if samples and loads
        else "machine load not recorded"
    )
    return machine() | {"packages": package_versions(), "sessions": sessions, "sharedNote": note}


def write_report(
    *,
    out_dir: Path = OUT_DIR,
    measure_path: Path = MEASURE_PATH,
    manifest_path: Path = SIGNAL_MANIFEST,
    preds_root: Path = PREDS_DIR,
    n_boot: int = N_BOOT,
    log: TextIO = sys.stderr,
) -> list[Path]:
    rows = matrix_rows()
    measured = latest_by_key(read_records(measure_path))
    verdict: dict[str, Sentiment | None] | None = None
    lede_shas: dict[str, str | None] | None = None
    signal_meta: Record | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = signal_config()
        cache = cfg.cache(manifest["model_digest"], preds_root)
        if cache.key != manifest["cache_key"]:
            print(
                "signal predictions were made under another prompt or runner config; re-run `signal`",
                file=log,
            )
        else:
            records = cache.records()
            verdict = verdicts(records, cfg.threshold)
            lede_shas = {key: r.get("input_sha") for key, r in records.items()}
            reused = sum(1 for r in records.values() if r.get("reused_from"))
            signal_meta = {
                "model": cfg.run.model,
                "config": cfg.run.name,
                "promptVersion": cfg.run.prompt.version,
                "threshold": cfg.threshold,
                "ollamaVersion": manifest.get("ollama_version"),
                "inputs": len(records),
                "distinctInputs": len({r["input_sha"] for r in records.values() if r.get("input_sha")}),
                "modelCalls": len(records)
                - reused
                - sum(1 for r in records.values() if r.get("extraction_error")),
            }
    else:
        print(f"no {rel(manifest_path)}: signal agreement left empty", file=log)
    matrix = build_matrix(
        measured,
        rows,
        verdict=verdict,
        signal_meta=signal_meta,
        install=install_sizes(rows),
        environment=environment(sessions_for(measure_path)),
        n_boot=n_boot,
        lede_shas=lede_shas,
    )
    matrix = rounded(matrix)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "matrix.json"
    json_path.write_text(json.dumps(matrix, indent=1) + "\n", encoding="utf-8")
    summary_path = out_dir / "summary.md"
    summary_path.write_text(render_summary(matrix), encoding="utf-8")
    return [json_path, summary_path]


# --- CLI -------------------------------------------------------------------------------------


def _quiet() -> None:
    """Extractor libraries log and warn about page content; nothing of it may reach the console."""
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")


def cmd_list(args: argparse.Namespace) -> int:
    rows = matrix_rows()
    for row in rows:
        kind = "own" if row.parser else "third-party"
        print(f"{row.name:28s} {kind}")
    print(f"{len(rows)} rows")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    _quiet()
    rows = select_rows(args.row)
    if args.fresh:
        args.out.unlink(missing_ok=True)
        sessions_for(args.out).unlink(missing_ok=True)
    written = run_measure(iter_pages(args.limit), rows, runs=args.runs, out=args.out)
    print(f"measure: {written} records -> {rel(args.out)}", file=sys.stderr)
    return 0


async def _signal(args: argparse.Namespace) -> int:
    rows = select_rows(args.row)
    cfg = signal_config()
    settings = get_settings()
    client = OllamaClient(
        args.base_url or settings.ollama_base_url,
        timeout=settings.inference_timeout_s,
        keep_alive=settings.ollama_keep_alive,
    )
    async with client:
        try:
            stats = await run_signal(iter_pages(args.limit), rows, cfg, client, preds_root=args.preds_root)
        except LlmUnreachable as exc:
            print(f"signal aborted: {exc.message}", file=sys.stderr)
            return 1
    print(
        f"signal: {stats.jobs} inputs, {stats.cached} cached, {stats.calls} model calls, "
        f"{stats.reused} reused, {stats.extraction_failed} extraction failures, {stats.errors} model errors",
        file=sys.stderr,
    )
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    _quiet()
    return asyncio.run(_signal(args))


def cmd_report(args: argparse.Namespace) -> int:
    paths = write_report(
        out_dir=args.out_dir, measure_path=args.measure, preds_root=args.preds_root, n_boot=args.n_boot
    )
    for path in paths:
        print(rel(path))
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `extraction` on a parent CLI; dispatch with `run(args)`."""
    bench: argparse.ArgumentParser = subparsers.add_parser("extraction", help="parser x extractor matrix")
    actions = bench.add_subparsers(dest="extraction_action", required=True)

    actions.add_parser("list", help="print the matrix rows").set_defaults(extraction_func=cmd_list)

    measure = actions.add_parser("measure", help="time and score every row on the stored HTML (resumable)")
    measure.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="timed runs per article after a warm-up"
    )
    measure.add_argument("--limit", type=int, default=0, help="first N articles only (smoke runs)")
    measure.add_argument("--row", action="append", help="restrict to this row (repeatable)")
    measure.add_argument("--out", type=Path, default=MEASURE_PATH)
    measure.add_argument("--fresh", action="store_true", help="discard earlier measurements first")
    measure.set_defaults(extraction_func=cmd_measure)

    signal = actions.add_parser("signal", help="decision-model verdicts on every row's lede (concurrency 1)")
    signal.add_argument("--limit", type=int, default=0, help="first N articles only (smoke runs)")
    signal.add_argument("--row", action="append", help="restrict to this row (repeatable)")
    signal.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    signal.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    signal.set_defaults(extraction_func=cmd_signal)

    report = actions.add_parser("report", help="write evals/results/extraction/{matrix.json,summary.md}")
    report.add_argument("--out-dir", type=Path, default=OUT_DIR)
    report.add_argument("--measure", type=Path, default=MEASURE_PATH)
    report.add_argument("--preds-root", type=Path, default=PREDS_DIR)
    report.add_argument("--n-boot", type=int, default=N_BOOT)
    report.set_defaults(extraction_func=cmd_report)
    return bench


def run(args: argparse.Namespace) -> int:
    code: int = args.extraction_func(args)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.extraction_bench", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["extraction", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
