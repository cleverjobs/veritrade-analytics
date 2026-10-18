"""Fetch candidates through the serving HttpFetcher, keep the raw HTML privately, sanity-check extraction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from veritrade.config import get_settings
from veritrade.errors import VtError
from veritrade.extract import registry
from veritrade.fetch import Fetcher, HttpFetcher
from veritrade_evals.articles import (
    ArticlePaths,
    ArticlesError,
    Candidate,
    FetchRecord,
    read_fetched,
    read_jsonl,
    sha256_bytes,
)
from veritrade_evals.discovery import ArchiveClient, Politeness, parse_date, utc_now


def pick_top(
    candidates: Sequence[Candidate], paths: ArticlePaths, n: int, *, since: date, until: date
) -> list[Candidate]:
    """Newest unfetched candidates in the window; undated ones come after every dated one."""
    pool = [
        c
        for c in candidates
        if not _fetched_ok(paths, c.id) and (c.published is None or since <= c.published <= until)
    ]
    pool.sort(key=lambda c: (c.published is not None, c.published_at or "", c.id), reverse=True)
    return pool[:n]


def _fetched_ok(paths: ArticlePaths, article: str) -> bool:
    record = read_fetched(paths, article)
    return record is not None and record.ok


def resolve_ids(candidates: Sequence[Candidate], ids: Sequence[str]) -> list[Candidate]:
    by_id = {c.id: c for c in candidates}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ArticlesError(
            f"{len(unknown)} ids are not candidates of this source, e.g. {', '.join(unknown[:3])}"
        )
    return [by_id[i] for i in ids]


def analyze(html: str, url: str, candidate: Candidate) -> dict[str, object]:
    """Cascade extraction as a sanity check: word count, winning method, page publish date."""
    fields: dict[str, object] = {"words": 0, "method": None, "published_at": None, "extract_error": None}
    try:
        article = registry.extract(html, url, parser="selectolax", extractor="cascade")
    except VtError as exc:
        fields["extract_error"] = exc.code.value
    else:
        fields.update(
            words=article.word_count,
            method=article.method,
            published_at=article.published_at,
            paywalled=article.paywalled,
        )
    page_date = parse_date(str(fields["published_at"])) if fields["published_at"] else None
    if page_date is not None:
        fields.update(published_date=page_date.isoformat(), published_from="page")
    elif candidate.published_at:
        fields.update(published_date=candidate.published_at, published_from="candidate")
    return fields


async def fetch_one(
    candidate: Candidate, paths: ArticlePaths, fetcher: Fetcher, polite: Politeness, robots: ArchiveClient
) -> FetchRecord:
    base: dict[str, object] = {
        "id": candidate.id,
        "source": candidate.source,
        "url": candidate.url,
        "fetched_at": utc_now(),
    }
    if not await robots.allowed(candidate.url):
        return FetchRecord.model_validate(base | {"error": "robots_disallowed"})
    lock = await polite.wait(candidate.url)
    try:
        result = await fetcher.get(candidate.url)
    except VtError as exc:
        return FetchRecord.model_validate(base | {"error": exc.code.value, "status": exc.upstream_status})
    finally:
        polite.done(candidate.url, lock)
    raw = result.html.encode("utf-8")
    html_path = paths.html(candidate.id)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_bytes(raw)
    fields = analyze(result.html, result.final_url, candidate)
    return FetchRecord.model_validate(
        base
        | {"final_url": result.final_url, "status": result.status, "html_sha256": sha256_bytes(raw)}
        | fields
    )


async def fetch_articles(
    candidates: Sequence[Candidate],
    *,
    paths: ArticlePaths,
    force: bool = False,
    fetcher: Fetcher | None = None,
    robots: ArchiveClient | None = None,
) -> list[tuple[Candidate, FetchRecord | None]]:
    """Sequential fetches; (candidate, None) marks one skipped because it was already fetched."""
    own_fetcher = fetcher is None
    own_robots = robots is None
    active: Fetcher = fetcher or HttpFetcher(get_settings())
    robots_client = robots or ArchiveClient(paths.cache)
    polite = Politeness()
    results: list[tuple[Candidate, FetchRecord | None]] = []
    try:
        for candidate in candidates:
            if not force and _fetched_ok(paths, candidate.id):
                results.append((candidate, None))
                continue
            record = await fetch_one(candidate, paths, active, polite, robots_client)
            path = paths.fetched(candidate.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
            results.append((candidate, record))
    finally:
        if own_fetcher:
            await active.aclose()
        if own_robots:
            await robots_client.aclose()
    return results


def load_candidates(paths: ArticlePaths, source: str) -> list[Candidate]:
    candidates = read_jsonl(paths.candidates(source), Candidate)
    if not candidates:
        raise ArticlesError(f"no candidates for {source}; run `articles discover --source {source}` first")
    return candidates
