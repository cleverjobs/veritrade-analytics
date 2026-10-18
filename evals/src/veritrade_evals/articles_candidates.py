"""Candidate store: discovery results per source, deduplicated by article id and merged across runs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from veritrade_evals.articles import ArticlePaths, Candidate, article_id, read_jsonl, write_jsonl
from veritrade_evals.discovery import (
    ArchiveClient,
    ArchiveStats,
    Discoverer,
    Found,
    query_matches,
    slug_text,
    url_date,
    utc_now,
)

SEARCH_KINDS = frozenset({"search", "edgar-fts"})


@dataclass(slots=True)
class SourceReport:
    source: str
    total: int = 0
    in_window: int = 0
    undated: int = 0
    new: int = 0
    matched: int = 0
    out_of_window: int = 0
    query_miss: int = 0
    capped: int = 0
    archives: list[ArchiveStats] = field(default_factory=list)


def published_of(found: Found) -> date | None:
    return date.fromisoformat(found.link.published) if found.link.published else url_date(found.link.url)


def matched_query(found: Found, queries: Sequence[str]) -> str | None:
    """Search archives already filtered server-side; everything else is matched on title and URL slug."""
    if found.via in SEARCH_KINDS and found.query:
        return found.query
    haystack = f"{found.link.title or ''} {slug_text(found.link.url)}"
    return next((q for q in queries if query_matches(q, haystack)), None)


def to_candidate(source: str, found: Found, query: str | None) -> Candidate:
    published = published_of(found)
    return Candidate(
        id=article_id(found.link.url),
        source=source,
        url=found.link.url,
        title=found.link.title,
        published_at=published.isoformat() if published else None,
        via=found.via,
        via_url=found.via_url,
        query=query,
        extra={k: v for k, v in found.link.extra.items() if k != "lastmod"},
    )


def merge(existing: Sequence[Candidate], incoming: Iterable[Candidate]) -> tuple[list[Candidate], int]:
    """Existing order first; a later sighting only fills a missing title or date."""
    merged = {c.id: c for c in existing}
    new = 0
    for candidate in incoming:
        old = merged.get(candidate.id)
        if old is None:
            merged[candidate.id] = candidate
            new += 1
            continue
        fill = {}
        if not old.title and candidate.title:
            fill["title"] = candidate.title
        if not old.published_at and candidate.published_at:
            fill["published_at"] = candidate.published_at
        if fill:
            merged[candidate.id] = old.model_copy(update=fill)
    return list(merged.values()), new


def select_found(
    report: SourceReport,
    found: Sequence[Found],
    *,
    since: date,
    until: date,
    queries: Sequence[str],
    max_new: int | None,
    known: set[str],
) -> list[Candidate]:
    picked: list[Candidate] = []
    seen: set[str] = set()
    for item in found:
        report.matched += 1
        published = published_of(item)
        if published is not None and not since <= published <= until:
            report.out_of_window += 1
            continue
        query = matched_query(item, queries) if queries else None
        if queries and query is None:
            report.query_miss += 1
            continue
        candidate = to_candidate(report.source, item, query)
        if candidate.id not in known and candidate.id not in seen:
            if max_new is not None and len(seen) >= max_new:
                report.capped += 1
                continue
            seen.add(candidate.id)
        picked.append(candidate)
    return picked


async def discover(
    sources: Sequence[str],
    *,
    since: date,
    until: date,
    queries: Sequence[str] = (),
    listing_urls: Sequence[str] = (),
    max_new: int | None = None,
    pages: int | None = None,
    refresh: bool = False,
    paths: ArticlePaths | None = None,
    client: ArchiveClient | None = None,
) -> list[SourceReport]:
    paths = paths or ArticlePaths()
    archive_client = client or ArchiveClient(paths.cache, refresh=refresh)
    discoverer = Discoverer(archive_client)
    try:
        runs = await asyncio.gather(
            *(
                discoverer.run(
                    key, since=since, until=until, queries=queries, listing_urls=listing_urls, pages=pages
                )
                for key in sources
            )
        )
    finally:
        if client is None:
            await archive_client.aclose()
    reports = []
    for run in runs:
        report = SourceReport(run.source, archives=run.archives)
        existing = read_jsonl(paths.candidates(run.source), Candidate)
        picked = select_found(
            report,
            run.found,
            since=since,
            until=until,
            queries=queries,
            max_new=max_new,
            known={c.id for c in existing},
        )
        merged, report.new = merge(existing, picked)
        write_jsonl(paths.candidates(run.source), merged)
        report.total = len(merged)
        report.undated = sum(c.published is None for c in merged)
        report.in_window = sum(c.published is not None and since <= c.published <= until for c in merged)
        log_run(paths, report, since=since, until=until, queries=queries)
        reports.append(report)
    return reports


def log_run(
    paths: ArticlePaths, report: SourceReport, *, since: date, until: date, queries: Sequence[str]
) -> None:
    entry = {
        "at": utc_now(),
        "source": report.source,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "queries": list(queries),
        "total": report.total,
        "in_window": report.in_window,
        "undated": report.undated,
        "new": report.new,
        "archives": [
            {"kind": a.kind, "url": a.url, "requests": a.requests, "ok": a.ok, "errors": a.errors,
             "links": a.links, "matched": a.matched}
            for a in report.archives
        ],
    }  # fmt: skip
    paths.discovery_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.discovery_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def write_view(
    paths: ArticlePaths,
    source: str,
    *,
    queries: Sequence[str] = (),
    since: date | None = None,
    until: date | None = None,
    unfetched: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    """Private TSV (id, date, via, query, title or slug) for an agent to Read; returns (rows, candidates)."""
    candidates = read_jsonl(paths.candidates(source), Candidate)
    rows = []
    for c in sorted(candidates, key=lambda c: (c.published_at or "", c.id), reverse=True):
        if since and until and c.published is not None and not since <= c.published <= until:
            continue
        if unfetched and paths.fetched(c.id).exists():
            continue
        text = c.title or slug_text(c.url)
        if queries and not any(query_matches(q, f"{text} {slug_text(c.url)}") for q in queries):
            continue
        extra = c.extra.get("items", "")
        rows.append("\t".join((c.id, c.published_at or "-", c.via, c.query or "-", extra or "-", text)))
        if limit is not None and len(rows) >= limit:
            break
    path = paths.view(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "id\tpublished\tvia\tquery\tedgar_items\ttitle_or_slug\n"
    path.write_text(header + "".join(row + "\n" for row in rows), encoding="utf-8")
    return len(rows), len(candidates)
