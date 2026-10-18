"""`articles` command group. Prints ids, URLs, counts, statuses, dates and paths, never publisher text."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from veritrade.inference.types import SENTIMENTS
from veritrade_evals.articles import (
    PACKET_STAGES,
    REFERENCE_DATE,
    SEED,
    STAGES,
    WINDOW_START,
    ArticlePaths,
    ArticlesError,
    FetchRecord,
    iter_fetched,
    rel,
)
from veritrade_evals.articles_candidates import discover, write_view
from veritrade_evals.articles_fetch import fetch_articles, load_candidates, pick_top, resolve_ids
from veritrade_evals.articles_finalize import build_ledger, finalize, status, verify, write_finalized
from veritrade_evals.articles_packets import (
    build_gold_packet,
    prepare_adjudicate,
    prepare_label,
    write_gold_packet,
)
from veritrade_evals.articles_validate import adjudicate_requested, ingest
from veritrade_evals.discovery import load_sources, on_source_domain, primary_sources


def _sources(raw: Sequence[str] | None) -> list[str]:
    primary = primary_sources()
    if not raw or raw == ["all"]:
        return primary
    keys = [k.strip() for item in raw for k in item.split(",") if k.strip()]
    unknown = [k for k in keys if k not in primary]
    if unknown:
        raise ArticlesError(f"unknown or non-primary sources {unknown}; choose from {primary}")
    return keys


def _ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if getattr(args, "ids", None):
        ids += [i.strip() for i in args.ids.split(",") if i.strip()]
    if getattr(args, "ids_file", None):
        ids += [line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()]
    return ids


def _date(text: str) -> date:
    return date.fromisoformat(text)


# --- commands ---------------------------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    sources = _sources(args.source)
    registry = load_sources()
    for url in args.listing_url or []:
        if len(sources) != 1 or not on_source_domain(url, registry[sources[0]]):
            raise ArticlesError(f"--listing-url must be on the domain of exactly one --source: {url}")
    paths = ArticlePaths()
    reports = asyncio.run(
        discover(
            sources,
            since=args.since,
            until=args.until,
            queries=args.query or [],
            listing_urls=args.listing_url or [],
            max_new=args.max,
            pages=args.pages,
            refresh=args.refresh,
            paths=paths,
        )
    )
    for report in reports:
        print(
            f"{report.source}: total={report.total} in_window={report.in_window} undated={report.undated} "
            f"new={report.new} (matched={report.matched} out_of_window={report.out_of_window} "
            f"query_miss={report.query_miss} capped={report.capped})"
            f" -> {rel(paths.candidates(report.source))}"
        )
        for a in report.archives:
            errors = " ".join(f"{k}={v}" for k, v in sorted(a.errors.items())) or "-"
            print(
                f"  {a.kind:9} req={a.requests} ok={a.ok} errors={errors} links={a.links} "
                f"matched={a.matched} {a.url}"
            )
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    for source in _sources(args.source):
        rows, total = write_view(
            paths,
            source,
            queries=args.query or [],
            since=None if args.all_dates else WINDOW_START,
            until=None if args.all_dates else REFERENCE_DATE,
            unfetched=args.unfetched,
            limit=args.limit,
        )
        print(f"{source}: rows={rows} of candidates={total} -> {rel(paths.view(source))}")
    return 0


def _fetch_line(record: FetchRecord) -> str:
    if not record.ok:
        return f"{record.id} error {record.error} status={record.status}"
    extra = f" extract_error={record.extract_error}" if record.extract_error else ""
    return (
        f"{record.id} ok {record.status} words={record.words} method={record.method} "
        f"published={record.published_date or '-'}({record.published_from or '-'}){extra}"
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    [source] = _sources([args.source])
    candidates = load_candidates(paths, source)
    if args.top is not None:
        chosen = pick_top(candidates, paths, args.top, since=WINDOW_START, until=REFERENCE_DATE)
    else:
        chosen = resolve_ids(candidates, _ids(args))
    results = asyncio.run(fetch_articles(chosen, paths=paths, force=args.force))
    failed = 0
    for candidate, record in results:
        if record is None:
            print(f"{candidate.id} skip already fetched")
            continue
        failed += not record.ok
        print(_fetch_line(record))
    print(f"{source}: fetched={len(results)} errors={failed} -> {rel(paths.private / 'fetched')}")
    return 0


def _prepare_gold(paths: ArticlePaths, records: Sequence[FetchRecord], force: bool) -> int:
    written = 0
    for record in records:
        if not record.ok:
            continue
        if paths.packet("gold", record.id).exists() and not force:
            print(f"{record.id} skip packet exists")
            continue
        html = paths.html(record.id).read_text(encoding="utf-8")
        packet = build_gold_packet(record, html, paths)
        write_gold_packet(packet, paths)
        ratio = packet.visible_words / packet.body_words if packet.body_words else 0.0
        print(
            f"{record.id} lines={len(packet.lines)} rows={len(packet.rows)} "
            f"visible_words={packet.visible_words} "
            f"body_words={packet.body_words} ratio={ratio:.2f} located={str(packet.located).lower()} "
            f"bytes={len(packet.text.encode())}"
        )
        written += 1
    return written


def cmd_prepare(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    source = _sources([args.source])[0] if args.source else None
    wanted = set(_ids(args))
    records = [r for r in iter_fetched(paths, source) if not wanted or r.id in wanted]
    if args.stage == "gold":
        written = _prepare_gold(paths, records, args.force)
    else:
        ids = [r.id for r in records if r.ok]
        if args.stage == "adjudicate":
            requested = adjudicate_requested(paths)
            ids = [i for i in ids if i in requested]
        written = 0
        for article in ids:
            target = paths.packet(args.stage, article)
            if target.exists() and not args.force:
                continue
            prepared = (
                prepare_label(paths, article) if args.stage == "label" else prepare_adjudicate(paths, article)
            )
            if prepared.path is None:
                print(f"{article} skip {prepared.reason}")
                continue
            written += 1
            print(f"{article} ok {prepared.stats}".rstrip())
    print(f"{args.stage}: packets_written={written} -> {rel(paths.private / 'packets' / args.stage)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    source = _sources([args.source])[0] if args.source else None
    report = ingest(paths, args.stage, source)
    for result in report.results:
        verdict = "PASS" if result.ok else "FAIL " + ",".join(result.codes)
        print(f"{result.id} {verdict} {result.note}".rstrip())
    passed = sum(r.ok for r in report.results)
    print(f"{args.stage}: pass={passed} fail={len(report.results) - passed}")
    # Labellers run ingest too; the running agreement stays with the main loop to keep them independent.
    if (stats := report.agreement) is not None and args.agreement:
        raw = f"{stats.raw:.3f}" if stats.raw is not None else "-"
        kappa = f"{stats.kappa:.3f}" if stats.kappa is not None else "-"
        print(
            f"agreement: pairs={stats.n} agreed={stats.agreed} raw={raw} kappa={kappa} "
            f"to_adjudicate={len(stats.to_adjudicate)} -> {rel(paths.adjudicate_ids)}"
        )
    return 0 if passed == len(report.results) else 1


def _counts(counter: Any) -> str:
    return "/".join(str(counter[c]) for c in SENTIMENTS)


def cmd_status(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    rows = status(paths, _sources(args.source))
    print("counts by class are bullish/bearish/neutral")
    header = (
        f"{'source':14}{'cand':>6}{'win':>6}{'undat':>6}{'fetch':>6}{'ferr':>5}  {'gold usable':>11}"
        f"{'rej':>4}{'inv':>4}{'A':>4}{'B':>4}{'agree':>6}{'keep':>5}{'repl':>5}{'pend':>5}  {'eligible':>9}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.source:14}{r.candidates:>6}{r.in_window:>6}{r.undated:>6}{r.fetched_ok:>6}{r.fetch_errors:>5}"
            f"  {_counts(r.gold_usable):>11}{r.gold_rejected:>4}{r.gold_invalid:>4}{r.labelled_a:>4}"
            f"{r.labelled_b:>4}{r.agreed:>6}{r.adjudicated_keep:>5}{r.adjudicated_replace:>5}"
            f"{r.pending_adjudication:>5}  {_counts(r.eligible):>9}"
        )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    order = primary_sources()
    result = finalize(build_ledger(paths), order, seed=args.seed)
    write_finalized(result, paths)
    print(f"rows={len(result.rows)} splits={result.stats['splits']} classes={result.stats['classes']}")
    print(f"-> {rel(paths.rows)}")
    print(f"-> {rel(paths.stats)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    paths = ArticlePaths()
    results = verify(paths)
    bad = [(i, problems) for i, problems in results if problems]
    for article, problems in bad:
        print(f"{article} {','.join(problems)}")
    print(f"verified={len(results)} ok={len(results) - len(bad)} problems={len(bad)}")
    return 1 if bad else 0


def guarded(func: Any) -> Any:
    def run(args: argparse.Namespace) -> int:
        try:
            code: int = func(args)
        except ArticlesError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return code

    return run


# --- parser -----------------------------------------------------------------------------------------------


def register(commands: Any) -> None:
    group = commands.add_parser("articles", help="Opus-labelled article set (discover -> finalize)")
    sub = group.add_subparsers(dest="action", required=True)

    d = sub.add_parser("discover", help="collect candidate URLs from feeds, sitemaps, archives and search")
    d.add_argument("--source", action="append", help="primary source key(s), comma-separated, or 'all'")
    d.add_argument("--since", type=_date, default=WINDOW_START)
    d.add_argument("--until", type=_date, default=REFERENCE_DATE)
    d.add_argument("--listing-url", action="append", help="extra listing page on the source's domain")
    d.add_argument(
        "--query", action="append", help="keyword search where supported, else a title/slug filter"
    )
    d.add_argument("--max", type=int, help="cap on new candidates per source in this run")
    d.add_argument("--pages", type=int, help="pages per paged listing, search and EDGAR query")
    d.add_argument("--refresh", action="store_true", help="ignore the archive cache")
    d.set_defaults(func=guarded(cmd_discover))

    c = sub.add_parser("candidates", help="write a private TSV view of the candidates for an agent to read")
    c.add_argument("--source", action="append", required=True)
    c.add_argument("--query", action="append")
    c.add_argument("--unfetched", action="store_true")
    c.add_argument("--all-dates", action="store_true", help="include candidates dated outside the window")
    c.add_argument("--limit", type=int)
    c.set_defaults(func=guarded(cmd_candidates))

    f = sub.add_parser("fetch", help="fetch candidates through the serving fetcher")
    f.add_argument("--source", required=True)
    which = f.add_mutually_exclusive_group(required=True)
    which.add_argument("--ids", help="comma-separated article ids")
    which.add_argument("--ids-file", help="file with one article id per line")
    which.add_argument("--top", type=int, help="newest N unfetched candidates in the window")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=guarded(cmd_fetch))

    p = sub.add_parser("prepare", help="write agent packets for a stage")
    p.add_argument("--stage", choices=PACKET_STAGES, required=True)
    p.add_argument("--source")
    p.add_argument("--ids", help="comma-separated article ids")
    p.add_argument("--ids-file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=guarded(cmd_prepare))

    i = sub.add_parser("ingest", help="validate agent outputs of a stage")
    i.add_argument("--stage", choices=STAGES, required=True)
    i.add_argument("--source")
    i.add_argument("--agreement", action="store_true", help="print labeller agreement (main loop only)")
    i.set_defaults(func=guarded(cmd_ingest))

    s = sub.add_parser("status", help="per-source pipeline counts")
    s.add_argument("--source", action="append")
    s.set_defaults(func=guarded(cmd_status))

    z = sub.add_parser("finalize", help="select 3/3/3 per source, split 30/60, write committed files")
    z.add_argument("--seed", type=int, default=SEED)
    z.set_defaults(func=guarded(cmd_finalize))

    v = sub.add_parser("verify", help="committed rows match the local HTML and gold bodies")
    v.set_defaults(func=guarded(cmd_verify))
