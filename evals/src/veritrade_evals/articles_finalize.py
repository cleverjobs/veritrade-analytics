"""Ledger over every pipeline stage, then the 90-article selection, seeded dev/test split and stats."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from veritrade.inference.types import SENTIMENTS, Sentiment
from veritrade_evals.articles import (
    PER_CLASS,
    REFERENCE_DATE,
    SEED,
    WINDOW_START,
    AdjudicationSummary,
    ArticleLabels,
    ArticlePaths,
    ArticleRow,
    ArticlesError,
    ArticleSplit,
    Candidate,
    FetchRecord,
    LabelOutput,
    LabelSummary,
    iter_fetched,
    read_jsonl,
    row_problems,
    sha256_bytes,
    write_json,
    write_jsonl,
)
from veritrade_evals.articles_validate import (
    AdjudicateCheck,
    GoldCheck,
    LabelCheck,
    adjudicate_check,
    adjudicate_requested,
    agreement,
    cohen_kappa,
    gold_body,
    gold_check,
    label_check,
    needs_adjudication,
    shares_run,
    shingles,
)

Route = Literal["agreed", "adjudicated"]
SLACK = 1
ALGORITHM = (
    "Eligible: gold usable, published in the window, both labellers valid, and either the labels agree with "
    "neither flagged ambiguous or the adjudicator kept the article (its label is final). Quotas start at "
    "min(3, eligible) per source and class; a source below 2 in any class makes the set impossible. Classes "
    "short of 30 (largest deficit first) take a 4th article from the source with the fewest selected so far, "
    "then the most spare eligible articles, then source rank. Within a cell, agreed articles precede "
    "adjudicated ones, each group in seeded order; one seeded article per cell goes to dev, the rest to test."
)


# --- ledger -----------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Entry:
    id: str
    source: str
    fetched: FetchRecord
    gold: GoldCheck
    gold_ingested: bool
    a: LabelCheck
    b: LabelCheck
    adjudication: AdjudicateCheck | None

    @property
    def pair_ok(self) -> bool:
        return self.gold_ingested and self.a.ok and self.b.ok

    @property
    def published(self) -> date | None:
        return self.fetched.published

    def final(
        self, since: date = WINDOW_START, until: date = REFERENCE_DATE
    ) -> tuple[Sentiment, Route] | None:
        if not self.pair_ok or self.published is None or not since <= self.published <= until:
            return None
        a, b = self.a.output, self.b.output
        assert a is not None
        assert b is not None
        if not needs_adjudication(a, b):
            return a.label, "agreed"
        adj = self.adjudication
        if adj is not None and adj.ok and adj.output is not None and adj.output.decision == "keep":
            return adj.output.label, "adjudicated"
        return None


def build_ledger(paths: ArticlePaths, source: str | None = None) -> list[Entry]:
    requested = adjudicate_requested(paths)
    entries = []
    for record in iter_fetched(paths, source):
        if not record.ok:
            continue
        gold = gold_check(paths, record.id)
        body = gold_body(paths, record.id)
        entries.append(
            Entry(
                id=record.id,
                source=record.source,
                fetched=record,
                gold=gold,
                gold_ingested=gold.usable and body is not None and body == gold.body,
                a=label_check(paths, "label_a", record.id),
                b=label_check(paths, "label_b", record.id),
                adjudication=adjudicate_check(paths, record.id, requested)
                if record.id in requested
                else None,
            )
        )
    return entries


# --- status -----------------------------------------------------------------------------------------------


@dataclass(slots=True)
class SourceStatus:
    source: str
    candidates: int = 0
    in_window: int = 0
    undated: int = 0
    fetched_ok: int = 0
    fetch_errors: int = 0
    gold_usable: Counter[str] = field(default_factory=Counter)
    gold_rejected: int = 0
    gold_invalid: int = 0
    labelled_a: int = 0
    labelled_b: int = 0
    agreed: int = 0
    adjudicated_keep: int = 0
    adjudicated_replace: int = 0
    pending_adjudication: int = 0
    eligible: Counter[str] = field(default_factory=Counter)


def status(paths: ArticlePaths, sources: Sequence[str]) -> list[SourceStatus]:
    out = []
    for source in sources:
        row = SourceStatus(source)
        candidates = read_jsonl(paths.candidates(source), Candidate)
        row.candidates = len(candidates)
        row.in_window = sum(
            c.published is not None and WINDOW_START <= c.published <= REFERENCE_DATE for c in candidates
        )
        row.undated = sum(c.published is None for c in candidates)
        for record in iter_fetched(paths, source):
            if not record.ok:
                row.fetch_errors += 1
        entries = build_ledger(paths, source)
        pairs = label_pairs(entries)
        for entry in entries:
            row.fetched_ok += 1
            gold = entry.gold
            if gold.codes == ("missing_output",):
                pass
            elif gold.output is None or not gold.ok:
                row.gold_invalid += 1
            elif not gold.output.usable:
                row.gold_rejected += 1
            elif entry.gold_ingested and gold.output.provisional_label:
                row.gold_usable[gold.output.provisional_label] += 1
            row.labelled_a += entry.a.ok
            row.labelled_b += entry.b.ok
            decision = adjudication_decision(entry)
            if entry.id in pairs:
                if not needs_adjudication(*pairs[entry.id]):
                    row.agreed += 1
                elif decision is None:
                    row.pending_adjudication += 1
            row.adjudicated_keep += decision == "keep"
            row.adjudicated_replace += decision == "replace"
            if final := entry.final():
                row.eligible[final[0]] += 1
        out.append(row)
    return out


# --- quotas, selection, split -----------------------------------------------------------------------------


class ShortfallError(ArticlesError):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("article set cannot be filled:\n  " + "\n  ".join(problems))
        self.problems = list(problems)


def plan_quotas(
    available: Mapping[str, Mapping[Sentiment, int]],
    order: Sequence[str],
    *,
    per_class: int = PER_CLASS,
    slack: int = SLACK,
) -> dict[str, dict[Sentiment, int]]:
    """Per (source, class) counts within per_class +- slack whose class totals are per_class * sources."""
    low, high, target = per_class - slack, per_class + slack, per_class * len(order)
    avail = {s: {c: int(available.get(s, {}).get(c, 0)) for c in SENTIMENTS} for s in order}
    problems = [
        f"{s} {c}: {avail[s][c]} eligible, at least {low} needed"
        for s in order
        for c in SENTIMENTS
        if avail[s][c] < low
    ]
    quota = {s: {c: min(per_class, avail[s][c]) for c in SENTIMENTS} for s in order}
    deficit = {c: target - sum(quota[s][c] for s in order) for c in SENTIMENTS}
    for c in sorted(SENTIMENTS, key=lambda c: (-deficit[c], SENTIMENTS.index(c))):
        while (total := sum(quota[s][c] for s in order)) < target:
            options = [s for s in order if quota[s][c] < high and avail[s][c] > quota[s][c]]
            if not options:
                capacity = sum(min(high, avail[s][c]) for s in order)
                problems.append(
                    f"{c}: at most {capacity} of {target} with +-{slack} per source (have {total})"
                )
                break
            best = min(
                options,
                key=lambda s: (sum(quota[s].values()), -(avail[s][c] - quota[s][c]), order.index(s)),
            )
            quota[best][c] += 1
    if problems:
        raise ShortfallError(problems)
    return quota


def _rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(":".join((str(seed), *parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def choose(cell: Mapping[str, Route], k: int, *, seed: int, source: str, label: str) -> list[str]:
    rng = _rng(seed, "pick", source, label)
    groups = []
    for route in ("agreed", "adjudicated"):
        ids = sorted(i for i, r in cell.items() if r == route)
        rng.shuffle(ids)
        groups += ids
    return groups[:k]


def split_cell(ids: Sequence[str], *, seed: int, source: str, label: str) -> dict[str, ArticleSplit]:
    order = sorted(ids)
    _rng(seed, "split", source, label).shuffle(order)
    return {article: ("dev" if n == 0 else "test") for n, article in enumerate(order)}


# --- rows, guard, stats -----------------------------------------------------------------------------------


def _summary(check: LabelCheck) -> LabelSummary:
    out = check.output
    assert out is not None
    return LabelSummary(label=out.label, score=out.score, ambiguous=out.ambiguous, confidence=out.confidence)


def build_row(entry: Entry, route: Route, split: ArticleSplit) -> ArticleRow:
    a, b, gold = entry.a.output, entry.b.output, entry.gold.output
    assert a is not None
    assert b is not None
    assert gold is not None
    assert entry.gold.body is not None
    adj = entry.adjudication.output if entry.adjudication is not None and entry.adjudication.ok else None
    if route == "adjudicated":
        assert adj is not None
        label, score, entity, rationale = adj.label, float(adj.score), adj.entity, adj.rationale
    else:
        label, score, entity, rationale = a.label, (a.score + b.score) / 2, a.entity, a.rationale
    ticker = next((t for t in (adj.ticker if adj else None, a.ticker, b.ticker, gold.ticker) if t), None)
    summary = AdjudicationSummary(decision=adj.decision, label=adj.label, score=adj.score) if adj else None
    published = entry.published
    assert published is not None
    return ArticleRow(
        id=entry.id,
        url=entry.fetched.url,
        source=entry.source,
        published_at=published.isoformat(),
        fetched_at=entry.fetched.fetched_at,
        html_sha256=entry.fetched.html_sha256 or "",
        gold_sha256=sha256_bytes(entry.gold.body.encode("utf-8")),
        label=label,
        score=score,
        entity=entity,
        ticker=ticker,
        rationale=rationale,
        key_facts=list(gold.key_facts),
        split=split,
        labels=ArticleLabels(a=_summary(entry.a), b=_summary(entry.b), adjudicated=summary, final=label),
    )


TITLE_GUARD_WORDS = 5


def publisher_text_problems(
    rows: Sequence[ArticleRow], bodies: Mapping[str, str], titles: Mapping[str, str]
) -> list[str]:
    """Committed strings must not share a 10-word run with the body nor contain the headline."""
    problems = []
    for row in rows:
        body_runs = shingles(bodies.get(row.id, ""))
        title = (titles.get(row.id) or "").casefold()
        for name, text in [("rationale", row.rationale), ("entity", row.entity)] + [
            (f"key_facts[{n}]", fact) for n, fact in enumerate(row.key_facts)
        ]:
            if shares_run(text, body_runs):
                problems.append(f"{row.id}:{name}:verbatim_body")
            if len(title.split()) >= TITLE_GUARD_WORDS and title in text.casefold():
                problems.append(f"{row.id}:{name}:headline")
    return problems


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def label_pairs(entries: Sequence[Entry]) -> dict[str, tuple[LabelOutput, LabelOutput]]:
    pairs = {}
    for e in entries:
        if e.pair_ok and e.a.output is not None and e.b.output is not None:
            pairs[e.id] = (e.a.output, e.b.output)
    return pairs


def adjudication_decision(entry: Entry) -> str | None:
    adj = entry.adjudication
    return adj.output.decision if adj is not None and adj.ok and adj.output is not None else None


def pool_stats(entries: Sequence[Entry]) -> dict[str, object]:
    agreed = agreement(label_pairs(entries))
    decisions = [d for e in entries if (d := adjudication_decision(e))]
    adjudicated, replaced = len(decisions), decisions.count("replace")
    return {
        "gold_usable": sum(e.gold_ingested for e in entries),
        "double_labelled": agreed.n,
        "label_agreement": agreed.agreed,
        "raw_agreement": _round(agreed.raw),
        "cohen_kappa": _round(agreed.kappa),
        "sent_to_adjudication": len(agreed.to_adjudicate),
        "adjudication_rate": _rate(len(agreed.to_adjudicate), agreed.n),
        "adjudicated": adjudicated,
        "replaced": replaced,
        "replacement_rate": _rate(replaced, adjudicated),
    }


@dataclass(frozen=True, slots=True)
class Finalized:
    rows: tuple[ArticleRow, ...]
    stats: dict[str, object]


def finalize(
    entries: Sequence[Entry], order: Sequence[str], *, seed: int = SEED, per_class: int = PER_CLASS
) -> Finalized:
    by_id = {e.id: e for e in entries}
    cells: dict[str, dict[Sentiment, dict[str, Route]]] = {s: {c: {} for c in SENTIMENTS} for s in order}
    for entry in entries:
        if entry.source in cells and (final := entry.final()):
            cells[entry.source][final[0]][entry.id] = final[1]
    available = {s: {c: len(cells[s][c]) for c in SENTIMENTS} for s in order}
    quota = plan_quotas(available, order, per_class=per_class)
    rows: list[ArticleRow] = []
    for source in order:
        for label in SENTIMENTS:
            picked = choose(cells[source][label], quota[source][label], seed=seed, source=source, label=label)
            splits = split_cell(picked, seed=seed, source=source, label=label)
            rows += [build_row(by_id[i], cells[source][label][i], splits[i]) for i in sorted(picked)]
    bodies = {r.id: by_id[r.id].gold.body or "" for r in rows}
    titles = {r.id: by_id[r.id].gold.title or "" for r in rows}
    if problems := publisher_text_problems(rows, bodies, titles):
        raise ArticlesError("publisher text in committed fields: " + ", ".join(problems[:6]))
    chosen = [by_id[r.id] for r in rows]
    final_a = [e.a.output.label for e in chosen if e.a.output]
    final_b = [e.b.output.label for e in chosen if e.b.output]
    stats: dict[str, object] = {
        "reference_date": REFERENCE_DATE.isoformat(),
        "window": {"since": WINDOW_START.isoformat(), "until": REFERENCE_DATE.isoformat()},
        "seed": seed,
        "n": len(rows),
        "splits": dict(sorted(Counter(r.split for r in rows).items())),
        "classes": {c: sum(r.label == c for r in rows) for c in SENTIMENTS},
        "per_source": {
            s: {c: sum(r.source == s and r.label == c for r in rows) for c in SENTIMENTS} for s in order
        },
        "quota_adjustments": [
            {"source": s, "label": c, "quota": quota[s][c]}
            for s in order
            for c in SENTIMENTS
            if quota[s][c] != per_class
        ],
        "pool": pool_stats(entries),
        "final_set": {
            "raw_agreement": _round(sum(x == y for x, y in zip(final_a, final_b, strict=True)) / len(rows))
            if rows
            else None,
            "cohen_kappa": _round(cohen_kappa(final_a, final_b)),
            "adjudicated_kept": sum(r.labels.adjudicated is not None for r in rows),
        },
        "algorithm": ALGORITHM,
    }
    return Finalized(tuple(rows), stats)


def write_finalized(result: Finalized, paths: ArticlePaths) -> None:
    write_jsonl(paths.rows, result.rows)
    write_json(paths.stats, result.stats)


def verify(paths: ArticlePaths) -> list[tuple[str, list[str]]]:
    rows = read_jsonl(paths.rows, ArticleRow)
    if not rows:
        raise ArticlesError(f"no committed rows at {paths.rows}")
    return [(row.id, row_problems(row, paths)) for row in rows]
