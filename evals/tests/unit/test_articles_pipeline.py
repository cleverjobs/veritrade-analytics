from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from veritrade.fetch.types import FetchResult
from veritrade.inference.types import SENTIMENTS, Sentiment
from veritrade_evals import __main__ as cli
from veritrade_evals import articles_cli
from veritrade_evals.articles import (
    ArticlePaths,
    ArticleRow,
    ArticlesError,
    Candidate,
    FetchRecord,
    article_id,
    load_articles,
    read_jsonl,
    write_json,
)
from veritrade_evals.articles_candidates import merge, select_found, write_view
from veritrade_evals.articles_fetch import fetch_articles, pick_top
from veritrade_evals.articles_finalize import (
    ShortfallError,
    build_ledger,
    choose,
    finalize,
    plan_quotas,
    publisher_text_problems,
    split_cell,
    status,
    verify,
    write_finalized,
)
from veritrade_evals.articles_validate import adjudicate_requested, ingest, shingles
from veritrade_evals.discovery import ArchiveClient, Found, Link, Politeness
from veritrade_evals.paths import ARTICLES_DIR

SOURCES = ["alpha", "beta", "gamma"]
SCORES: dict[Sentiment, int] = {"bullish": 2, "bearish": -1, "neutral": 0}


@pytest.fixture
def paths(tmp_path: Path) -> ArticlePaths:
    return ArticlePaths(private=tmp_path / "private", committed=tmp_path / "committed")


def body_lines(tag: str) -> list[str]:
    return [
        f"{tag} sentence {n} reports that the {tag} unit shipped {n * 11} crates while costs held near "
        f"{n * 3} percent overall"
        for n in range(1, 8)
    ]


def seed(
    paths: ArticlePaths,
    article: str,
    source: str,
    label: Sentiment,
    *,
    b_label: Sentiment | None = None,
    ambiguous: bool = False,
    title: str | None = None,
    published: str = "2026-10-01",
) -> None:
    html = f"<html><body>{article}</body></html>".encode()
    paths.html(article).parent.mkdir(parents=True, exist_ok=True)
    paths.html(article).write_bytes(html)
    from veritrade_evals.articles import sha256_bytes

    record = FetchRecord(
        id=article,
        source=source,
        url=f"https://{source}.example.com/{article}",
        fetched_at="2026-10-18T09:00:00+00:00",
        status=200,
        html_sha256=sha256_bytes(html),
        words=150,
        method="profile",
        published_date=published,
        published_from="page",
    )
    write_json(paths.fetched(article), record.model_dump())
    lines = ["Menu", title or f"{article} headline about {label} news", *body_lines(article)]
    write_json(paths.lines(article), {"id": article, "lines": lines})
    gold = {
        "id": article,
        "usable": True,
        "title_line": 2,
        "body_ranges": [[3, 9]],
        "key_facts": [
            "The unit shipped 77 crates.",
            "Costs held near 21 percent.",
            "Shipments began at 11 crates.",
        ],
        "entity": f"{source} holdings",
        "ticker": None,
        "provisional_label": label,
    }
    write_json(paths.out("gold", article), gold)
    for stage, value, flag in (("label_a", label, False), ("label_b", b_label or label, ambiguous)):
        write_json(
            paths.out(stage, article),  # type: ignore[arg-type]
            {
                "id": article,
                "label": value,
                "score": SCORES[value],
                "entity": f"{source} holdings",
                "ticker": "ABC" if stage == "label_b" else None,
                "rationale": f"The news points {value} for the company in our reading.",
                "ambiguous": flag,
                "confidence": 4,
            },
        )


def adjudicate(paths: ArticlePaths, article: str, decision: str, label: Sentiment) -> None:
    write_json(
        paths.out("adjudicate", article),
        {
            "id": article,
            "decision": decision,
            "label": label,
            "score": SCORES[label],
            "entity": "adjudicated entity",
            "ticker": "ADJ",
            "rationale": "The adjudicator weighed both readings.",
        },
    )


def ingest_all(paths: ArticlePaths) -> None:
    for stage in ("gold", "label_a", "label_b"):
        ingest(paths, stage)


def seed_pool(paths: ArticlePaths, per_cell: int = 4) -> None:
    for source in SOURCES:
        for label in SENTIMENTS:
            for n in range(per_cell):
                seed(paths, f"art-{source}{label}{n:02d}", source, label)
    # One disagreement per source, kept by the adjudicator, and one replaced.
    for source in SOURCES:
        seed(paths, f"art-{source}disputed", source, "bullish", b_label="neutral")
        seed(paths, f"art-{source}replaced", source, "bearish", ambiguous=True)
    ingest_all(paths)
    for source in SOURCES:
        adjudicate(paths, f"art-{source}disputed", "keep", "bullish")
        adjudicate(paths, f"art-{source}replaced", "replace", "bearish")
    ingest(paths, "adjudicate")


# --- ingest -----------------------------------------------------------------------------------------------


def test_gold_ingest_writes_bodies_and_drops_stale_ones(paths: ArticlePaths) -> None:
    seed(paths, "art-one", "alpha", "bullish")
    report = ingest(paths, "gold")
    assert [(r.id, r.ok) for r in report.results] == [("art-one", True)]
    assert paths.gold("art-one").read_text() == "\n\n".join(body_lines("art-one"))
    raw = json.loads(paths.out("gold", "art-one").read_text())
    write_json(paths.out("gold", "art-one"), raw | {"body_ranges": [[3, 99]]})
    report = ingest(paths, "gold")
    assert report.results[0].codes == ("range_out_of_bounds",)
    assert not paths.gold("art-one").exists()


def test_label_ingest_lists_disagreements_and_ambiguity(paths: ArticlePaths) -> None:
    seed(paths, "art-agree", "alpha", "bullish")
    seed(paths, "art-split", "alpha", "bullish", b_label="bearish")
    seed(paths, "art-unsure", "alpha", "neutral", ambiguous=True)
    ingest(paths, "gold")
    ingest(paths, "label_a")
    report = ingest(paths, "label_b")
    assert report.agreement is not None
    assert report.agreement.n == 3
    assert report.agreement.agreed == 2
    assert adjudicate_requested(paths) == {"art-split", "art-unsure"}


# --- quotas, selection, split -----------------------------------------------------------------------------


def avail(**overrides: tuple[int, int, int]) -> dict[str, dict[Sentiment, int]]:
    out = {s: dict(zip(SENTIMENTS, overrides.get(s, (5, 5, 5)), strict=True)) for s in SOURCES}
    return out


def test_quotas_default_to_three_each() -> None:
    quota = plan_quotas(avail(), SOURCES)
    assert all(q == 3 for per in quota.values() for q in per.values())


def test_quotas_rebalance_a_short_class_across_sources() -> None:
    quota = plan_quotas(avail(alpha=(5, 2, 5)), SOURCES)
    assert quota["alpha"]["bearish"] == 2
    assert quota["beta"]["bearish"] == 4
    assert {c: sum(quota[s][c] for s in SOURCES) for c in SENTIMENTS} == dict.fromkeys(SENTIMENTS, 9)


def test_quotas_prefer_sources_that_lost_a_slot() -> None:
    # alpha is short of bearish, beta of neutral: each makes up the other's class.
    quota = plan_quotas(avail(alpha=(5, 2, 5), beta=(5, 5, 2), gamma=(3, 3, 3)), SOURCES)
    assert quota["alpha"] == {"bullish": 3, "bearish": 2, "neutral": 4}
    assert quota["beta"] == {"bullish": 3, "bearish": 4, "neutral": 2}
    assert sum(quota["alpha"].values()) == sum(quota["beta"].values()) == 9


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alpha": (5, 1, 5)}, "alpha bearish: 1 eligible, at least 2 needed"),
        ({"alpha": (5, 2, 5), "beta": (5, 2, 5)}, "bearish: at most 8 of 9"),
        ({"alpha": (3, 3, 3), "beta": (3, 3, 3), "gamma": (2, 3, 3)}, "bullish: at most 8 of 9"),
    ],
)
def test_quotas_report_impossible_sets(overrides: dict[str, tuple[int, int, int]], message: str) -> None:
    with pytest.raises(ShortfallError) as exc:
        plan_quotas(avail(**overrides), SOURCES)
    assert any(message in problem for problem in exc.value.problems)


def test_choose_prefers_agreed_and_is_seeded() -> None:
    cell = {"art-a": "agreed", "art-b": "adjudicated", "art-c": "agreed", "art-d": "agreed"}
    picked = choose(cell, 3, seed=1, source="alpha", label="bullish")  # type: ignore[arg-type]
    assert sorted(picked) == ["art-a", "art-c", "art-d"]
    assert choose(cell, 4, seed=1, source="alpha", label="bullish")[-1] == "art-b"  # type: ignore[arg-type]
    assert picked == choose(cell, 3, seed=1, source="alpha", label="bullish")  # type: ignore[arg-type]


def test_split_cell_puts_exactly_one_in_dev() -> None:
    splits = split_cell(["art-a", "art-b", "art-c"], seed=7, source="alpha", label="bearish")
    assert sorted(splits.values()) == ["dev", "test", "test"]
    assert splits == split_cell(["art-c", "art-b", "art-a"], seed=7, source="alpha", label="bearish")


# --- finalize end to end ----------------------------------------------------------------------------------


def test_finalize_selects_splits_and_is_deterministic(paths: ArticlePaths) -> None:
    seed_pool(paths)
    first = finalize(build_ledger(paths), SOURCES, seed=11)
    again = finalize(build_ledger(paths), SOURCES, seed=11)
    assert first == again
    rows = first.rows
    assert len(rows) == 27
    assert first.stats["classes"] == dict.fromkeys(SENTIMENTS, 9)
    assert first.stats["splits"] == {"dev": 9, "test": 18}
    for source in SOURCES:
        for label in SENTIMENTS:
            cell = [r for r in rows if r.source == source and r.label == label]
            assert len(cell) == 3
            assert [r.split for r in cell].count("dev") == 1
    assert not any(r.id.endswith("replaced") for r in rows)
    pool = first.stats["pool"]
    assert isinstance(pool, dict)
    assert pool["adjudicated"] == 6
    assert pool["replaced"] == 3
    assert pool["sent_to_adjudication"] == 6
    other = finalize(build_ledger(paths), SOURCES, seed=12)
    assert [(r.id, r.split) for r in other.rows] != [(r.id, r.split) for r in rows]


def test_finalize_row_fields_follow_the_labels(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=3)
    # With one agreed article gone, the adjudicated one is needed to fill alpha's bullish cell.
    paths.fetched("art-alphabullish02").unlink()
    rows = {r.id: r for r in finalize(build_ledger(paths), SOURCES, seed=3).rows}
    disputed = rows["art-alphadisputed"]
    assert disputed.label == "bullish"
    assert disputed.entity == "adjudicated entity"
    assert disputed.labels.adjudicated is not None
    assert disputed.labels.adjudicated.decision == "keep"
    agreed = rows["art-alphabullish00"]
    assert agreed.score == 2.0
    assert agreed.ticker == "ABC"
    assert agreed.labels.adjudicated is None
    assert agreed.labels.final == "bullish"


def test_finalize_ignores_articles_outside_the_window(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=3)
    seed(paths, "art-old", "alpha", "bullish", published="2026-08-18")
    ingest_all(paths)
    assert "art-old" not in {r.id for r in finalize(build_ledger(paths), SOURCES).rows}


def test_finalize_fails_loudly_with_a_shortfall(paths: ArticlePaths) -> None:
    for source in SOURCES:
        for label in SENTIMENTS:
            seed(paths, f"art-{source}{label}", source, label)
    ingest_all(paths)
    with pytest.raises(ShortfallError, match="eligible, at least 2 needed"):
        finalize(build_ledger(paths), SOURCES)


def test_verify_and_loader_catch_drift(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=3)
    write_finalized(finalize(build_ledger(paths), SOURCES), paths)
    assert all(not problems for _, problems in verify(paths))
    dev = load_articles("dev", paths=paths)
    assert len(dev) == 9
    assert dev[0].gold_text == paths.gold(dev[0].id).read_text()
    assert len(load_articles("all", paths=paths)) == 27
    paths.gold(dev[0].id).write_text("edited")
    paths.html(dev[1].id).unlink()
    found = dict(verify(paths))
    assert found[dev[0].id] == ["gold_drifted"]
    assert found[dev[1].id] == ["html_missing"]
    with pytest.raises(ArticlesError, match="missing or drifted"):
        load_articles("dev", paths=paths)


# --- no publisher text --------------------------------------------------------------------------------------


def test_committed_output_carries_no_publisher_text(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=3)
    result = finalize(build_ledger(paths), SOURCES)
    write_finalized(result, paths)
    committed = paths.rows.read_text() + paths.stats.read_text()
    for row in result.rows:
        lines = json.loads(paths.lines(row.id).read_text())["lines"]
        assert lines[1] not in committed
        assert all(line not in committed for line in lines[2:])
    bodies = {r.id: paths.gold(r.id).read_text() for r in result.rows}
    titles = {r.id: json.loads(paths.lines(r.id).read_text())["lines"][1] for r in result.rows}
    assert publisher_text_problems(result.rows, bodies, titles) == []


def test_publisher_text_guard_flags_copied_text(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=3)
    row = finalize(build_ledger(paths), SOURCES).rows[0]
    body = paths.gold(row.id).read_text()
    title = "A distinctive invented headline about crates"
    copied = " ".join(body.split()[:12])
    bad = row.model_copy(update={"key_facts": [copied], "rationale": f"See: {title}"})
    problems = publisher_text_problems([bad], {row.id: body}, {row.id: title})
    assert f"{row.id}:key_facts[0]:verbatim_body" in problems
    assert f"{row.id}:rationale:headline" in problems


def test_committed_article_rows_are_publisher_free() -> None:
    """Runs on the real committed set once it exists; the schema forbids any title or body field."""
    path = ARTICLES_DIR / "articles.jsonl"
    if not path.exists():
        pytest.skip("article set not finalized yet")
    rows = read_jsonl(path, ArticleRow)
    assert rows
    real = ArticlePaths()
    for row in rows:
        assert all(len(fact.split()) <= 25 for fact in row.key_facts)
        assert len(row.rationale.split()) <= 50
        gold = real.gold(row.id)
        if gold.exists():
            assert publisher_text_problems([row], {row.id: gold.read_text()}, {}) == []


def test_cli_prints_ids_and_counts_only(
    paths: ArticlePaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    title = "Zebracorn Logistics posts quarterly surprise"
    seed(paths, "art-cliarticle", "cnbc", "bullish", title=title)
    monkeypatch.setattr(articles_cli, "ArticlePaths", lambda: paths)
    assert cli.main(["articles", "ingest", "--stage", "gold"]) == 0
    assert cli.main(["articles", "ingest", "--stage", "label_a"]) == 0
    assert cli.main(["articles", "prepare", "--stage", "label"]) == 0
    assert cli.main(["articles", "status", "--source", "cnbc"]) == 0
    out = capsys.readouterr().out
    assert "art-cliarticle PASS" in out
    assert "Zebracorn" not in out
    assert not shingles(out, 4) & shingles(" ".join(body_lines("art-cliarticle")), 4)
    assert title in paths.packet("label", "art-cliarticle").read_text()


def test_agreement_is_printed_only_on_request(
    paths: ArticlePaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seed(paths, "art-agreement", "cnbc", "bearish", b_label="neutral")
    monkeypatch.setattr(articles_cli, "ArticlePaths", lambda: paths)
    cli.main(["articles", "ingest", "--stage", "gold"])
    cli.main(["articles", "ingest", "--stage", "label_a"])
    assert "agreement:" not in capsys.readouterr().out
    assert adjudicate_requested(paths) == {"art-agreement"}
    cli.main(["articles", "ingest", "--stage", "label_b", "--agreement"])
    assert "agreement: pairs=1 agreed=0" in capsys.readouterr().out


def test_cli_rejects_unknown_sources(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["articles", "status", "--source", "nowhere"]) == 1
    assert "unknown or non-primary" in capsys.readouterr().err


# --- candidates and fetch -----------------------------------------------------------------------------------


def found(url: str, *, title: str | None = None, published: str | None = None, via: str = "sitemap") -> Found:
    return Found(
        Link(url, title, published),
        via,
        "https://example.com/sitemap.xml",
        "loss" if via == "search" else None,
    )


def test_select_found_applies_window_query_and_cap() -> None:
    from veritrade_evals.articles_candidates import SourceReport

    items = [
        found("https://example.com/2026/10/01/acme-widens-loss.html"),
        found("https://example.com/2026/07/01/acme-widens-loss-again.html"),
        found("https://example.com/b", title="Acme posts record profit", published="2026-10-02"),
        found("https://example.com/c", via="search"),
        found("https://example.com/d", title="Acme narrows losses"),
    ]
    report = SourceReport("wire")
    picked = select_found(
        report,
        items,
        since=date(2026, 8, 19),
        until=date(2026, 10, 18),
        queries=["loss"],
        max_new=2,
        known=set(),
    )
    assert [c.url for c in picked] == [
        "https://example.com/2026/10/01/acme-widens-loss.html",
        "https://example.com/c",
    ]
    assert picked[0].published_at == "2026-10-01"
    assert picked[1].query == "loss"
    assert (report.out_of_window, report.query_miss, report.capped) == (1, 1, 1)


def test_merge_keeps_order_and_fills_missing_fields() -> None:
    old = Candidate(id="art-1", source="s", url="https://e/1", via="feed", via_url="u")
    new = Candidate(
        id="art-1", source="s", url="https://e/1", title="T", published_at="2026-10-01", via="x", via_url="v"
    )
    other = Candidate(id="art-2", source="s", url="https://e/2", via="feed", via_url="u")
    merged, added = merge([old], [new, other])
    assert added == 1
    assert [c.id for c in merged] == ["art-1", "art-2"]
    assert merged[0].title == "T"
    assert merged[0].via == "feed"


def candidate(n: int, published: str | None) -> Candidate:
    url = f"https://www.cnbc.com/2026/10/0{n}/story-{n}.html"
    return Candidate(
        id=article_id(url), source="cnbc", url=url, published_at=published, via="feed", via_url="u"
    )


def test_view_and_pick_top_respect_the_window(paths: ArticlePaths) -> None:
    pool = [
        candidate(1, "2026-10-01"),
        candidate(2, None),
        candidate(3, "2026-07-01"),
        candidate(4, "2026-10-04"),
    ]
    from veritrade_evals.articles import write_jsonl

    write_jsonl(paths.candidates("cnbc"), pool)
    rows, total = write_view(paths, "cnbc", since=date(2026, 8, 19), until=date(2026, 10, 18))
    assert (rows, total) == (3, 4)
    assert paths.view("cnbc").read_text().splitlines()[0].startswith("id\t")
    top = pick_top(pool, paths, 3, since=date(2026, 8, 19), until=date(2026, 10, 18))
    assert [c.published_at for c in top] == ["2026-10-04", "2026-10-01", None]


ARTICLE_HTML = (
    '<html><head><script type="application/ld+json">{"@type": "NewsArticle", "headline": "Synthetic", '
    '"datePublished": "2026-10-05T10:00:00Z", "articleBody": "'
    + " ".join(["Synthetic body sentence about invented companies."] * 20)
    + '"}</script></head><body><p>Visible.</p></body></html>'
)


class FakeFetcher:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url: str) -> FetchResult:
        self.urls.append(url)
        if url.endswith("blocked.html"):
            from veritrade.errors import SourceBlocked

            raise SourceBlocked("blocked", upstream_status=403)
        return FetchResult(url, url, 200, ARTICLE_HTML, "text/html", 12.0)

    async def aclose(self) -> None:
        return None


class AllowAll(ArchiveClient):
    async def allowed(self, url: str) -> bool:
        return True


async def test_fetch_records_status_words_and_dates(
    paths: ArticlePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    from veritrade_evals import articles_fetch

    monkeypatch.setattr(articles_fetch, "Politeness", lambda: Politeness(0.0, 0.0))
    ok, blocked = candidate(5, "2026-10-04"), candidate(6, None)
    blocked = blocked.model_copy(update={"url": "https://www.cnbc.com/2026/10/06/blocked.html"})
    fetcher = FakeFetcher()
    robots = AllowAll(paths.cache)
    results = await fetch_articles([ok, blocked], paths=paths, fetcher=fetcher, robots=robots)
    good, bad = (record for _, record in results)
    assert good is not None
    assert good.ok
    assert good.method == "jsonld"
    assert good.words > 40
    assert good.published_date == "2026-10-05"
    assert good.published_from == "page"
    assert paths.html(ok.id).read_text() == ARTICLE_HTML
    assert bad is not None
    assert (bad.error, bad.status, bad.ok) == ("source_blocked", 403, False)
    again = await fetch_articles([ok], paths=paths, fetcher=fetcher, robots=robots)
    assert again[0][1] is None
    assert len(fetcher.urls) == 2
    await robots.aclose()


def test_status_counts_each_stage(paths: ArticlePaths) -> None:
    seed_pool(paths, per_cell=2)
    rows = {r.source: r for r in status(paths, SOURCES)}
    alpha = rows["alpha"]
    assert alpha.fetched_ok == 8
    assert alpha.gold_usable == {"bullish": 3, "bearish": 3, "neutral": 2}
    assert (alpha.labelled_a, alpha.labelled_b, alpha.agreed) == (8, 8, 6)
    assert (alpha.adjudicated_keep, alpha.adjudicated_replace, alpha.pending_adjudication) == (1, 1, 0)
    assert alpha.eligible == {"bullish": 3, "bearish": 2, "neutral": 2}


def _unused(_: Any) -> None:
    return None
