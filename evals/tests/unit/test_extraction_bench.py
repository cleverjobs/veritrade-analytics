from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from veritrade.errors import ExtractionEmpty, LlmUnreachable, SchemaViolation
from veritrade.extract import registry
from veritrade.extract.types import EXTRACTORS, PARSERS, Article, ExtractorName, ParserName
from veritrade_evals import extraction_bench as bench
from veritrade_evals.analysis import rounded
from veritrade_evals.extraction_bench import GOLD_ROW, Page, Row, SignalJob
from veritrade_evals.grid import GRID_SPLIT, grid_cells
from veritrade_evals.predcache import PredCache

FILLER = (
    "Ostrava Logistics said quarterly revenue rose to 41 million euros as freight volumes recovered "
    "across its central European network, and the board kept its outlook for the full year unchanged "
    "while it reviews two smaller acquisitions in the region."
)
BODY = [FILLER, "Analysts at Harwick expect margins to improve next year. " + FILLER]
HTML = (
    "<html><head><title>Ostrava Logistics results</title></head><body>"
    "<nav><a href='/'>Home</a> <a href='/m'>Markets</a></nav>"
    "<article><h1>Ostrava Logistics revenue rises</h1>"
    + "".join(f"<p>{p}</p>" for p in BODY)
    + "</article><footer>Subscribe to our newsletter for daily market updates and alerts</footer>"
    "</body></html>"
)


def page(html: str = HTML, article: str = "art-000000000001", source: str = "cnbc") -> Page:
    return Page(
        id=article,
        source=source,
        split="dev",
        url="https://example.com/news/ostrava-results",
        html=html,
        gold_text="\n\n".join(BODY),
        gold_title="Ostrava Logistics revenue rises",
        label="bullish",
    )


def article(paragraphs: Sequence[str], extractor: ExtractorName, parser: ParserName | None) -> Article:
    return Article(
        url="https://example.com/news/ostrava-results",
        title="Ostrava Logistics revenue rises",
        paragraphs=tuple(paragraphs),
        published_at=None,
        source=None,
        method=extractor,
        parser=parser,
        extractor=extractor,
        parse_ms=1.0 if parser else 0.0,
        extract_ms=1.0,
    )


# --- rows ------------------------------------------------------------------------------------


def test_matrix_has_every_registry_combination() -> None:
    rows = bench.matrix_rows()
    names = [r.name for r in rows]
    assert len(rows) == len(set(names)) == 22
    own = [r for r in rows if r.parser is not None]
    assert {(r.parser, r.extractor) for r in own} == {(p, e) for p in PARSERS for e in bench.OWN_EXTRACTORS}
    assert len(own) == 16
    third = [r for r in rows if r.parser is None]
    assert [r.extractor for r in third] == [e for e in EXTRACTORS if e not in bench.OWN_EXTRACTORS]
    assert bench.SERVED_ROW in names
    assert set(bench.EXTRACTOR_DISTS) == set(bench.THIRD_PARTY)
    assert set(bench.PARSER_DISTS) == set(PARSERS)


def test_select_rows_rejects_unknown_names() -> None:
    assert [r.name for r in bench.select_rows(["trafilatura", "lxml/density"])] == [
        "trafilatura",
        "lxml/density",
    ]
    assert len(bench.select_rows(None)) == 22
    with pytest.raises(bench.BenchError, match="unknown rows"):
        bench.select_rows(["selectolax/magic"])


# --- metrics ---------------------------------------------------------------------------------


def test_bag_of_words_scores_count_multiplicity() -> None:
    q = bench.quality("The cat sat. The cat!", "the cat sat on the mat")
    # the: min(2, 2) + cat: min(2, 1) + sat: 1 = 4 shared tokens out of 5 extracted and 6 gold.
    assert (q.extracted, q.gold, q.common) == (5, 6, 4)
    assert q.precision == pytest.approx(0.8)
    assert q.recall == pytest.approx(4 / 6)
    assert q.f1 == pytest.approx(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6))
    assert bench.word_tokens("Q3 EPS: $1.25, up 28%") == ["q3", "eps", "1", "25", "up", "28"]


def test_shingle_leakage_marks_tokens_outside_shared_windows() -> None:
    gold = ["a", "b", "c", "d", "e", "f", "g", "h"]
    extracted = ["a", "b", "c", "d", "e", "x", "y", "f", "g", "h"]
    # 3-grams abc, bcd, cde and fgh occur in the gold; only x and y sit in no shared window.
    assert bench.unsupported_tokens(extracted, gold, k=3) == 2
    cat = bench.word_tokens("The cat sat. The cat!")
    # Only "the cat sat" occurs in the gold with k=3; with the default k=5 nothing does.
    assert bench.unsupported_tokens(cat, bench.word_tokens("the cat sat on the mat"), k=3) == 2
    q = bench.quality("The cat sat. The cat!", "the cat sat on the mat")
    assert q.leaked == 5
    assert q.leakage == 1.0


def test_shingle_leakage_edge_cases() -> None:
    assert bench.unsupported_tokens([], ["a", "b"], k=5) == 0
    # Shorter than the window: the whole extract must occur in order.
    assert bench.unsupported_tokens(["b", "c"], ["a", "b", "c"], k=5) == 0
    assert bench.unsupported_tokens(["c", "b"], ["a", "b", "c"], k=5) == 2
    empty = bench.Quality(0, 10, 0, 0)
    assert (empty.precision, empty.recall, empty.f1, empty.leakage) == (0.0, 0.0, 0.0, 0.0)


# --- measure ---------------------------------------------------------------------------------


def test_measure_row_runs_the_serving_extractor_on_synthetic_html() -> None:
    p = page()
    strip = bench.measure_row(Row(None, "strip-tags"), p, runs=2, lede_budget=2000)
    assert strip["error"] is None
    assert len(strip["ms_runs"]) == 2
    assert strip["stable"] is True
    assert strip["parse_ms"] is None
    # Every visible block is kept, so the whole gold body is recovered along with the chrome around it.
    assert strip["tokens_common"] == strip["tokens_gold"]
    assert strip["tokens_leaked"] > 0
    assert strip["full_tokens"] > strip["lede_tokens"] > 0
    density = bench.measure_row(Row("lxml", "density"), p, runs=1, lede_budget=2000)
    assert density["error"] is None
    assert density["parse_ms"] is not None
    assert density["tokens_common"] / density["tokens_gold"] >= 0.9
    # The served title is the <title> tag here, not the h1 headline the gold uses.
    assert density["title_matches_gold"] is False
    assert not {"text", "title", "paragraphs", "html"} & set(strip)


def test_measure_records_failures_by_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract(html: str, url: str, *, parser: ParserName, extractor: ExtractorName) -> Article:
        if extractor == "jsonld":
            raise ExtractionEmpty("too short")
        if extractor == "density":
            raise ValueError("boom")
        return article(BODY, extractor, parser)

    monkeypatch.setattr(registry, "extract", fake_extract)
    p = page()
    empty = bench.measure_row(Row("selectolax", "jsonld"), p, runs=3, lede_budget=2000)
    crash = bench.measure_row(Row("selectolax", "density"), p, runs=1, lede_budget=2000)
    ok = bench.measure_row(Row("selectolax", "profile"), p, runs=1, lede_budget=2000)
    assert (empty["error"], crash["error"], ok["error"]) == ("ExtractionEmpty", "ValueError", None)
    assert len(empty["ms_runs"]) == 3
    assert empty["ms"] is not None
    assert "tokens_extracted" not in empty
    assert bench.quality_scores(empty) == (0.0, 0.0, 0.0)
    assert bench.quality_scores(ok) == pytest.approx((1.0, 1.0, 1.0))


def test_run_measure_resumes_and_writes_gold_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "ollama_resident", lambda base_url: ["some-model"])
    out = tmp_path / "measure.jsonl"
    rows = [Row(None, "strip-tags"), Row("selectolax", "density")]
    pages = [page(article="art-a"), page(article="art-b")]
    written = bench.run_measure(pages, rows[:1], runs=1, out=out, base_url="http://ollama.test")
    assert written == 4
    again = bench.run_measure(pages, rows, runs=1, out=out, base_url="http://ollama.test")
    assert again == 2
    keys = [(r["row"], r["id"]) for r in bench.read_records(out)]
    assert len(keys) == len(set(keys)) == 6
    assert keys.count((GOLD_ROW, "art-a")) == 1
    sessions = bench.sessions_for(out)
    assert sessions.name == "measure.sessions.json"
    assert "some-model" in sessions.read_text(encoding="utf-8")


# --- signal cache ----------------------------------------------------------------------------

PRED = {"kind": "decision", "choice": "bullish", "probs": {"bullish": 0.9, "neutral": 0.08, "bearish": 0.02}}


class FakeModel:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def __call__(self, text: str) -> tuple[dict[str, Any], dict[str, float]]:
        self.calls.append(text)
        if self.fail is not None:
            raise self.fail
        return dict(PRED), {"wall_ms": 1.0}


def job(row: str, text: str | None, error: str | None = None, article_id: str = "art-a") -> SignalJob:
    sha = bench.text_sha(text) if text is not None else None
    return SignalJob(bench.record_key(row, article_id), "bullish", sha, 10 if text else None, text, error)


def cache(root: Path) -> PredCache:
    return PredCache("model:test", {"bench": "extraction"}, model_digest="d", prompt_sha="p", root=root)


async def test_fill_calls_once_per_distinct_input(tmp_path: Path) -> None:
    jobs = [
        job(GOLD_ROW, "same lede"),
        job("lxml/cascade", "same lede"),
        job("trafilatura", "other lede"),
        job("selectolax/jsonld", None, "ExtractionEmpty"),
    ]
    model = FakeModel()
    with cache(tmp_path) as c:
        stats = await bench.fill(c, jobs, model)
        records = c.records()
    assert model.calls == ["same lede", "other lede"]
    assert (stats.calls, stats.reused, stats.extraction_failed, stats.cached) == (2, 1, 1, 0)
    assert records["lxml/cascade|art-a"]["reused_from"] == "gold|art-a"
    assert records["lxml/cascade|art-a"]["probs"] == PRED["probs"]
    assert records["selectolax/jsonld|art-a"]["error"]["code"] == "extraction_failed"
    assert records["selectolax/jsonld|art-a"]["extraction_error"] == "ExtractionEmpty"
    assert all("text" not in r and "input" not in r for r in records.values())

    with cache(tmp_path) as c:
        stats = await bench.fill(c, jobs, model)
    assert (stats.calls, stats.cached) == (0, 4)

    # A changed extraction re-keys only its own record; a lede that matches an answered one is reused.
    changed = [*jobs[:2], job("trafilatura", "same lede"), job("selectolax/jsonld", "new lede")]
    with cache(tmp_path) as c:
        stats = await bench.fill(c, changed, model)
        records = c.records()
    assert model.calls[2:] == ["new lede"]
    assert (stats.calls, stats.reused, stats.cached) == (1, 1, 2)
    assert records["trafilatura|art-a"]["reused_from"] == "gold|art-a"
    assert records["selectolax/jsonld|art-a"]["error"] is None


async def test_fill_records_model_errors_and_aborts_when_unreachable(tmp_path: Path) -> None:
    jobs = [job(GOLD_ROW, "lede one"), job("bs4/profile", "lede one")]
    with cache(tmp_path) as c:
        stats = await bench.fill(c, jobs, FakeModel(SchemaViolation("bad output")))
        records = c.records()
    assert stats.errors == 2
    assert records["gold|art-a"]["error"] == {
        "code": "schema_violation",
        "message": "SchemaViolation",
        "retryable": False,
    }
    with cache(tmp_path) as c:
        assert (await bench.fill(c, jobs, FakeModel())).cached == 2
    with cache(tmp_path / "other") as c, pytest.raises(LlmUnreachable):
        await bench.fill(c, jobs, FakeModel(LlmUnreachable("down")))


def test_signal_cache_key_is_separate_from_the_grid(tmp_path: Path) -> None:
    cell = next(c for c in grid_cells([bench.SIGNAL_MODEL]) if c.name == "v4")
    cfg = bench.SignalConfig(cell, 0.8)
    config = cfg.run_config()
    assert config["bench"] == "extraction"
    assert (config["scope"], config["title"], config["split"]) == ("lede", "gold", bench.SIGNAL_SPLIT)
    assert config["prompt_version"] == cell.prompt.version
    key = cfg.cache("digest-a", tmp_path).key
    assert key != cfg.cache("digest-b", tmp_path).key
    assert key != cell.cache(GRID_SPLIT, "digest-a", tmp_path).key
    # The threshold is applied offline, so moving it never invalidates cached answers.
    assert bench.SignalConfig(cell, 0.5).cache("digest-a", tmp_path).key == key
    assert bench.record_key("lxml/cascade", "art-1") == "lxml/cascade|art-1"


def test_signal_input_holds_the_title_at_the_gold_headline() -> None:
    p = page()
    extracted = article(BODY, "trafilatura", None)
    text = bench.signal_text(extracted, p.gold_title, 2000)
    assert text == bench.signal_text(p.gold_article, p.gold_title, 2000)
    other = bench.signal_text(article(BODY[1:], "trafilatura", None), p.gold_title, 2000)
    assert other != text
    assert other.startswith(f"TITLE: {p.gold_title}")


# --- report ----------------------------------------------------------------------------------


def test_verdicts_apply_the_operating_threshold() -> None:
    records: dict[str, dict[str, Any]] = {
        "a": {"error": None, "probs": {"bullish": 0.7, "neutral": 0.2, "bearish": 0.1}},
        "b": {"error": {"code": "extraction_failed"}, "probs": None},
    }
    assert bench.verdicts(records, 0.8) == {"a": "neutral", "b": None}
    assert bench.verdicts(records, 0.5) == {"a": "bullish", "b": None}


def test_signal_stats_count_misses_flips_and_abstentions() -> None:
    verdict: dict[str, Any] = {
        "gold|a": "bullish",
        "gold|b": "bearish",
        "gold|c": "neutral",
        "gold|d": "bullish",
        "gold|e": "bearish",
        "r|a": "bullish",
        "r|b": "bullish",
        "r|c": "bearish",
        "r|d": None,
        "r|e": "neutral",
    }
    stats, matches = bench.signal_stats("r", ["a", "b", "c", "d", "e", "f"], verdict)
    assert matches == [1.0, 0.0, 0.0, 0.0, 0.0, None]
    assert stats["signalAgreement"] == pytest.approx(1 / 5)
    assert stats["signalAgreementOk"] == pytest.approx(1 / 4)
    assert (stats["signalN"], stats["signalMissing"]) == (5, 1)
    assert (stats["signalFlips"], stats["signalLost"], stats["signalNew"]) == (1, 1, 1)


def test_identical_share_counts_failures_as_different() -> None:
    shas = {"gold|a": "x", "gold|b": "y", "gold|c": None, "r|a": "x", "r|b": "z", "r|c": "w", "r|d": "v"}
    # Article c has no gold input and d no gold record, so only a and b count.
    assert bench.identical_share("r", ["a", "b", "c", "d"], shas) == 0.5
    assert bench.identical_share("missing", ["a", "b"], shas) == 0.0
    assert bench.identical_share("r", ["a"], None) is None


def measured_record(
    row: str, article_id: str, source: str, *, error: str | None = None, ms: float = 2.0
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "row": row,
        "id": article_id,
        "source": source,
        "split": "dev",
        "error": error,
        "ms": ms,
    }
    if row == GOLD_ROW:
        return rec | {"ms": None, "tokens_gold": 100, "full_tokens": 160, "lede_tokens": 60}
    if error is None:
        rec |= {
            "method": "density",
            "tokens_extracted": 120,
            "tokens_gold": 100,
            "tokens_common": 90,
            "tokens_leaked": 24,
            "full_tokens": 190,
            "lede_tokens": 70,
            "title_matches_gold": True,
            "parse_ms": 1.0,
            "stable": True,
        }
    return rec


FRONTEND_KEYS = {
    "parser",
    "extractor",
    "n",
    "p50Ms",
    "p95Ms",
    "precision",
    "recall",
    "f1",
    "leakage",
    "tokensToLlm",
    "failureRate",
    "installMb",
}


def test_build_matrix_aggregates_rows_and_renders() -> None:
    rows = [Row("selectolax", "cascade"), Row(None, "justext")]
    measured = {}
    for article_id, source in (("a", "cnbc"), ("b", "sec")):
        for rec in (
            measured_record(GOLD_ROW, article_id, source),
            measured_record("selectolax/cascade", article_id, source, ms=3.0),
            measured_record(
                "justext", article_id, source, error="ExtractionEmpty" if source == "sec" else None
            ),
        ):
            measured[(rec["row"], rec["id"])] = rec
    install = {r.name: {"installMb": 1.5, "parserMb": 1.5, "extractorMb": 0.0} for r in rows}
    verdict: dict[str, Any] = {"gold|a": "bullish", "gold|b": "neutral", "selectolax/cascade|a": "bullish"}
    verdict |= {"selectolax/cascade|b": "neutral", "justext|a": "bearish", "justext|b": None}
    matrix = bench.build_matrix(
        measured,
        rows,
        verdict=verdict,
        signal_meta={"model": "m", "config": "v4", "threshold": 0.8},
        install=install,
        environment={"chip": "test", "ramGb": 1, "python": "3.13"},
        n_boot=50,
    )
    first, second = matrix["rows"]
    assert first["name"] == "selectolax/cascade"
    assert set(first) >= FRONTEND_KEYS
    assert first["f1"] == pytest.approx(2 * 0.75 * 0.9 / 1.65)
    assert first["leakage"] == pytest.approx(0.2)
    assert first["signalAgreement"] == 1.0
    assert second["failureRate"] == 0.5
    assert second["failures"] == {"ExtractionEmpty": 1}
    assert second["f1"] == pytest.approx(first["f1"] / 2)
    assert second["leakage"] == pytest.approx(0.2)
    assert second["signalAgreement"] == 0.0
    assert second["perSource"]["sec"]["failures"] == 1
    assert matrix["signal"]["goldVerdicts"] == {"bullish": 1, "neutral": 1}
    assert matrix["servedGap"]["best"] == "justext"
    assert matrix["servedGap"]["f1"]["point"] == pytest.approx(second["f1"] - first["f1"])
    summary = bench.render_summary(rounded(matrix))
    assert summary.startswith("# Extraction matrix")
    assert "| 1 | `selectolax` | `cascade` |" in summary
    assert chr(0x2014) not in summary


def test_build_matrix_reports_incomplete_rows() -> None:
    measured = {
        (GOLD_ROW, "a"): measured_record(GOLD_ROW, "a", "cnbc"),
        (GOLD_ROW, "b"): measured_record(GOLD_ROW, "b", "cnbc"),
        ("justext", "a"): measured_record("justext", "a", "cnbc"),
    }
    install = {"justext": {"installMb": 1.0, "parserMb": 0.0, "extractorMb": 1.0}}
    matrix = bench.build_matrix(
        measured, [Row(None, "justext")], verdict=None, signal_meta=None, install=install, environment={}
    )
    assert matrix["incomplete"] == ["justext (1)"]
    assert matrix["rows"][0]["signalAgreement"] is None
    assert matrix["servedGap"] == {}
    with pytest.raises(bench.BenchError, match="no measurements"):
        bench.build_matrix({}, [], verdict=None, signal_meta=None, install={}, environment={})


def test_install_sizes_count_only_what_a_row_adds() -> None:
    core = bench.app_core()
    assert "fastapi" in core
    assert not {"lxml", "selectolax", "beautifulsoup4", "trafilatura"} & core
    assert {"beautifulsoup4", "soupsieve"} <= bench.closure(["beautifulsoup4"])
    sizes = bench.install_sizes(bench.matrix_rows())
    for name in ("selectolax/cascade", "bs4-lxml/density"):
        assert sizes[name]["extractorMb"] == 0.0
        assert sizes[name]["installMb"] == sizes[name]["parserMb"] > 0
    assert sizes["bs4-lxml/cascade"]["installMb"] > sizes["bs4/cascade"]["installMb"]
    assert sizes["trafilatura"]["installMb"] == sizes["trafilatura-precision"]["installMb"] > 0
    assert sizes["strip-tags"]["installMb"] == sizes["selectolax/jsonld"]["installMb"]
