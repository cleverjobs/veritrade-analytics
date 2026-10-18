from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from veritrade.errors import ExtractionEmpty
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import ModelSpec
from veritrade_evals.articles import ArticleExample
from veritrade_evals.summary_eval import (
    COVERAGE_THRESHOLD,
    EvalArticle,
    GenRun,
    Pair,
    ScoreRow,
    SummaryEvalError,
    SummaryPaths,
    aggregate,
    best_matches,
    build_report,
    canonical_number,
    check_judgements,
    coverage,
    entity_aliases,
    generate,
    grounding,
    label_agreement,
    load_summaries,
    names_entity,
    parse_labels,
    parse_pair,
    prepare_packets,
    quality_ranking,
    render_markdown,
    score_split,
    sentences,
    summary_text,
    unit_rows,
    unsupported_numbers,
    was_capped,
    word_count,
    write_scores,
)
from veritrade_evals.summary_judge import Blinding, JudgeRow

BASE_URL = "http://ollama.test"
DIGEST = "sha256:" + "e" * 64
SUMMARY = Pair("summary", "qwen3.5:2b-q4_K_M")
ANALYZE = Pair("analyze", "ministral-3:3b")
BODY = (
    "Zentor Mills said second-quarter revenue climbed 12% to $1,410 million as demand recovered.\n\n"
    "The company lifted its full-year outlook and plans three new plants."
)
FACTS = ("Zentor Mills revenue climbed 12% to $1,410 million.", "Zentor Mills plans three new plants.")


def article(i: int, tmp_path: Path, **extra: Any) -> EvalArticle:
    html = tmp_path / f"art-{i:012d}.html"
    html.write_text("<html></html>", encoding="utf-8")
    example = ArticleExample(
        id=f"art-{i:012d}",
        source="alpha",
        url=f"https://alpha.example.com/{i}",
        label="bullish",
        score=1.0,
        entity=extra.get("entity", "Zentor Mills Inc."),
        ticker=extra.get("ticker", "ZNTR"),
        key_facts=extra.get("facts", FACTS),
        gold_text=extra.get("body", BODY),
        html_path=html,
        split="dev",
    )
    return EvalArticle(example, example.url, "Zentor Mills beats")


def served_text(_: EvalArticle, __: ModelSpec) -> str:
    return "TITLE: Zentor Mills beats\n\n" + BODY + "\n\nPublished 2026."


def fake_vector(text: str) -> list[float]:
    vec = [0.0] * 32
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
    return vec


def embed_route(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    return httpx.Response(200, json={"embeddings": [fake_vector(t) for t in inputs]})


def chat_body(content: str, done_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": "m",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "total_duration": 900_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 600,
        "prompt_eval_duration": 300_000_000,
        "eval_count": 40,
        "eval_duration": 500_000_000,
    }


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client() -> AsyncIterator[OllamaClient]:
    ollama = OllamaClient(BASE_URL, timeout=2.0, keep_alive="5m")
    yield ollama
    await ollama.aclose()


@pytest.fixture
def paths(tmp_path: Path) -> SummaryPaths:
    return SummaryPaths(private=tmp_path / "private", results=tmp_path / "results")


# --- numbers ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("1,500.0", "1500"), ("1.50", "1.5"), ("0857", "857"), ("100", "100"), ("1.2.3", "1.2.3")],
)
def test_canonical_number(raw: str, canonical: str) -> None:
    assert canonical_number(raw) == canonical


def test_numbers_ignore_percent_currency_and_separators() -> None:
    source = grounding(BODY)
    assert unsupported_numbers("Zentor Mills revenue rose 12 percent to $1410 million.", source) == []
    assert unsupported_numbers("Revenue rose 12.0% to US$1,410m.", source) == []
    assert unsupported_numbers("Revenue rose 15% to $1,410 million.", source) == ["15"]


def test_number_words_and_fiscal_periods_ground_their_digits() -> None:
    source = grounding(BODY + " First-half profit doubled.")
    assert unsupported_numbers("Q2 revenue rose 12%; it plans 3 plants and H1 profit doubled.", source) == []
    assert unsupported_numbers("Q3 revenue rose 12%.", source) == ["Q3"]
    assert unsupported_numbers("2Q revenue rose 12%.", source) == []
    assert (
        unsupported_numbers("Profit was RMB103.94 billion.", grounding("profit of RMB 103.94 billion")) == []
    )


def test_clock_times_match_the_bare_hour_but_not_other_minutes() -> None:
    source = grounding("A call follows at 5PM; trading resumed at 10:45 a.m. on day 20.")
    assert unsupported_numbers("The call is at 5:00 p.m. on day 20.", source) == []
    assert unsupported_numbers("Trading resumed at 10:45.", source) == []
    assert unsupported_numbers("Trading resumed at 10:54.", source) == ["10:54"]


def test_fiscal_years_calendar_quarters_and_scale_abbreviations() -> None:
    source = grounding(
        "Synergies flow in fiscal 2027. Record September-quarter revenue; traffic fell in the April to June "
        "period. It targets 80,000 to 81,000 homes and revenue of $2,400 million."
    )
    text = "FY27 synergies; Q3 revenue; Q2 traffic; 80K-81K homes; $2.4B revenue; FY2027 target."
    assert unsupported_numbers(text, source) == []
    assert unsupported_numbers("Q1 and Q4 revenue; 82K homes.", source) == ["82", "Q1", "Q4"]
    assert unsupported_numbers("Q3 output.", grounding("output in the June to August period")) == ["Q3"]


# --- entity -------------------------------------------------------------------------------------------------


def test_entity_aliases_cover_suffix_first_word_and_ticker() -> None:
    aliases = entity_aliases("Zentor Mills Inc.", "NYSE: ZNTR")
    assert aliases.names == {"zentor mills inc.", "zentor mills", "zentor"}
    assert aliases.tickers == {"ZNTR"}
    assert names_entity("Zentor's revenue rose.", aliases)
    assert names_entity("ZENTOR MILLS lifted its outlook.", aliases)
    assert names_entity("Shares of ZNTR rose.", aliases)
    assert not names_entity("Shares of znTR rose; Zentorx fell.", aliases)


def test_generic_first_words_and_exchange_suffixes() -> None:
    bank = entity_aliases("Bank of Harrow", None)
    assert "bank" not in bank.names
    assert not names_entity("The bank raised rates.", bank)
    assert names_entity("Bank of Harrow raised rates.", bank)
    listed = entity_aliases("Orvane Energy (Hong Kong)", "0857.HK")
    assert listed.tickers == {"0857.HK", "0857"}
    assert names_entity("Profit at 0857 rose.", listed)
    curly = entity_aliases("America's Car-Mart", "CRMT")
    assert names_entity("America’s Car-Mart shares fell.", curly)
    assert not names_entity("America's economy grew.", curly)


def test_descriptor_suffixes_and_scattered_words_name_the_entity() -> None:
    parks = entity_aliases("Pellow Parks Entertainment Inc.", "PPK")
    assert "pellow parks" in parks.names
    assert names_entity("Pellow Parks shares fell.", parks)
    economy = entity_aliases("UK economy", None)
    assert names_entity("The UK Chancellor said the economy's growth held.", economy)
    assert not names_entity("UK output grew 0.4%.", economy)
    assert not names_entity("Global shares rose.", entity_aliases("Global Parks Technologies", None))
    assert entity_aliases("Technologies", None).names == {"technologies"}


def test_sentence_and_word_counts_use_the_serving_splitter() -> None:
    text = "Shares of U.S. Steel rose 5% on Monday. Profit fell. Guidance held."
    assert len(sentences(text)) == 3
    assert word_count(text) == 12
    assert sentences("  ") == []


# --- coverage -----------------------------------------------------------------------------------------------


def test_best_matches_and_threshold() -> None:
    facts = unit_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sents = unit_rows([[1.0, 0.0, 0.0], [0.0, 0.75, (1 - 0.75**2) ** 0.5]])
    cosines, best = best_matches(facts, sents)
    assert best[:2] == [0, 1]
    assert cosines[0] == pytest.approx(1.0)
    assert cosines[1] == pytest.approx(0.75)
    assert coverage(cosines, 0.75) == pytest.approx(2 / 3)
    assert coverage(cosines) == 1.0
    assert coverage([COVERAGE_THRESHOLD - 0.0001]) == 0.0
    assert coverage([], COVERAGE_THRESHOLD) == 0.0
    assert best_matches(facts, unit_rows([])) == ([0.0, 0.0, 0.0], [-1, -1, -1])
    assert np.allclose(unit_rows([[3.0, 4.0], [0.0, 0.0]]), [[0.6, 0.8], [0.0, 0.0]])


# --- generation ---------------------------------------------------------------------------------------------


async def test_generate_records_summaries_errors_and_resumes(
    client: OllamaClient, mock_ollama: respx.MockRouter, paths: SummaryPaths, tmp_path: Path
) -> None:
    articles = [article(i, tmp_path) for i in range(3)]

    def inputs(item: EvalArticle, spec: ModelSpec) -> str:
        if item.id.endswith("2"):
            raise ExtractionEmpty("only 3 words")
        return served_text(item, spec)

    route = mock_ollama.post("/api/chat").respond(
        json=chat_body(json.dumps({"summary": "Zentor Mills revenue climbed 12%."}))
    )
    run = GenRun.resolve(SUMMARY)
    entry = await generate(client, run, articles, split="dev", digest=DIGEST, paths=paths, inputs=inputs)
    assert (entry.n, entry.done, entry.errors, entry.truncated) == (3, 2, 1, 0)
    records = load_summaries(paths, "dev", SUMMARY)
    assert records[articles[0].id]["summary"] == "Zentor Mills revenue climbed 12%."
    assert records[articles[0].id]["input_chars"] == len(served_text(articles[0], run.spec))
    assert (records[articles[0].id]["capped"], records[articles[0].id]["raw_words"]) == (False, 5)
    assert records[articles[2].id]["error"]["code"] == "extraction_empty"
    assert "text" not in records[articles[0].id]
    await generate(client, run, articles, split="dev", digest=DIGEST, paths=paths, inputs=inputs)
    assert route.call_count == 2


async def test_generate_stores_the_served_capped_summary(
    client: OllamaClient, mock_ollama: respx.MockRouter, paths: SummaryPaths, tmp_path: Path
) -> None:
    raw = "Zentor Mills grew. Revenue rose 12%. Plants opened. Costs fell."
    mock_ollama.post("/api/chat").respond(json=chat_body(json.dumps({"summary": raw})))
    articles = [article(0, tmp_path)]
    await generate(
        client, GenRun.resolve(SUMMARY), articles, split="dev", digest=DIGEST, paths=paths, inputs=served_text
    )
    record = load_summaries(paths, "dev", SUMMARY)[articles[0].id]
    assert record["summary"] == "Zentor Mills grew. Revenue rose 12%. Plants opened."
    assert (record["capped"], record["raw_words"], record["truncated"]) == (True, 10, False)


def test_cached_generations_are_scored_as_served() -> None:
    raw = {"summary": "One. Two. Three. Four.", "error": None}
    assert summary_text(raw) == "One. Two. Three."
    assert was_capped(raw)
    assert not was_capped({"summary": "One. Two.", "error": None})
    assert was_capped({"summary": "One. Two.", "capped": True, "error": None})
    assert summary_text({"error": {"code": "schema_violation"}}) is None


async def test_generate_analyze_keeps_score_and_salvages_cut_output(
    client: OllamaClient, mock_ollama: respx.MockRouter, paths: SummaryPaths, tmp_path: Path
) -> None:
    cut = '{"score": 1, "entity": "Zentor Mills", "summary": "Zentor Mills raised its outlook. It also pl'
    mock_ollama.post("/api/chat").respond(json=chat_body(cut, done_reason="length"))
    run = GenRun.resolve(ANALYZE)
    articles = [article(0, tmp_path)]
    entry = await generate(client, run, articles, split="dev", digest=DIGEST, paths=paths, inputs=served_text)
    assert entry.truncated == 1
    record = load_summaries(paths, "dev", ANALYZE)[articles[0].id]
    assert record["summary"] == "Zentor Mills raised its outlook."
    assert record["score"] == 1
    assert record["truncated"] is True


def test_load_summaries_needs_a_matching_generate_run(paths: SummaryPaths) -> None:
    with pytest.raises(SummaryEvalError, match="run `generate` first"):
        load_summaries(paths, "dev", SUMMARY)


def test_cache_key_follows_prompt_and_endpoint() -> None:
    summary, analyze = (
        GenRun.resolve(Pair("summary", "qwen3.5:4b-q4_K_M")),
        GenRun.resolve(Pair("analyze", "qwen3.5:4b-q4_K_M")),
    )
    assert summary.prompt_sha() != analyze.prompt_sha()
    assert summary.run_config("dev")["options"]["temperature"] == 0
    with pytest.raises(SummaryEvalError, match="not allowlisted"):
        GenRun.resolve(Pair("summary", "ministral-3:3b"))


def test_parse_pair() -> None:
    assert parse_pair("analyze/qwen3:4b-instruct-2507-q4_K_M") == Pair(
        "analyze", "qwen3:4b-instruct-2507-q4_K_M"
    )
    for bad in ("qwen3.5:2b-q4_K_M", "signal/qwen3.5:0.8b-q8_0", "summary/"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_pair(bad)


# --- scoring and report -------------------------------------------------------------------------------------


async def test_score_split_checks_and_embeds_once_per_article(
    client: OllamaClient, mock_ollama: respx.MockRouter, tmp_path: Path
) -> None:
    route = mock_ollama.post("/api/embed").mock(side_effect=embed_route)
    articles = [article(0, tmp_path), article(1, tmp_path)]
    good = (
        "Zentor Mills revenue climbed 12% to $1,410 million. Zentor Mills plans three new plants. "
        "Demand recovered. Costs rose sharply in every region."
    )
    summaries: dict[Pair, dict[str, dict[str, Any]]] = {
        SUMMARY: {
            articles[0].id: {"id": articles[0].id, "summary": good, "truncated": False, "error": None},
            articles[1].id: {"id": articles[1].id, "error": {"code": "schema_violation"}},
        },
        ANALYZE: {
            articles[0].id: {"id": articles[0].id, "summary": "Revenue rose 2026% in 2026.", "error": None},
        },
    }
    rows = await score_split(client, articles, summaries, inputs=served_text)
    assert route.call_count == 1
    by_key = {(r.id, r.candidate): r for r in rows}
    best = by_key[(articles[0].id, SUMMARY.key)]
    assert best.passed
    assert (best.capped, best.sentences) == (True, 3)
    assert best.fact_cosines == pytest.approx([1.0, 1.0])
    assert best.fact_best_sentence == [0, 1]
    weak = by_key[(articles[0].id, ANALYZE.key)]
    assert weak.unsupported_numbers == ["2026"]
    assert weak.unsupported_in_input == 1
    assert not weak.entity_named
    assert not weak.passed
    assert by_key[(articles[1].id, SUMMARY.key)].error == "schema_violation"
    assert by_key[(articles[1].id, ANALYZE.key)].error == "missing"


def test_aggregate_counts_every_article_and_gates() -> None:
    ids = ["art-a", "art-b", "art-c", "art-d"]
    ok = ScoreRow(
        id="art-a", candidate=SUMMARY.key, words=20, sentences=2, entity_named=True, fact_cosines=[0.9, 0.5]
    )
    long = ok.model_copy(update={"id": "art-b", "words": 70, "fact_cosines": [0.65, 0.65], "capped": True})
    cut = ok.model_copy(update={"id": "art-c", "truncated": True, "fact_cosines": [0.8, 0.8]})
    judged = [
        JudgeRow(id="art-a", candidate=SUMMARY.key, faithfulness=5, coverage=4, hallucinations=0, leads=True),
        JudgeRow(
            id="art-b", candidate=SUMMARY.key, faithfulness=3, coverage=3, hallucinations=2, leads=False
        ),
        JudgeRow(
            id="art-x", candidate=SUMMARY.key, faithfulness=1, coverage=1, hallucinations=1, leads=False
        ),
    ]
    row = aggregate(SUMMARY, [ok, long, cut], ids, judged)
    assert row["n"] == 4
    assert row["deterministicPassRate"] == 0.25
    assert row["wordsOkRate"] == 0.5
    assert row["truncatedRate"] == 0.25
    assert row["errorRate"] == 0.25
    assert row["cappedRate"] == 0.25
    assert row["keyFactCoverage"] == pytest.approx((0.5 + 1.0 + 1.0 + 0.0) / 4)
    assert row["keyFactCoverageStrict"] == pytest.approx((0.5 + 0.0 + 1.0 + 0.0) / 4)
    assert row["keyFactCoverageAt"]["0.70"] == pytest.approx((0.5 + 1.0) / 4)
    assert row["medianWords"] == 20
    assert (row["judgeFaithfulness"], row["judgeCoverage"], row["hallucinationRate"]) == (4.0, 3.5, 0.5)
    assert row["passesGate"] is False
    empty = aggregate(ANALYZE, [], ids)
    assert empty["judgeFaithfulness"] is None
    assert empty["errorRate"] == 1.0


def test_quality_ranking_orders_gated_candidates_by_judge() -> None:
    rows = [
        {
            "endpoint": "summary",
            "model": "a",
            "passesGate": True,
            "judgeFaithfulness": 4.5,
            "judgeCoverage": 3.0,
        },
        {
            "endpoint": "summary",
            "model": "b",
            "passesGate": True,
            "judgeFaithfulness": 4.5,
            "judgeCoverage": 3.5,
        },
        {
            "endpoint": "summary",
            "model": "c",
            "passesGate": False,
            "judgeFaithfulness": 5.0,
            "judgeCoverage": 5.0,
        },
        {
            "endpoint": "analyze",
            "model": "d",
            "passesGate": True,
            "judgeFaithfulness": None,
            "judgeCoverage": None,
        },
    ]
    assert quality_ranking(rows) == {"summary": ["b", "a"], "analyze": ["d"]}


def test_report_and_packets_carry_no_summary_text(paths: SummaryPaths, tmp_path: Path) -> None:
    articles = [article(0, tmp_path), article(1, tmp_path)]
    secret = "Zentor Mills secretly doubled revenue to 999 widgets."
    summaries = {
        SUMMARY: {a.id: {"id": a.id, "summary": secret, "error": None} for a in articles},
        ANALYZE: {articles[0].id: {"id": articles[0].id, "summary": "Zentor Mills grew.", "error": None}},
    }
    rows = [
        ScoreRow(
            id=a.id, candidate=pair.key, words=8, sentences=1, entity_named=True, fact_cosines=[0.9, 0.2]
        )
        for a in articles
        for pair in (SUMMARY, ANALYZE)
    ]
    write_scores(paths, "dev", rows, {SUMMARY.key, ANALYZE.key})
    written = prepare_packets("dev", articles, summaries, paths.judge)
    assert len(written) == 2
    blinding = Blinding.model_validate_json(paths.judge.blinding(articles[0].id).read_text())
    assert len(blinding.labels) == 2
    labels = {key: label for label, key in blinding.labels.items()}
    verdicts = [
        {
            "label": labels[SUMMARY.key],
            "faithfulness": 2,
            "coverage": 2,
            "hallucinations": ["invents a count"],
        },
        {"label": labels[ANALYZE.key], "faithfulness": 5, "coverage": 2, "hallucinations": []},
    ]
    for verdict in verdicts:
        verdict["leads_with_market_moving_fact"] = True
    paths.judge.out(articles[0].id).parent.mkdir(parents=True)
    paths.judge.out(articles[0].id).write_text(json.dumps({"id": articles[0].id, "summaries": verdicts}))
    results = dict(
        (i, (codes, judged)) for i, codes, judged in check_judgements(articles, summaries, paths.judge)
    )
    assert results[articles[0].id][0] == ()
    assert results[articles[1].id][0] == ("missing_output",)
    judged_rows = results[articles[0].id][1]
    assert {r.candidate: r.faithfulness for r in judged_rows} == {SUMMARY.key: 2, ANALYZE.key: 5}
    paths.judge.ingested("dev").parent.mkdir(parents=True)
    paths.judge.ingested("dev").write_text("".join(r.model_dump_json() + "\n" for r in judged_rows))
    block = build_report("dev", [SUMMARY, ANALYZE], [a.id for a in articles], paths)
    report = {"config": {}, "splits": {"dev": block}}
    committed = json.dumps(report) + render_markdown(report)
    assert secret not in committed
    assert "999" not in committed
    assert chr(0x2014) not in committed
    assert block["rows"][0]["judgeFaithfulness"] == 2.0
    assert block["judged"] is True


def test_hand_labels_give_agreement_per_threshold() -> None:
    sample = [{"id": "art-a", "cosines": [0.9, 0.62, 0.4]}, {"id": "art-b", "cosines": [0.7, 0.58]}]
    stated = parse_labels("110,11", sample)
    assert stated == [[True, True, False], [True, True]]
    for entry, flags in zip(sample, stated, strict=True):
        entry["stated"] = flags
    agreement = label_agreement(sample, thresholds=(0.60, 0.75))
    assert agreement["0.60"] == {"right": 4, "falseCovered": 0, "falseNotCovered": 1}
    assert agreement["0.75"] == {"right": 2, "falseCovered": 0, "falseNotCovered": 3}
    for bad in ("110", "11,11", "1x0,11"):
        with pytest.raises(SummaryEvalError):
            parse_labels(bad, sample)
