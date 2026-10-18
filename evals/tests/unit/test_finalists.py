from __future__ import annotations

from pathlib import Path
from typing import Any

from veritrade.api.contracts import AnalyzeRequest, SignalRequest
from veritrade.extract.types import Article
from veritrade.inference.types import Sentiment
from veritrade_evals import finalists, grid
from veritrade_evals.analysis import (
    CellInput,
    DevSet,
    best_candidate,
    candidate_cis,
    candidates,
    chosen_point,
    rounded,
)
from veritrade_evals.articles import ArticleExample
from veritrade_evals.finalists import Dev300Point, Extracted, Finalist

GOLD: tuple[Sentiment, ...] = ("bullish", "bullish", "bearish", "bearish", "neutral", "neutral")
DEV = DevSet(tuple(f"art-{i}" for i in range(6)), GOLD, None, buckets=("price move",) * 3 + ("other",) * 3)
MODEL = "qwen3.5:0.8b-q8_0"


def test_inputs_follow_the_serving_request_defaults() -> None:
    assert SignalRequest.model_fields["parser"].default == finalists.PARSER
    assert SignalRequest.model_fields["extractor"].default == finalists.EXTRACTOR
    assert SignalRequest.model_fields["scope"].default in finalists.SIGNAL_SCOPES
    assert {f.scope for f in finalists.FINALISTS if f.model in finalists.ANALYZE_FINALISTS} == {
        AnalyzeRequest.model_fields["scope"].default
    }


def test_finalists_are_grid_cells_grouped_by_model() -> None:
    cells = finalists.finalist_cells()
    assert len(cells) == len(finalists.FINALISTS) == 14
    for finalist, cell in cells:
        assert cell.name == finalist.config
        assert cell.endpoint == ("signal" if finalist.model in finalists.SIGNAL_FINALISTS else "analyze")
    order = [cell.model for _, cell in cells]
    assert order == sorted(order, key=list(dict.fromkeys(order)).index)
    assert finalists.CACHE_SPLIT != grid.GRID_SPLIT


def _example(i: int, label: Sentiment) -> ArticleExample:
    return ArticleExample(
        id=f"art-{i}",
        source="bbc",
        url=f"https://example.test/{i}",
        label=label,
        score=0.0,
        entity="Acme",
        ticker=None,
        key_facts=(),
        gold_text="",
        html_path=Path("unused.html"),
        split="dev",
    )


def _article(title: str, paragraphs: tuple[str, ...]) -> Article:
    return Article(
        url="https://example.test",
        title=title,
        paragraphs=paragraphs,
        published_at=None,
        source=None,
        method="cascade",
        parser="selectolax",
        extractor="cascade",
        parse_ms=0.0,
        extract_ms=0.0,
    )


def test_inputs_carry_an_identity_that_re_keys_the_cache() -> None:
    body = tuple(f"Paragraph {i} about Acme results and the outlook for next year." for i in range(12))
    extracted = [
        Extracted(_example(0, "bullish"), _article("Acme beats", body)),
        Extracted(_example(1, "neutral"), None, "extraction_empty"),
    ]
    lede, lede_id = finalists.inputs_for(extracted, "lede", 3000)
    full, full_id = finalists.inputs_for(extracted, "full", 3000)
    assert [i.id for i in lede] == [i.id for i in full] == ["art-0"]
    assert lede[0].text.startswith("TITLE: Acme beats")
    assert len(lede[0].text) < len(full[0].text)
    assert lede_id["sha"] != full_id["sha"]
    assert {k: lede_id[k] for k in ("articles", "parser", "extractor", "scope")} == {
        "articles": "dev",
        "parser": "selectolax",
        "extractor": "cascade",
        "scope": "lede",
    }
    edited = [Extracted(extracted[0].example, _article("Acme misses", body))]
    assert finalists.inputs_for(edited, "lede", 3000)[1]["sha"] != lede_id["sha"]


def _label(bull: float, bear: float) -> dict[str, Any]:
    probs = {"bullish": bull, "bearish": bear, "neutral": 1 - bull - bear}
    return {"kind": "label", "fallback": "none", "label": "neutral", "probs": probs, "t": {"wall_ms": 10.0}}


def _cell(name: str, records: list[dict[str, Any]]) -> CellInput:
    return CellInput("signal", MODEL, name, "label", f"signal/{MODEL}/v2@0", tuple(records))


def _dev300(cell: CellInput) -> dict[tuple[str, str, str], Dev300Point]:
    """A neutral-leaning model whose dev300 best is the tau rule."""
    cands = candidates([cell], DEV)
    best, _ = best_candidate(cands)
    out: dict[tuple[str, str, str], Dev300Point] = {}
    for c in cands:
        point = chosen_point(c, c.evaluation.operating is not None)
        cis = candidate_cis(c, point, DEV, 20)
        out[(MODEL, "v2-label", c.evaluation.policy)] = Dev300Point(c, point, cis, c is best)
    return out


LEANING = [
    _label(0.4, 0),
    _label(0.2, 0),
    _label(0, 0.45),
    _label(0.1, 0.3),
    _label(0.05, 0.05),
    _label(0.3, 0),
]


def test_articles_are_scored_at_the_dev300_thresholds_and_checked_for_clear_failures() -> None:
    d300 = _dev300(_cell("v2-label", LEANING))
    tau = d300[(MODEL, "v2-label", "label-logprob+tau")]
    assert tau.best
    # On the articles the lede reads like dev300; the full body flips every directional call.
    inverted = [_label(0, 0.3), _label(0, 0.3), _label(0.3, 0), _label(0.3, 0), _label(0, 0), _label(0, 0)]
    cells = [
        (Finalist(MODEL, "v2-label", "lede"), _cell("v2-label lede", LEANING)),
        (Finalist(MODEL, "v2-label", "full"), _cell("v2-label full", inverted)),
    ]
    scored = finalists.score_cells(cells, DEV, d300)
    assert [(s.finalist.scope, s.policy, s.served) for s in scored] == [
        ("lede", "label-logprob", False),
        ("lede", "label-logprob+tau", True),
        ("full", "label-logprob", False),
        ("full", "label-logprob+tau", True),
    ]
    assert all(s.point.threshold == d300[(MODEL, "v2-label", s.policy)].point.threshold for s in scored)

    rows = {(r["scope"], r["policy"]): r for r in (finalists.row_report(s, DEV, 50) for s in scored)}
    lede = rows[("lede", "label-logprob+tau")]
    assert lede["articles"]["f05_macro"] == tau.point.f05_macro
    assert lede["clear_failure"] == []
    full = rows[("full", "label-logprob+tau")]
    assert (full["articles"]["inversions"], full["articles"]["directional_calls"]) == (4, 4)
    assert "inversion CI low" in full["clear_failure"][0]
    assert full["repick"] is None
    # Plain rules are reported but never checked: they are not what the model serves.
    assert rows[("full", "label-logprob")]["clear_failure"] == []

    what = [c["comparison"] for c in finalists.comparisons(scored, DEV, 50)]
    assert what == ["scope: lede - full", "rule: tau - plain", "rule: tau - plain"]


def test_misses_and_markdown_use_the_served_rule_at_the_served_scope() -> None:
    d300 = _dev300(_cell("v2-label", LEANING))
    cells = [
        (Finalist(MODEL, "v2-label", scope), _cell(f"v2-label {scope}", LEANING))
        for scope in ("lede", "full")
    ]
    scored = finalists.score_cells(cells, DEV, d300)
    extracted = [Extracted(_example(i, g), None) for i, g in enumerate(GOLD)]
    misses = finalists.misses(scored, DEV, extracted)
    assert list(misses["by_title_bucket"]["per_model"]) == [f"{MODEL}/v2-label/lede"]
    assert misses["by_source"]["bbc"]["n"] == 6
    report = rounded(
        {
            "n": 6,
            "gold": {"bullish": 2, "bearish": 2, "neutral": 2},
            "extraction": {"parser": "selectolax", "extractor": "cascade", "failures": 0},
            "inversion_cap": 0.02,
            "collapse_share": 0.5,
            "n_boot": 50,
            "rows": [finalists.row_report(s, DEV, 50) for s in scored],
            "comparisons": finalists.comparisons(scored, DEV, 50),
            "misses": misses,
        }
    )
    md = finalists.render(report, {"ollama_version": "0.40.6", "backend": "native", "cells": {}})
    for heading in ("## /signal finalists", "## Paired comparisons", "## Clear-failure check", "## Misses"):
        assert heading in md
    assert "No served rule clearly fails" in md
