from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from veritrade.inference.types import Sentiment
from veritrade_evals.analysis import (
    PROB_THRESHOLDS,
    CellInput,
    DevSet,
    Record,
    Served,
    best_candidate,
    bucket_of,
    candidates,
    chosen_point,
    counts,
    diagnostics,
    directional_mass,
    directional_mass_report,
    error_patterns,
    evaluate_policy,
    knob_comparison,
    latency,
    replay,
    rounded,
    served_calls,
    thin_sweep,
)
from veritrade_evals.runner import Variant

GOLD: tuple[Sentiment, ...] = ("bullish", "bullish", "bearish", "bearish", "neutral", "neutral")
DEV = DevSet(ids=tuple(f"x:{i}" for i in range(6)), gold=GOLD, weights=None)


def decision(bull: float, bear: float, neutral: float) -> dict[str, Any]:
    return {
        "kind": "decision",
        "choice": "neutral",
        "probs": {"bullish": bull, "bearish": bear, "neutral": neutral},
    }


def ordinal(score: int, sign_probs: dict[str, float] | None = None) -> dict[str, Any]:
    return {"kind": "ordinal", "score": score, "sign_probs": sign_probs}


def analyze(score: int) -> dict[str, Any]:
    return {"kind": "analyze", "score": score}


def failed() -> dict[str, Any]:
    return {"error": {"code": "schema_violation", "message": "x", "retryable": False}}


def make_cell(variant: Variant, records: Sequence[Record | None], name: str = "v1", **kw: Any) -> CellInput:
    return CellInput("signal", "m", name, variant, f"signal/m/{name}@0", tuple(records), **kw)


# Exact binary fractions: e1-e3 right, e4 inverts (bull on gold bear), e5 false alarm, e6 neutral.
DECISION_RECORDS = [
    decision(0.875, 0.0625, 0.0625),
    decision(0.625, 0.125, 0.25),
    decision(0.0625, 0.8125, 0.125),
    decision(0.5625, 0.25, 0.1875),
    decision(0.0625, 0.6875, 0.25),
    decision(0.0625, 0.0625, 0.875),
]


def test_replay_per_policy() -> None:
    assert replay("probability", DECISION_RECORDS[3]) == ("bullish", 0.5625)
    assert replay("ordinal", analyze(-2)) == ("bearish", 2.0)
    assert replay("ordinal", analyze(0)) == ("neutral", math.inf)
    top, p = replay("label-logprob", ordinal(1, {"bullish": 0.6, "neutral": 0.4, "bearish": 0.0}))
    assert top == "bullish"
    assert p == pytest.approx(0.6)
    label = {"kind": "label", "probs": {"bullish": 0.2, "bearish": 0.3, "neutral": 0.5}, "fallback": "none"}
    top, p = replay("label-logprob", label)
    assert top == "neutral"
    assert p == pytest.approx(0.5)
    # Without probabilities the serving chain applies the ordinal policy with the frozen t.
    fallback = {"kind": "label", "probs": None, "fallback": "ordinal", "score": 1}
    assert replay("label-logprob", fallback, ordinal_t=1) == ("bullish", math.inf)
    assert replay("label-logprob", fallback, ordinal_t=2) == ("bullish", -math.inf)
    assert replay("label-logprob", ordinal(-2, None)) == ("bearish", math.inf)
    assert replay("probability", None) == ("neutral", -math.inf)
    assert replay("ordinal", failed()) == ("neutral", -math.inf)
    with pytest.raises(ValueError, match="no probabilities"):
        replay("probability", {"kind": "decision", "probs": None})


def test_probability_sweep_hand_computed() -> None:
    ev = evaluate_policy("probability", DECISION_RECORDS, GOLD, None)
    by_t = {p.threshold: p for p in ev.sweep}
    # t=0: bull calls e1 e2 e4 (P 2/3, R 1 -> F 5/7), bear calls e3 e5 (P 1/2, R 1/2 -> 1/2).
    assert by_t[0.0].f05_macro == pytest.approx((5 / 7 + 0.5) / 2)
    assert by_t[0.0].inversion_rate == pytest.approx(1 / 5)
    # 0.57 drops the inversion; 0.63 drops a right call; 0.69 drops the false alarm.
    assert by_t[0.57].f05_macro == pytest.approx(0.75)
    assert by_t[0.57].inversion_rate == 0.0
    assert by_t[0.63].f05_macro == pytest.approx((1.25 * 0.5 / 0.75 + 0.5) / 2)
    assert by_t[0.69].f05_macro == pytest.approx(1.25 * 0.5 / 0.75)
    assert by_t[0.69].coverage == pytest.approx(0.5)
    op = ev.operating
    assert op is not None
    assert op.threshold == 0.69
    assert ev.default.threshold == 0.0
    assert ev.abstained(0.69) == [False, True, False, True, True, False]


def test_weights_flow_into_precision() -> None:
    weighted = evaluate_policy("probability", DECISION_RECORDS, GOLD, [1, 1, 1, 1, 2, 2])
    # The false alarm on a neutral now weighs 2: bear P = 1/3, R = 1/2 -> F0.5 = 5/14.
    assert weighted.default.f05_macro == pytest.approx((5 / 7 + 5 / 14) / 2)


def test_ordinal_thresholds_hand_computed() -> None:
    ev = evaluate_policy("ordinal", [analyze(s) for s in (2, 1, -2, 1, 0, -1)], GOLD, None)
    assert [p.threshold for p in ev.sweep] == [1.0, 2.0]
    t1, t2 = ev.sweep
    assert t1.f05_macro == pytest.approx((5 / 7 + 0.5) / 2)
    assert t1.inversion_rate == pytest.approx(1 / 5)
    assert t2.f05_macro == pytest.approx(1.25 * 0.5 / 0.75)
    # The zero score stays a confident neutral: e1, e3 and e5 answer.
    assert t2.coverage == pytest.approx(0.5)
    assert ev.operating is t2


def test_best_candidate_needs_the_inversion_cap() -> None:
    # Cell a: higher F0.5 but an inversion survives every threshold; cell b: fewer calls, none wrong.
    a = make_cell("sentiment-first", [analyze(s) for s in (2, 2, -2, 2, 0, 0)], "a")
    b = make_cell("sentiment-first", [analyze(s) for s in (2, 0, -2, 0, 0, 0)], "b")
    best, cap_met = best_candidate(candidates([a, b], DEV))
    assert (best.cell.name, cap_met) == ("b", True)
    assert chosen_point(best, cap_met).threshold == 1.0

    only_a, cap_met = best_candidate(candidates([a], DEV))
    assert (only_a.cell.name, cap_met) == ("a", False)
    assert chosen_point(only_a, cap_met) is only_a.evaluation.default


def test_exact_ties_go_to_the_earlier_config() -> None:
    same = [analyze(s) for s in (2, 0, -2, 0, 0, 0)]
    best, _ = best_candidate(
        candidates([make_cell("sentiment-first", same, "v1"), make_cell("sentiment-first", same, "v2")], DEV)
    )
    assert best.cell.name == "v1"


def test_each_variant_is_scored_by_its_serving_policy() -> None:
    sign = {"bullish": 0.9, "neutral": 0.1, "bearish": 0.0}
    cells = [
        make_cell("decision", DECISION_RECORDS),
        make_cell("label", [{"kind": "label", "fallback": "none", "label": "bullish", "probs": sign}] * 6),
        make_cell("ordinal", [ordinal(1, sign)] * 6),
        make_cell("sentiment-first", [analyze(1)] * 6),
    ]
    cands = candidates(cells, DEV)
    assert [c.evaluation.policy for c in cands] == ["probability", "label-logprob", "ordinal", "ordinal"]
    assert [c.reads for c in cands] == ["class probabilities", "label tokens", "score value", "score value"]


def test_knob_comparison_prefers_the_cleaner_knob() -> None:
    classes: list[Sentiment] = ["bullish", "bearish", "neutral"]
    gold = tuple(c for c in classes for _ in range(20))
    dev = DevSet(tuple(f"x:{i}" for i in range(60)), gold, None)
    sure = {"bullish": 0.9, "bearish": 0.05, "neutral": 0.05}
    label = [
        {"kind": "label", "fallback": "none", "label": g, "probs": {**{k: 0.05 for k in sure}, g: 0.9}}
        for g in gold
    ]
    # The ordinal cell calls every neutral bullish with |score| 2, so no t removes the false alarms.
    scores = [2] * 20 + [-2] * 20 + [2] * 20
    ords = [ordinal(s, {"bullish": 1.0, "bearish": 0.0, "neutral": 0.0} if s > 0 else None) for s in scores]
    cells = [make_cell("label", label, "v1-label"), make_cell("ordinal", ords, "v1-ordinal")]
    knob = knob_comparison(candidates(cells, dev), dev, n_boot=200)
    assert knob is not None
    assert knob["verdict"] == "logprob"
    assert knob["knobs"]["logprob"]["config"] == "v1-label"
    assert knob["knobs"]["ordinal"]["reads"] == "score value"
    assert knob["paired_bootstrap"]["delta_logprob_minus_ordinal"]["low"] > 0
    assert knob_comparison(candidates(cells[:1], dev), dev, n_boot=10) is None


def test_directional_mass_recovers_calls_the_argmax_rule_never_makes() -> None:
    def lab(bull: float, bear: float) -> dict[str, Any]:
        probs = {"bullish": bull, "bearish": bear, "neutral": 1 - bull - bear}
        return {"kind": "label", "fallback": "none", "label": "neutral", "probs": probs}

    # Neutral is the argmax everywhere, so every serving threshold makes zero calls.
    cell = make_cell(
        "label", [lab(0.4, 0), lab(0.2, 0), lab(0, 0.45), lab(0.1, 0.3), lab(0.05, 0.05), lab(0.3, 0)]
    )
    side, p = directional_mass(cell.records[3])
    assert side == "bearish"
    assert p == pytest.approx(0.3)
    assert directional_mass(cell.records[4])[0] == "bullish"
    assert directional_mass(failed()) == ("neutral", -math.inf)
    assert directional_mass({"kind": "label", "probs": None, "score": -1}) == ("bearish", 1.0)

    report = directional_mass_report([cell], candidates([cell], DEV), DEV, n_boot=50)
    assert report is not None
    # From tau 0.06 to 0.2: bull calls e0 e1 e5 (P 2/3, R 1 -> 5/7), bear calls e2 e3 (F0.5 1).
    assert report["threshold"] == 0.06
    assert report["f05_macro"] == pytest.approx((5 / 7 + 1) / 2)
    assert report["inversion_rate"] == 0.0
    assert report["paired_bootstrap"]["serving"]["point"] == 0.0
    assert report["verdict"] == "directional-mass"
    assert directional_mass_report([make_cell("sentiment-first", [analyze(1)] * 6)], [], DEV, 10) is None


def test_error_patterns_count_by_kind_bucket_and_model() -> None:
    dev = DevSet(DEV.ids, GOLD, None, buckets=("analyst action",) * 3 + ("other",) * 3)
    a = Served(
        "a",
        ("bullish", "neutral", "bullish", "bearish", "bearish", "neutral"),
        (False, False, False, True, False, False),
        (False, True, False, False, False, False),
    )
    b = Served("b", ("bullish",) * 6, (False,) * 6, (False,) * 6)
    patterns = error_patterns([a, b], dev)
    assert patterns["per_model"]["a"] == {
        "directional_calls": 3,
        "wrong_calls": 2,
        "neutral_called_bullish": 0,
        "neutral_called_bearish": 1,
        "bullish_called_bearish": 0,
        "bearish_called_bullish": 1,
        "bullish_missed": 1,
        "bearish_missed": 1,
        "failed": 1,
    }
    b_tally = patterns["per_model"]["b"]
    assert (b_tally["neutral_called_bullish"], b_tally["bearish_called_bullish"]) == (2, 2)
    analyst = patterns["buckets"]["analyst action"]
    assert analyst["gold"] == {"bullish": 2, "bearish": 1, "neutral": 0}
    assert analyst["models"]["a"] == {"directional_calls": 2, "wrong_calls": 1, "missed_directional": 1}
    assert patterns["cross_model"] == {
        "models": 2,
        "directional_gold_no_model_calls_correctly": 2,
        "neutral_gold_every_model_calls_directional": 1,
        "any_model_inverts": 2,
    }


def test_served_calls_follow_the_chosen_threshold() -> None:
    cell = make_cell("decision", [*DECISION_RECORDS[:5], failed()])
    best, cap_met = best_candidate(candidates([cell], DEV))
    served = served_calls(best, cap_met)
    assert served.failed == (False,) * 5 + (True,)
    assert served.abstained[-1] is True


def test_bucket_heuristics() -> None:
    assert bucket_of("Zentor upgraded to Buy at Halvard; PT raised to $40") == "analyst action"
    assert bucket_of("Orvane Q3 EPS beats by $0.05") == "earnings / guidance"
    assert bucket_of("Shares of Pellow jump 9% premarket") == "price move"
    assert bucket_of("Fed holds rates steady") == "macro / markets"
    assert bucket_of("Pellow names a new chair") == "other"


def test_latency_counts_and_diagnostics() -> None:
    records: list[Record | None] = [
        {"kind": "analyze", "score": 1, "error": None, "t": {"wall_ms": 100.0}},
        {"kind": "analyze", "score": -2, "error": None, "t": {"wall_ms": 300.0, "total_ms": 250.0}},
        {"kind": "analyze", "score": 0, "error": None, "t": {"wall_ms": 200.0}},
        {**failed(), "t": {"wall_ms": 5000.0}},
        None,
    ]
    lat = latency(records)
    assert (lat["n"], lat["wall_ms_p50"], lat["server_ms_p50"]) == (3, 200.0, 250.0)
    assert counts(records) == {"n": 5, "done": 4, "missing": 1, "errors": 1, "error_schema_violation": 1}
    hist = diagnostics(make_cell("sentiment-first", records))["score_histogram"]
    assert hist == {"-2": 1, "-1": 0, "0": 1, "1": 1, "2": 0}
    label = {
        "kind": "label",
        "fallback": "none",
        "label": "bullish",
        "probs": {"bullish": 0.4, "neutral": 0.6, "bearish": 0},
    }
    diag = diagnostics(make_cell("label", [label, {**label, "fallback": "ordinal", "probs": None}]))
    assert diag == {"fallback": {"none": 1, "ordinal": 1}, "generated_label_not_argmax": 1}


def test_thin_sweep_and_rounding() -> None:
    ev = evaluate_policy("probability", DECISION_RECORDS, GOLD, None)
    rows = thin_sweep(ev.sweep)
    assert rows[0]["threshold"] == 0.0
    # Thresholds up to the lowest winning probability change nothing, so they collapse into t=0.
    assert rows[1]["threshold"] == 0.57
    assert len(rows) < len(PROB_THRESHOLDS)
    assert rounded({"a": [1 / 3, math.inf], "b": (2.0,)}) == {"a": [0.3333, None], "b": [2.0]}
