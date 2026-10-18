from __future__ import annotations

import json
import math

import numpy as np
import pytest

from veritrade.inference.types import Sentiment
from veritrade_evals.metrics import (
    Candidate,
    ConfidenceInterval,
    IndexArray,
    bootstrap_ci,
    candidate_thresholds,
    example_weights,
    fbeta,
    headline_cis,
    knee,
    pareto_front,
    risk_coverage_sweep,
    select_operating_point,
    signal_metrics,
    stratified_indices,
)

# Hand-computed case: 8 examples, one abstention, one bullish->bearish inversion.
GOLD: list[Sentiment] = [
    "bullish",
    "bullish",
    "bullish",
    "bearish",
    "bearish",
    "neutral",
    "neutral",
    "neutral",
]
PRED: list[Sentiment] = [
    "bullish",
    "bullish",
    "bearish",
    "bearish",
    "neutral",
    "bullish",
    "neutral",
    "bearish",
]
ABSTAINED = [False, False, False, False, False, False, True, False]


def test_fbeta_weights_precision_over_recall() -> None:
    assert fbeta(1.0, 0.5) == pytest.approx(1.25 * 0.5 / (0.25 + 0.5))
    assert fbeta(0.5, 1.0) < fbeta(1.0, 0.5)
    assert fbeta(0.0, 0.0) == 0.0


def test_signal_metrics_unweighted_hand_computed() -> None:
    m = signal_metrics(GOLD, PRED, ABSTAINED)
    bull, bear = m.per_class["bullish"], m.per_class["bearish"]
    assert (bull.precision, bull.recall, bull.f05) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert (bear.precision, bear.recall, bear.f05) == pytest.approx((1 / 3, 1 / 2, 5 / 14))
    assert (bull.calls, bull.support, bear.calls, bear.support) == (3, 3, 3, 2)
    assert m.f05_macro == pytest.approx(43 / 84)
    assert m.inversion_rate == pytest.approx(1 / 6)
    assert m.coverage == pytest.approx(7 / 8)
    assert m.signal_rate == pytest.approx(6 / 8)
    assert m.accuracy == pytest.approx(4 / 8)
    assert m.selective_accuracy == pytest.approx(3 / 7)
    assert m.confusion == ((2, 1, 0, 0), (0, 1, 1, 0), (1, 1, 0, 1))
    assert m.n == 8


def test_signal_metrics_weighted_hand_computed() -> None:
    weights = example_weights(GOLD, {"bullish": 2.0, "bearish": 1.0, "neutral": 0.5})
    m = signal_metrics(GOLD, PRED, ABSTAINED, weights)
    bull, bear = m.per_class["bullish"], m.per_class["bearish"]
    assert (bull.precision, bull.recall, bull.f05) == pytest.approx((8 / 9, 2 / 3, 5 / 6))
    assert (bear.precision, bear.recall, bear.f05) == pytest.approx((2 / 7, 1 / 2, 5 / 16))
    assert m.f05_macro == pytest.approx(55 / 96)
    assert m.inversion_rate == pytest.approx(2 / 8)
    assert m.coverage == pytest.approx(9 / 9.5)
    assert m.signal_rate == pytest.approx(8 / 9.5)
    assert m.accuracy == pytest.approx(5.5 / 9.5)
    assert m.confusion == ((2, 1, 0, 0), (0, 1, 1, 0), (1, 1, 0, 1))


def test_abstained_directional_prediction_is_not_a_call() -> None:
    m = signal_metrics(["bullish", "bearish"], ["bearish", "bearish"], [True, False])
    assert m.inversion_rate == 0.0
    assert m.per_class["bearish"].calls == 1
    assert m.confusion[0] == (0, 0, 0, 1)
    assert m.accuracy == pytest.approx(0.5)


def test_all_neutral_scores_zero_f05_without_nan() -> None:
    m = signal_metrics(GOLD, ["neutral"] * 8, [False] * 8)
    assert m.f05_macro == 0.0
    assert m.per_class["bullish"].precision == 0.0
    assert m.inversion_rate == 0.0
    assert m.signal_rate == 0.0
    assert m.coverage == 1.0
    assert m.accuracy == pytest.approx(3 / 8)
    json.dumps(m.to_dict(), allow_nan=False)


def test_all_abstained_has_zero_coverage() -> None:
    m = signal_metrics(GOLD, PRED, [True] * 8)
    assert (m.coverage, m.signal_rate, m.f05_macro, m.selective_accuracy) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("gold", "pred", "abstained", "weights"),
    [
        ([], [], [], None),
        (["bullish"], ["bullish", "bearish"], [False], None),
        (["bullish"], ["up"], [False], None),
        (["bullish"], ["bullish"], [False], [-1.0]),
    ],
)
def test_signal_metrics_rejects_bad_input(
    gold: list[Sentiment], pred: list[Sentiment], abstained: list[bool], weights: list[float] | None
) -> None:
    with pytest.raises(ValueError, match=r"non-empty|unknown|weights"):
        signal_metrics(gold, pred, abstained, weights)


# Sweep case: conf[4] belongs to a neutral prediction, so t=0.56 and t=0.58 tie on F0.5.
SW_GOLD: list[Sentiment] = ["bullish", "bullish", "bearish", "bearish", "neutral", "neutral"]
SW_PRED: list[Sentiment] = ["bullish", "bullish", "bearish", "bullish", "neutral", "bearish"]
SW_CONF = [0.9, 0.6, 0.8, 0.55, 0.57, 0.5]


def test_risk_coverage_sweep_hand_computed() -> None:
    points = risk_coverage_sweep(SW_GOLD, SW_PRED, SW_CONF, [0.95, 0.0, 0.56, 0.7])
    by_t = {p.threshold: p for p in points}
    assert [p.threshold for p in points] == [0.0, 0.56, 0.7, 0.95]
    assert by_t[0.0].f05_macro == pytest.approx((5 / 7 + 1 / 2) / 2)
    assert by_t[0.0].inversion_rate == pytest.approx(1 / 5)
    assert by_t[0.56].f05_macro == pytest.approx(11 / 12)
    assert by_t[0.56].inversion_rate == 0.0
    assert by_t[0.56].coverage == pytest.approx(4 / 6)
    assert by_t[0.56].risk == pytest.approx(0.0)
    assert by_t[0.7].f05_macro == pytest.approx(5 / 6)
    assert by_t[0.95].coverage == 0.0


def test_nan_confidence_always_abstains() -> None:
    (point,) = risk_coverage_sweep(["bullish"], ["bullish"], [math.nan], [0.0])
    assert point.coverage == 0.0


def test_candidate_thresholds() -> None:
    assert candidate_thresholds([0.5, 0.9, 0.5, math.nan]) == [0.0, 0.5, 0.9]


def test_operating_point_respects_cap_and_breaks_ties_on_coverage() -> None:
    points = risk_coverage_sweep(SW_GOLD, SW_PRED, SW_CONF, [0.0, 0.56, 0.58, 0.7, 0.95])
    best = select_operating_point(points, inversion_cap=0.02)
    assert best is not None
    assert best.threshold == 0.56
    assert next(p for p in points if p.threshold == 0.58).f05_macro == pytest.approx(best.f05_macro)
    assert select_operating_point(points[:1], inversion_cap=0.02) is None
    assert select_operating_point(points[:1], inversion_cap=0.2) is points[0]


def test_operating_point_cap_is_inclusive() -> None:
    gold: list[Sentiment] = ["bullish"] * 49
    gold.append("bearish")
    pred: list[Sentiment] = ["bullish"] * 50
    (point,) = risk_coverage_sweep(gold, pred, [1.0] * 50, [0.0])
    assert point.inversion_rate == pytest.approx(0.02)
    assert select_operating_point([point]) is point


def test_bootstrap_is_deterministic_and_brackets_the_point() -> None:
    gold = SW_GOLD * 10
    g, p = np.asarray(gold), np.asarray(SW_PRED * 10)
    resamples: list[tuple[int, ...]] = []

    def acc(idx: IndexArray) -> float:
        resamples.append(tuple(idx.tolist()))
        return float((g[idx] == p[idx]).mean())

    def run(seed: int) -> tuple[ConfidenceInterval, list[tuple[int, ...]]]:
        resamples.clear()
        ci = bootstrap_ci(acc, gold, n_boot=200, seed=seed)
        return ci, list(resamples)

    (ci3, draws3), (ci3_again, draws3_again), (_, draws4) = run(3), run(3), run(4)
    assert (ci3, draws3) == (ci3_again, draws3_again)
    assert draws3[1:] != draws4[1:]
    assert len(draws3) == 201
    assert ci3.low <= ci3.point <= ci3.high
    assert ci3.point == pytest.approx(4 / 6)


def test_stratified_resample_keeps_class_sizes() -> None:
    gold = np.asarray(["bullish"] * 3 + ["bearish"] * 5 + ["neutral"] * 2)
    idx = stratified_indices(gold, np.random.default_rng(0))
    labels, counts = np.unique(gold[idx], return_counts=True)
    assert dict(zip(labels.tolist(), counts.tolist(), strict=True)) == {
        "bearish": 5,
        "bullish": 3,
        "neutral": 2,
    }


def test_headline_cis_cover_headline_metrics() -> None:
    cis = headline_cis(SW_GOLD * 5, SW_PRED * 5, [False] * 30, n_boot=100, seed=1)
    assert set(cis) == {"f05_macro", "inversion_rate", "coverage", "signal_rate", "accuracy"}
    assert cis["coverage"].low == cis["coverage"].high == 1.0
    assert cis == headline_cis(SW_GOLD * 5, SW_PRED * 5, [False] * 30, n_boot=100, seed=1)


A = Candidate("a", f05=0.50, latency_ms=100)
B = Candidate("b", f05=0.70, latency_ms=200)
C = Candidate("c", f05=0.60, latency_ms=300)
D = Candidate("d", f05=0.80, latency_ms=1000)
F = Candidate("f", f05=0.50, latency_ms=150)


def test_pareto_front_drops_dominated_points() -> None:
    assert pareto_front([D, C, F, B, A]) == [A, B, D]


def test_knee_is_the_bulge_toward_fast_and_accurate() -> None:
    assert knee([A, B, C, D, F]) == B


def test_knee_degenerate_fronts_prefer_quality() -> None:
    assert knee([A]) == A
    assert knee([A, D]) == D
    with pytest.raises(ValueError, match="empty"):
        knee([])
