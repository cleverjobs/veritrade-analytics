"""Signal metrics: bullish/bearish calls are the positives; neutral and abstain are the safe default."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veritrade.inference.types import SENTIMENTS, Sentiment

DIRECTIONAL: tuple[Sentiment, Sentiment] = ("bullish", "bearish")
CONFUSION_COLUMNS = ("bullish", "bearish", "neutral", "abstain")
BETA = 0.5
INVERSION_CAP = 0.02
HEADLINE = ("f05_macro", "inversion_rate", "coverage", "signal_rate", "accuracy")
BOOTSTRAP_SEED = 20261018

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class ClassScores:
    precision: float
    recall: float
    f05: float
    calls: float
    support: float


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """Rates are prevalence-weighted when weights are given; `confusion` is raw counts (gold x columns)."""

    n: int
    per_class: dict[str, ClassScores]
    f05_macro: float
    inversion_rate: float
    coverage: float
    signal_rate: float
    accuracy: float
    selective_accuracy: float
    confusion: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _div(num: float, den: float) -> float:
    # Zero calls or zero support score 0 (not NaN): "always neutral" cannot look good and JSON stays strict.
    return float(num / den) if den > 0 else 0.0


def fbeta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta * beta
    den = b2 * precision + recall
    return (1 + b2) * precision * recall / den if den > 0 else 0.0


def _labels(values: Sequence[str], what: str) -> NDArray[np.str_]:
    arr = np.asarray(list(values), dtype=np.str_)
    unknown = set(arr.tolist()) - set(SENTIMENTS)
    if unknown:
        raise ValueError(f"unknown {what} labels: {sorted(unknown)}")
    return arr


def _weights(weights: Sequence[float] | None, n: int) -> FloatArray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be one non-negative value per example with a positive sum")
    return w


def signal_metrics(
    gold: Sequence[Sentiment],
    pred: Sequence[Sentiment],
    abstained: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> SignalMetrics:
    g, p = _labels(gold, "gold"), _labels(pred, "pred")
    a = np.asarray(abstained, dtype=bool)
    n = len(g)
    if n == 0 or p.shape != (n,) or a.shape != (n,):
        raise ValueError("gold, pred and abstained must be non-empty and the same length")
    w = _weights(weights, n)
    total = float(w.sum())
    answered = ~a
    call = np.where(answered & np.isin(p, DIRECTIONAL), p, "")
    directional = call != ""

    per_class: dict[str, ClassScores] = {}
    for c in DIRECTIONAL:
        tp = float(w[(call == c) & (g == c)].sum())
        calls = float(w[call == c].sum())
        support = float(w[g == c].sum())
        precision, recall = _div(tp, calls), _div(tp, support)
        per_class[c] = ClassScores(precision, recall, fbeta(precision, recall), calls, support)

    inverted = directional & np.isin(g, DIRECTIONAL) & (call != g)
    served = np.where(a, "neutral", p)
    confusion = tuple(
        (*(int(((g == gl) & answered & (p == pl)).sum()) for pl in SENTIMENTS), int(((g == gl) & a).sum()))
        for gl in SENTIMENTS
    )
    return SignalMetrics(
        n=n,
        per_class=per_class,
        f05_macro=float(np.mean([per_class[c].f05 for c in DIRECTIONAL])),
        inversion_rate=_div(float(w[inverted].sum()), float(w[directional].sum())),
        coverage=float(w[answered].sum()) / total,
        signal_rate=float(w[directional].sum()) / total,
        accuracy=float(w[served == g].sum()) / total,
        selective_accuracy=_div(float(w[answered & (p == g)].sum()), float(w[answered].sum())),
        confusion=confusion,
    )


def example_weights(gold: Sequence[Sentiment], class_weights: Mapping[str, float]) -> list[float]:
    return [class_weights[label] for label in gold]


# --- risk-coverage and operating point -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepPoint:
    threshold: float
    metrics: SignalMetrics

    @property
    def f05_macro(self) -> float:
        return self.metrics.f05_macro

    @property
    def inversion_rate(self) -> float:
        return self.metrics.inversion_rate

    @property
    def coverage(self) -> float:
        return self.metrics.coverage

    @property
    def risk(self) -> float:
        return 1.0 - self.metrics.selective_accuracy


def candidate_thresholds(confidences: Sequence[float]) -> list[float]:
    """0 (never abstain) plus each distinct finite confidence: the only points where calls change."""
    finite = {float(c) for c in confidences if math.isfinite(c)}
    return sorted(finite | {0.0})


def risk_coverage_sweep(
    gold: Sequence[Sentiment],
    labels: Sequence[Sentiment],
    confidences: Sequence[float],
    thresholds: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[SweepPoint]:
    """Abstain whenever confidence < threshold; a NaN confidence never clears any threshold."""
    conf = np.asarray(confidences, dtype=np.float64)
    points = []
    for t in sorted(set(thresholds)):
        abstained = ~(conf >= t)
        points.append(SweepPoint(float(t), signal_metrics(gold, labels, abstained.tolist(), weights)))
    return points


def select_operating_point(
    points: Sequence[SweepPoint], inversion_cap: float = INVERSION_CAP
) -> SweepPoint | None:
    """Best F0.5 among points under the inversion cap; ties go to higher coverage, then lower threshold."""
    eligible = [p for p in points if p.inversion_rate <= inversion_cap + 1e-12]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (round(p.f05_macro, 12), round(p.coverage, 12), -p.threshold))


# --- bootstrap -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    level: float
    n_boot: int


def stratified_indices(gold: NDArray[np.str_], rng: np.random.Generator) -> IndexArray:
    """Resample with replacement inside each gold class so every replicate keeps the class sizes."""
    parts = [
        rng.choice(np.flatnonzero(gold == label), size=int((gold == label).sum()))
        for label in np.unique(gold)
    ]
    return np.concatenate(parts).astype(np.intp)


def bootstrap_cis(
    metrics_fn: Callable[[IndexArray], Mapping[str, float]],
    gold: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> dict[str, ConfidenceInterval]:
    """Percentile CIs for several metrics from one shared set of stratified resamples."""
    g = np.asarray(list(gold), dtype=np.str_)
    rng = np.random.default_rng(seed)
    point = metrics_fn(np.arange(len(g), dtype=np.intp))
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        values = metrics_fn(stratified_indices(g, rng))
        for k in samples:
            samples[k].append(values[k])
    tail = (1 - level) / 2
    out: dict[str, ConfidenceInterval] = {}
    for k, vals in samples.items():
        low, high = np.quantile(np.asarray(vals), [tail, 1 - tail])
        out[k] = ConfidenceInterval(float(point[k]), float(low), float(high), level, n_boot)
    return out


def bootstrap_ci(
    metric_fn: Callable[[IndexArray], float],
    gold: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> ConfidenceInterval:
    return bootstrap_cis(lambda idx: {"m": metric_fn(idx)}, gold, n_boot=n_boot, seed=seed, level=level)["m"]


def headline_cis(
    gold: Sequence[Sentiment],
    pred: Sequence[Sentiment],
    abstained: Sequence[bool],
    weights: Sequence[float] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, ConfidenceInterval]:
    g = np.asarray(list(gold), dtype=np.str_)
    p = np.asarray(list(pred), dtype=np.str_)
    a = np.asarray(abstained, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def fn(idx: IndexArray) -> dict[str, float]:
        m = signal_metrics(
            g[idx].tolist(), p[idx].tolist(), a[idx].tolist(), None if w is None else w[idx].tolist()
        )
        return {k: float(getattr(m, k)) for k in HEADLINE}

    return bootstrap_cis(fn, g.tolist(), n_boot=n_boot, seed=seed)


# --- quality vs latency ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    f05: float
    latency_ms: float


def pareto_front(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Non-dominated candidates (F0.5 up, latency down), fastest first."""
    front: list[Candidate] = []
    best = -math.inf
    for c in sorted(candidates, key=lambda c: (c.latency_ms, -c.f05, c.name)):
        if c.f05 > best:
            front.append(c)
            best = c.f05
    return front


def _minmax(values: FloatArray) -> FloatArray:
    span = float(values.max() - values.min())
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def knee(candidates: Sequence[Candidate]) -> Candidate:
    """Front point farthest above the fastest-to-best chord on min-max axes; ties prefer higher F0.5."""
    front = pareto_front(candidates)
    if not front:
        raise ValueError("knee of an empty candidate set")
    if len(front) <= 2:
        return max(front, key=lambda c: (c.f05, -c.latency_ms))
    lat = _minmax(np.array([c.latency_ms for c in front], dtype=np.float64))
    q = _minmax(np.array([c.f05 for c in front], dtype=np.float64))
    dx, dy = lat[-1] - lat[0], q[-1] - q[0]
    # Signed so only points bulging toward the fast-and-accurate corner count as a knee.
    dist = (dx * (q - q[0]) - dy * (lat - lat[0])) / math.hypot(dx, dy)
    return max(zip(front, dist.tolist(), strict=True), key=lambda cd: (round(cd[1], 12), cd[0].f05))[0]
