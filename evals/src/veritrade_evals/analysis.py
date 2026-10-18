"""Offline grid analysis: replay each serving policy on cached raw outputs, sweep its threshold, pick the
operating point under the inversion cap, then compare configs, knobs and models. Counts only, no text."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from veritrade.inference.confidence import normalise, winner
from veritrade.inference.schemas import score_sentiment
from veritrade.inference.types import Sentiment
from veritrade_evals.metrics import (
    DIRECTIONAL,
    INVERSION_CAP,
    ConfidenceInterval,
    IndexArray,
    SignalMetrics,
    SweepPoint,
    bootstrap_cis,
    headline_cis,
    risk_coverage_sweep,
    select_operating_point,
    signal_metrics,
)
from veritrade_evals.runner import Variant

PolicyName = Literal["probability", "label-logprob", "ordinal"]
Record = Mapping[str, Any]

# Round min_p values to freeze; finer near 1 because decision models are often very peaked.
PROB_THRESHOLDS: tuple[float, ...] = (
    0.0,
    *(round(0.34 + 0.01 * i, 2) for i in range(66)),
    *(round(0.991 + 0.001 * i, 3) for i in range(9)),
    0.9995,
    0.9999,
)
ORDINAL_THRESHOLDS: tuple[float, ...] = (1.0, 2.0)
# A directional class can win a call with far less than 1/3 of the mass, so the grid starts near 0.
MASS_THRESHOLDS: tuple[float, ...] = tuple(
    sorted({round(0.02 * i, 2) for i in range(1, 17)} | set(PROB_THRESHOLDS[1:]))
)
# Score-sign probabilities stay out of the knobs: the qwen tokenizer fuses " -" into one token, so the
# first value token misses the sign decision (records keep them as raw `sign_probs`).
POLICIES: dict[Variant, tuple[PolicyName, ...]] = {
    "decision": ("probability",),
    "label": ("label-logprob",),
    "ordinal": ("ordinal",),
    "sentiment-first": ("ordinal",),
}

# Keyword heuristics for error analysis; the first match wins and nothing but the bucket name leaves.
BUCKETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "analyst action",
        re.compile(
            r"\b(upgrade\w*|downgrade\w*|price target|target price|pt|initiat\w*|reiterat\w*|outperform|"
            r"underperform|overweight|underweight|equal[- ]weight|market perform|sector perform|"
            r"buy rating|sell rating|hold rating|rating)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "earnings / guidance",
        re.compile(
            r"\b(earnings|eps|revenue|revenues|sales|results|quarter\w*|q[1-4]|fy\d*|guidance|forecasts?|"
            r"outlook|profits?|loss(es)?|beats?|miss(es)?|dividend\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "price move",
        re.compile(
            r"\b(ris(e|es|ing)|rose|jump\w*|surg\w*|soar\w*|rall\w*|gain\w*|climb\w*|fall(s|ing)?|fell|"
            r"drop\w*|slump\w*|sink\w*|sank|slid\w*|tumbl\w*|plung\w*|higher|lower|up|down)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "macro / markets",
        re.compile(
            r"\b(fed|inflation|rates?|treasur\w*|yields?|gdp|jobs|payrolls|unemployment|oil|dollar|"
            r"econom\w*|recession|cpi|ecb|tariffs?|stocks|markets?|futures|dow|nasdaq|s&p)\b",
            re.IGNORECASE,
        ),
    ),
)
OTHER_BUCKET = "other"
BUCKET_NAMES: tuple[str, ...] = (*(name for name, _ in BUCKETS), OTHER_BUCKET)


def bucket_of(text: str) -> str:
    for name, pattern in BUCKETS:
        if pattern.search(text):
            return name
    return OTHER_BUCKET


# --- policy replay ---------------------------------------------------------------------------


def _ordinal(score: int) -> tuple[Sentiment, float]:
    # A zero score is a confident neutral, never an abstention.
    label = score_sentiment(score)
    return label, math.inf if label == "neutral" else float(abs(score))


def replay(policy: PolicyName, record: Record | None, *, ordinal_t: int = 1) -> tuple[Sentiment, float]:
    """(raw class, confidence) as the serving policy sees one cached output; the row abstains when the
    confidence is below the threshold, so failed or missing calls (-inf) always abstain."""
    if record is None or record.get("error") is not None:
        return "neutral", -math.inf
    if policy == "ordinal":
        return _ordinal(int(record["score"]))
    probs = record.get("sign_probs") if record.get("kind") == "ordinal" else record.get("probs")
    if probs:
        p = normalise(cast(Mapping[Sentiment, float], probs))
        top = winner(p)
        return top, p[top]
    if policy == "probability" or record.get("score") is None:
        raise ValueError(f"record {record.get('id')!r} has no probabilities for {policy}")
    # Serving ends the fallback chain at the ordinal policy with the model's frozen t.
    label, confidence = _ordinal(int(record["score"]))
    return label, math.inf if confidence >= ordinal_t else -math.inf


@dataclass(frozen=True, slots=True)
class PolicyEval:
    policy: PolicyName | Literal["directional-mass"]
    labels: tuple[Sentiment, ...]
    confidences: tuple[float, ...]
    sweep: tuple[SweepPoint, ...]

    @property
    def default(self) -> SweepPoint:
        return self.sweep[0]

    @property
    def operating(self) -> SweepPoint | None:
        return select_operating_point(self.sweep)

    def abstained(self, threshold: float) -> list[bool]:
        return [not c >= threshold for c in self.confidences]


def evaluate_policy(
    policy: PolicyName,
    records: Sequence[Record | None],
    gold: Sequence[Sentiment],
    weights: Sequence[float] | None,
    *,
    ordinal_t: int = 1,
) -> PolicyEval:
    calls = [replay(policy, r, ordinal_t=ordinal_t) for r in records]
    labels = tuple(label for label, _ in calls)
    confidences = tuple(c for _, c in calls)
    thresholds = ORDINAL_THRESHOLDS if policy == "ordinal" else PROB_THRESHOLDS
    sweep = risk_coverage_sweep(gold, labels, confidences, thresholds, weights)
    return PolicyEval(policy, labels, confidences, tuple(sweep))


# --- cells and candidates --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevSet:
    """Gold side of the split, aligned by position; buckets are keyword classes, never text."""

    ids: tuple[str, ...]
    gold: tuple[Sentiment, ...]
    weights: tuple[float, ...] | None
    buckets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CellInput:
    endpoint: str
    model: str
    name: str
    variant: Variant
    prompt_version: str
    records: tuple[Record | None, ...]
    dims: Mapping[str, str] = field(default_factory=dict)
    ordinal_t: int = 1
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Candidate:
    cell: CellInput
    evaluation: PolicyEval
    order: int

    @property
    def reads(self) -> str:
        """What the threshold is applied to."""
        if self.evaluation.policy == "ordinal":
            return "score value"
        if self.cell.variant == "decision":
            return "class probabilities"
        return "label tokens" if self.cell.variant == "label" else "score-sign tokens"


def candidates(cells: Sequence[CellInput], dev: DevSet) -> list[Candidate]:
    out: list[Candidate] = []
    for cell in cells:
        for policy in POLICIES[cell.variant]:
            evaluation = evaluate_policy(
                policy, cell.records, dev.gold, dev.weights, ordinal_t=cell.ordinal_t
            )
            out.append(Candidate(cell, evaluation, len(out)))
    return out


def best_candidate(cands: Sequence[Candidate]) -> tuple[Candidate, bool]:
    """Max F0.5 at each candidate's operating point; earlier (simpler) configs win exact ties. When no
    candidate meets the inversion cap, falls back to the best no-abstention F0.5 and says so."""
    if not cands:
        raise ValueError("no candidates")
    eligible = [(c, op) for c in cands if (op := c.evaluation.operating) is not None]
    if eligible:
        best, _ = max(
            eligible, key=lambda co: (round(co[1].f05_macro, 12), round(co[1].coverage, 12), -co[0].order)
        )
        return best, True
    return max(cands, key=lambda c: (round(c.evaluation.default.f05_macro, 12), -c.order)), False


def chosen_point(candidate: Candidate, cap_met: bool) -> SweepPoint:
    op = candidate.evaluation.operating
    return op if cap_met and op is not None else candidate.evaluation.default


# --- summaries -------------------------------------------------------------------------------


def point_row(point: SweepPoint) -> dict[str, float]:
    m = point.metrics
    return {
        "threshold": point.threshold,
        "f05_macro": m.f05_macro,
        "precision_bullish": m.per_class["bullish"].precision,
        "precision_bearish": m.per_class["bearish"].precision,
        "recall_bullish": m.per_class["bullish"].recall,
        "recall_bearish": m.per_class["bearish"].recall,
        "inversion_rate": m.inversion_rate,
        "coverage": m.coverage,
        "signal_rate": m.signal_rate,
        "risk": point.risk,
    }


def thin_sweep(sweep: Sequence[SweepPoint]) -> list[dict[str, float]]:
    """Drops points whose metrics equal the previous point's, keeping the lowest threshold of each run."""
    rows: list[dict[str, float]] = []
    previous: dict[str, float] | None = None
    for point in sweep:
        row = point_row(point)
        body = {k: v for k, v in row.items() if k != "threshold"}
        if body != previous:
            rows.append(row)
        previous = body
    return rows


def metrics_dict(metrics: SignalMetrics) -> dict[str, Any]:
    out = metrics.to_dict()
    out["confusion"] = {
        "rows_gold": ["bullish", "bearish", "neutral"],
        "columns_pred": ["bullish", "bearish", "neutral", "abstain"],
        "counts": [list(row) for row in metrics.confusion],
    }
    return out


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def latency(records: Sequence[Record | None]) -> dict[str, Any]:
    ok = [r for r in records if r is not None and r.get("error") is None]
    wall = [float(r["t"]["wall_ms"]) for r in ok if "wall_ms" in r.get("t", {})]
    server = [float(r["t"]["total_ms"]) for r in ok if "total_ms" in r.get("t", {})]
    return {
        "n": len(wall),
        "wall_ms_p50": percentile(wall, 50),
        "wall_ms_p95": percentile(wall, 95),
        "server_ms_p50": percentile(server, 50),
    }


def counts(records: Sequence[Record | None]) -> dict[str, int]:
    present = [r for r in records if r is not None]
    errors = Counter(str(r["error"]["code"]) for r in present if r.get("error") is not None)
    return {
        "n": len(records),
        "done": len(present),
        "missing": len(records) - len(present),
        "errors": sum(errors.values()),
        **{f"error_{code}": n for code, n in sorted(errors.items())},
    }


def diagnostics(cell: CellInput) -> dict[str, Any]:
    """Counts that explain a cell's behaviour: fallbacks, argmax disagreements, score histogram."""
    ok = [r for r in cell.records if r is not None and r.get("error") is None]
    out: dict[str, Any] = {}
    if cell.variant == "label":
        out["fallback"] = dict(sorted(Counter(str(r["fallback"]) for r in ok).items()))
        out["generated_label_not_argmax"] = sum(
            1
            for r in ok
            if r.get("probs") and r.get("label") and r["label"] != winner(normalise(cast(Any, r["probs"])))
        )
    if cell.variant == "decision":
        out["choice_not_argmax"] = sum(
            1 for r in ok if r["choice"] != winner(normalise(cast(Any, r["probs"])))
        )
    if cell.variant in ("ordinal", "sentiment-first"):
        hist = Counter(int(r["score"]) for r in ok)
        out["score_histogram"] = {str(s): hist.get(s, 0) for s in (-2, -1, 0, 1, 2)}
    if cell.variant == "ordinal":
        out["no_sign_probs"] = sum(1 for r in ok if not r.get("sign_probs"))
    return out


def policy_report(evaluation: PolicyEval) -> dict[str, Any]:
    op = evaluation.operating
    return {
        "default": {
            "threshold": evaluation.default.threshold,
            "metrics": metrics_dict(evaluation.default.metrics),
        },
        "operating_point": None
        if op is None
        else {"threshold": op.threshold, "metrics": metrics_dict(op.metrics)},
        "sweep": thin_sweep(evaluation.sweep),
    }


def ci_dict(cis: Mapping[str, ConfidenceInterval]) -> dict[str, dict[str, float]]:
    return {k: {"point": v.point, "low": v.low, "high": v.high} for k, v in cis.items()}


def candidate_cis(candidate: Candidate, point: SweepPoint, dev: DevSet, n_boot: int) -> dict[str, Any]:
    abstained = candidate.evaluation.abstained(point.threshold)
    return ci_dict(headline_cis(dev.gold, candidate.evaluation.labels, abstained, dev.weights, n_boot=n_boot))


def best_report(cands: Sequence[Candidate], dev: DevSet, n_boot: int) -> dict[str, Any]:
    best, cap_met = best_candidate(cands)
    point = chosen_point(best, cap_met)
    return {
        "config": best.cell.name,
        "variant": best.cell.variant,
        "policy": best.evaluation.policy,
        "prompt_version": best.cell.prompt_version,
        "threshold": point.threshold,
        "inversion_cap_met": cap_met,
        "metrics": metrics_dict(point.metrics),
        "cis": candidate_cis(best, point, dev, n_boot),
        "latency": latency(best.cell.records),
    }


# --- knob comparison -------------------------------------------------------------------------


def _knob(c: Candidate) -> str:
    return "ordinal" if c.evaluation.policy == "ordinal" else "logprob"


def paired_f05(
    a: tuple[Candidate, SweepPoint], b: tuple[Candidate, SweepPoint], dev: DevSet, n_boot: int
) -> dict[str, ConfidenceInterval]:
    """F0.5 of a, of b and of a - b on shared resamples, so example difficulty cancels in the gap."""
    g = np.asarray(dev.gold, dtype=np.str_)
    w = None if dev.weights is None else np.asarray(dev.weights, dtype=np.float64)
    served = [
        (np.asarray(c.evaluation.labels, dtype=np.str_), np.asarray(c.evaluation.abstained(p.threshold)))
        for c, p in (a, b)
    ]

    def f05(k: int, idx: IndexArray) -> float:
        pred, abstained = served[k]
        weights = None if w is None else w[idx].tolist()
        return signal_metrics(g[idx].tolist(), pred[idx].tolist(), abstained[idx].tolist(), weights).f05_macro

    def fn(idx: IndexArray) -> dict[str, float]:
        fa, fb = f05(0, idx), f05(1, idx)
        return {"a": fa, "b": fb, "gap": fa - fb}

    return bootstrap_cis(fn, dev.gold, n_boot=n_boot)


def _verdict(gap: ConfidenceInterval, a: str, b: str) -> str:
    return a if gap.low > 0 else b if gap.high < 0 else "tie"


def knob_comparison(cands: Sequence[Candidate], dev: DevSet, n_boot: int) -> dict[str, Any] | None:
    """Best logprob-threshold candidate vs best ordinal candidate, with a paired bootstrap on the gap."""
    by_knob = {k: [c for c in cands if _knob(c) == k] for k in ("logprob", "ordinal")}
    if not all(by_knob.values()):
        return None
    picks = {k: best_candidate(v) for k, v in by_knob.items()}
    points = {k: chosen_point(c, ok) for k, (c, ok) in picks.items()}
    paired = paired_f05(
        (picks["logprob"][0], points["logprob"]), (picks["ordinal"][0], points["ordinal"]), dev, n_boot
    )
    cis = {"logprob": paired["a"], "ordinal": paired["b"], "delta_logprob_minus_ordinal": paired["gap"]}
    verdict = _verdict(paired["gap"], "logprob", "ordinal")
    return {
        "knobs": {
            k: {
                "config": c.cell.name,
                "policy": c.evaluation.policy,
                "reads": c.reads,
                "inversion_cap_met": ok,
                **point_row(points[k]),
            }
            for k, (c, ok) in picks.items()
        },
        "paired_bootstrap": ci_dict(cis),
        "verdict": verdict,
    }


# --- informational: directional-mass threshold ------------------------------------------------


def directional_mass(record: Record | None) -> tuple[Sentiment, float]:
    """The likelier directional class and its probability, even when neutral is the argmax; rows without
    probabilities keep their ordinal outcome (directional calls never abstain at t=1)."""
    if record is None or record.get("error") is not None:
        return "neutral", -math.inf
    probs = record.get("probs")
    if not probs:
        return _ordinal(int(record["score"]))
    side: Sentiment = "bullish" if probs["bullish"] >= probs["bearish"] else "bearish"
    return side, float(normalise(cast(Mapping[Sentiment, float], probs))[side])


def evaluate_mass(records: Sequence[Record | None], dev: DevSet) -> PolicyEval:
    calls = [directional_mass(r) for r in records]
    labels = tuple(label for label, _ in calls)
    confidences = tuple(c for _, c in calls)
    sweep = risk_coverage_sweep(dev.gold, labels, confidences, MASS_THRESHOLDS, dev.weights)
    return PolicyEval("directional-mass", labels, confidences, tuple(sweep))


def directional_mass_report(
    cells: Sequence[CellInput], cands: Sequence[Candidate], dev: DevSet, n_boot: int
) -> dict[str, Any] | None:
    """Not a serving policy: shows how much F0.5 the argmax rule leaves behind on neutral-leaning models.
    Below tau the row is served neutral (counted as an abstention)."""
    prob_cells = [c for c in cells if c.variant in ("decision", "label")]
    if not prob_cells:
        return None
    mass = [Candidate(cell, evaluate_mass(cell.records, dev), i) for i, cell in enumerate(prob_cells)]
    best, ok = best_candidate(mass)
    serving, serving_ok = best_candidate(cands)
    point, serving_point = chosen_point(best, ok), chosen_point(serving, serving_ok)
    paired = paired_f05((best, point), (serving, serving_point), dev, n_boot)
    return {
        "config": best.cell.name,
        "inversion_cap_met": ok,
        **point_row(point),
        "per_config_f05": {
            c.cell.name: chosen_point(c, c.evaluation.operating is not None).f05_macro for c in mass
        },
        "serving_best": {"config": serving.cell.name, "policy": serving.evaluation.policy},
        "paired_bootstrap": ci_dict(
            {"directional_mass": paired["a"], "serving": paired["b"], "gap": paired["gap"]}
        ),
        "verdict": _verdict(paired["gap"], "directional-mass", "serving"),
    }


# --- error patterns --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Served:
    model: str
    pred: tuple[Sentiment, ...]
    abstained: tuple[bool, ...]
    failed: tuple[bool, ...]


def served_calls(best: Candidate, cap_met: bool) -> Served:
    point = chosen_point(best, cap_met)
    abstained = best.evaluation.abstained(point.threshold)
    failed = tuple(r is None or r.get("error") is not None for r in best.cell.records)
    return Served(best.cell.model, best.evaluation.labels, tuple(abstained), failed)


def _directional_call(s: Served, i: int) -> Sentiment | None:
    label = s.pred[i]
    return label if not s.abstained[i] and label in DIRECTIONAL else None


ERROR_KEYS = (
    "directional_calls",
    "wrong_calls",
    "neutral_called_bullish",
    "neutral_called_bearish",
    "bullish_called_bearish",
    "bearish_called_bullish",
    "bullish_missed",
    "bearish_missed",
    "failed",
)


def error_patterns(served: Sequence[Served], dev: DevSet) -> dict[str, Any]:
    """Raw counts (unweighted) of how each model's served calls go wrong, overall and per keyword bucket."""
    per_model: dict[str, Any] = {}
    for s in served:
        tally: Counter[str] = Counter()
        for i, gold in enumerate(dev.gold):
            call = _directional_call(s, i)
            if s.failed[i]:
                tally["failed"] += 1
            if call is not None:
                tally["directional_calls"] += 1
                if call != gold:
                    tally["wrong_calls"] += 1
                    tally[f"{gold}_called_{call}"] += 1
            elif gold in DIRECTIONAL:
                tally[f"{gold}_missed"] += 1
        per_model[s.model] = {k: tally.get(k, 0) for k in ERROR_KEYS}
    buckets: dict[str, Any] = {}
    if dev.buckets:
        for name in BUCKET_NAMES:
            idx = [i for i, b in enumerate(dev.buckets) if b == name]
            gold_counts = Counter(dev.gold[i] for i in idx)
            row: dict[str, Any] = {
                "n": len(idx),
                "gold": {k: gold_counts.get(k, 0) for k in ("bullish", "bearish", "neutral")},
                "models": {},
            }
            for s in served:
                calls = [(i, _directional_call(s, i)) for i in idx]
                made = [(i, c) for i, c in calls if c is not None]
                row["models"][s.model] = {
                    "directional_calls": len(made),
                    "wrong_calls": sum(1 for i, c in made if c != dev.gold[i]),
                    "missed_directional": sum(
                        1 for i, c in calls if c is None and dev.gold[i] in DIRECTIONAL
                    ),
                }
            buckets[name] = row
    directional_idx = [i for i, g in enumerate(dev.gold) if g in DIRECTIONAL]
    cross = {
        "models": len(served),
        "directional_gold_no_model_calls_correctly": sum(
            1 for i in directional_idx if all(_directional_call(s, i) != dev.gold[i] for s in served)
        ),
        "neutral_gold_every_model_calls_directional": sum(
            1
            for i, g in enumerate(dev.gold)
            if g == "neutral" and served and all(_directional_call(s, i) is not None for s in served)
        ),
        "any_model_inverts": sum(
            1
            for i in directional_idx
            if any((c := _directional_call(s, i)) is not None and c != dev.gold[i] for s in served)
        ),
    }
    return {"per_model": per_model, "buckets": buckets, "cross_model": cross}


# --- JSON helpers ----------------------------------------------------------------------------


def rounded(obj: Any, digits: int = 4) -> Any:
    """Rounds floats for compact committed JSON; non-finite values become null so JSON stays strict."""
    if isinstance(obj, float):
        return round(obj, digits) if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [rounded(v, digits) for v in obj]
    return obj


# --- markdown --------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def threshold_text(policy: str, threshold: float) -> str:
    if policy == "ordinal":
        return f"\\|s\\| >= {threshold:.0f}"
    return "none" if threshold == 0 else f"p >= {threshold:g}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _ci_text(cis: Mapping[str, Mapping[str, float]], key: str) -> str:
    ci = cis.get(key)
    return "-" if ci is None else f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def best_table(reports: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = []
    for rep in reports:
        best = rep["best"]
        m = best["metrics"]
        lat = best["latency"]
        rows.append(
            [
                f"/{rep['endpoint']}",
                f"`{rep['model']}`",
                best["config"],
                best["policy"],
                threshold_text(best["policy"], best["threshold"]),
                _ci_text(best["cis"], "f05_macro"),
                _num(m["per_class"]["bullish"]["precision"]),
                _num(m["per_class"]["bearish"]["precision"]),
                _pct(m["inversion_rate"]) + ("" if best["inversion_cap_met"] else " (cap missed)"),
                _pct(m["coverage"]),
                _pct(m["signal_rate"]),
                _ms(lat["wall_ms_p50"]),
                _ms(lat["wall_ms_p95"]),
            ]
        )
    header = [
        "endpoint", "model", "config", "policy", "threshold", "F0.5 macro [95% CI]", "P bull", "P bear",
        "inversion", "coverage", "signal rate", "p50 ms", "p95 ms",
    ]  # fmt: skip
    return _table(header, rows)


def config_table(report: Mapping[str, Any]) -> list[str]:
    rows = []
    for cfg in report["configs"]:
        dims = ", ".join(f"{k}={v}" for k, v in cfg["dims"].items())
        for policy, pol in cfg["policies"].items():
            op = pol["operating_point"]
            m = op["metrics"] if op else None
            rows.append(
                [
                    cfg["name"],
                    dims,
                    policy,
                    _num(pol["default"]["metrics"]["f05_macro"]),
                    _num(m["f05_macro"]) if m else "cap missed",
                    threshold_text(policy, op["threshold"]) if op else "-",
                    _num(m["per_class"]["bullish"]["precision"]) if m else "-",
                    _num(m["per_class"]["bearish"]["precision"]) if m else "-",
                    _pct(m["inversion_rate"]) if m else "-",
                    _pct(m["coverage"]) if m else "-",
                    str(cfg["counts"]["errors"]),
                    _ms(cfg["latency"]["wall_ms_p50"]),
                ]
            )
    header = [
        "config", "dims", "policy", "F0.5 no abstain", "F0.5 @ op", "threshold", "P bull", "P bear",
        "inversion", "coverage", "errors", "p50 ms",
    ]  # fmt: skip
    return _table(header, rows)


def knob_lines(model: str, knob: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            name,
            k["config"],
            k["reads"],
            threshold_text(k["policy"], k["threshold"]),
            _ci_text(knob["paired_bootstrap"], name),
            _num(k["precision_bullish"]),
            _num(k["precision_bearish"]),
            _pct(k["inversion_rate"]) + ("" if k["inversion_cap_met"] else " (cap missed)"),
            _pct(k["coverage"]),
        ]
        for name, k in knob["knobs"].items()
    ]
    delta = knob["paired_bootstrap"]["delta_logprob_minus_ordinal"]
    verdict = {
        "logprob": "the logprob threshold wins (95% CI of the gap excludes 0)",
        "ordinal": "the ordinal threshold wins (95% CI of the gap excludes 0)",
        "tie": "no significant difference at n=300 (95% CI of the gap spans 0)",
    }[knob["verdict"]]
    lines = [
        f"### `{model}`",
        "",
        *_table(
            [
                "knob",
                "config",
                "reads",
                "threshold",
                "F0.5 [95% CI]",
                "P bull",
                "P bear",
                "inversion",
                "coverage",
            ],
            rows,
        ),
        "",
        f"Paired bootstrap, logprob minus ordinal: F0.5 gap {delta['point']:+.3f} "
        f"[{delta['low']:+.3f}, {delta['high']:+.3f}]; {verdict}.",
    ]
    return lines


def error_lines(patterns: Mapping[str, Any]) -> list[str]:
    per_model = patterns["per_model"]
    rows = [[f"`{model}`", *(str(t[k]) for k in ERROR_KEYS)] for model, t in per_model.items()]
    header = [
        "model (best config)", "directional calls", "wrong", "neu -> bull", "neu -> bear", "bull -> bear",
        "bear -> bull", "missed bull", "missed bear", "failed calls",
    ]  # fmt: skip
    lines = _table(header, rows)
    buckets = patterns.get("buckets") or {}
    if buckets:
        models = list(per_model)
        header = ["bucket", "n (bull/bear/neu)", *(f"`{m}` wrong/calls, missed" for m in models)]
        brows = []
        for name, row in buckets.items():
            g = row["gold"]
            cells = [
                f"{v['wrong_calls']}/{v['directional_calls']}, {v['missed_directional']}"
                for v in (row["models"][m] for m in models)
            ]
            brows.append([name, f"{row['n']} ({g['bullish']}/{g['bearish']}/{g['neutral']})", *cells])
        lines += [
            "",
            "By keyword bucket (first match wins; heuristic, for orientation only):",
            "",
            *_table(header, brows),
        ]
    cross = patterns["cross_model"]
    lines += [
        "",
        f"Across the {cross['models']} best configs (tweet counts):",
        "",
        "- gold directional, no model calls it correctly: "
        f"{cross['directional_gold_no_model_calls_correctly']}",
        "- gold neutral, every model calls it directional: "
        f"{cross['neutral_gold_every_model_calls_directional']}",
        f"- gold directional, inverted by at least one model: {cross['any_model_inverts']}",
    ]
    return lines


def mass_lines(masses: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    rows = []
    for model, m in masses:
        boot = m["paired_bootstrap"]
        gap = boot["gap"]
        rows.append(
            [
                f"`{model}`",
                m["config"],
                f"p(class) >= {m['threshold']:g}",
                _ci_text(boot, "directional_mass"),
                _num(m["precision_bullish"]),
                _num(m["precision_bearish"]),
                _pct(m["inversion_rate"]) + ("" if m["inversion_cap_met"] else " (cap missed)"),
                _pct(m["signal_rate"]),
                f"{m['serving_best']['config']}: {_ci_text(boot, 'serving')}",
                f"{gap['point']:+.3f} [{gap['low']:+.3f}, {gap['high']:+.3f}]",
            ]
        )
    header = [
        "model", "config", "threshold", "F0.5 [95% CI]", "P bull", "P bear", "inversion", "signal rate",
        "serving best F0.5", "gap",
    ]  # fmt: skip
    return [
        "## Not served yet: directional-mass threshold",
        "",
        "The serving policies abstain only when the winning class is unsure, so a model that keeps most "
        "of its mass on neutral is never pushed into a call. This informational policy calls bullish or "
        "bearish whenever that class's probability clears tau, even when neutral is the argmax. It is "
        "tuned on the same 300 tweets as everything else here (gap: paired bootstrap against the serving "
        "best config).",
        "",
        *_table(header, rows),
    ]


def render_summary(
    reports: Sequence[Mapping[str, Any]],
    patterns: Mapping[str, Any],
    *,
    header: Mapping[str, Any],
    notes: str | None = None,
) -> str:
    weights = header["train_prior"]
    lines = [
        "# Signal grid on dev300",
        "",
        "Generated by `uv run python -m veritrade_evals.grid analyze` from the local prediction cache; "
        "no dataset text is stored here.",
        "",
        f"- Data: dev300, {header['n']} Twitter Financial News train tweets, 100 per class; every rate is "
        f"weighted back to the train prior (bullish {_pct(weights['bullish'])}, "
        f"bearish {_pct(weights['bearish'])}, neutral {_pct(weights['neutral'])}). "
        "Confusion and error counts are raw.",
        "- Selection: F0.5 macro over bullish/bearish; operating point = best F0.5 with bullish<->bearish "
        f"inversions <= {INVERSION_CAP:.0%} of directional calls; below the threshold the call abstains "
        "to neutral.",
        "- Policies: decision models threshold native class probabilities (min_p); generative /signal "
        "thresholds first-token label probabilities (label schema) or the |score| of an ordinal -2..2 schema "
        "(t in {1, 2}); /analyze is streamed sentiment-first and always uses |score|.",
        f"- Decoding: Ollama options {header['decoding']} on every generative call (greedy; overrides the "
        "penalties some Modelfiles ship).",
        f"- CIs: 95% stratified bootstrap (B={header['n_boot']}) at the chosen threshold; the threshold "
        "choice itself is not resampled, so these are optimistic.",
        f"- Latency: per-call wall ms during the grid run (Ollama {header['ollama_version']} "
        f"{header['backend']}, warm, client concurrency {header['concurrency']}); /analyze is time to the "
        "score with the stream stopped there. Indicative only: the latency suite runs separately.",
        "",
        "## Best config per model",
        "",
        *best_table(reports),
    ]
    for endpoint in ("signal", "analyze"):
        group = [r for r in reports if r["endpoint"] == endpoint]
        if not group:
            continue
        lines += ["", f"## /{endpoint} configs"]
        for rep in group:
            lines += ["", f"### `{rep['model']}`", "", *config_table(rep)]
    knobs = [(r["model"], r["knob_comparison"]) for r in reports if r.get("knob_comparison")]
    if knobs:
        lines += ["", "## Knob comparison: logprob threshold vs ordinal threshold"]
        for model, knob in knobs:
            lines += ["", *knob_lines(model, knob)]
    masses = [(r["model"], r["directional_mass"]) for r in reports if r.get("directional_mass")]
    if masses:
        lines += ["", *mass_lines(masses)]
    lines += [
        "",
        "## Error patterns (best config per model, at its operating point, raw counts)",
        "",
        *error_lines(patterns),
    ]
    if notes:
        lines += ["", notes.strip()]
    return "\n".join(lines) + "\n"
