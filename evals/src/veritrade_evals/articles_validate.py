"""Checks on agent outputs: schema, line ranges, grounded numbers, no verbatim runs, label agreement."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from veritrade.inference.types import SENTIMENTS, Sentiment
from veritrade_evals.articles import (
    KEY_FACT_MAX_WORDS,
    KEY_FACTS_MAX,
    KEY_FACTS_MIN,
    MIN_BODY_WORDS,
    RATIONALE_MAX_WORDS,
    SHINGLE_WORDS,
    AdjudicateOutput,
    ArticlePaths,
    GoldOutput,
    LabelOutput,
    Stage,
    read_fetched,
    schema_codes,
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z0-9]+")


def word_count(text: str) -> int:
    return len(text.split())


def number_set(text: str) -> set[str]:
    """Numbers as canonical decimals: thousands commas dropped, 1.50 == 1.5."""
    out: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group(0).replace(",", "")
        try:
            out.add(format(Decimal(raw).normalize(), "f"))
        except InvalidOperation:
            out.add(raw)
    return out


def shingles(text: str, n: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    tokens = _WORD.findall(text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def shares_run(text: str, body_shingles: set[tuple[str, ...]], n: int = SHINGLE_WORDS) -> bool:
    return not shingles(text, n).isdisjoint(body_shingles)


def score_matches(label: Sentiment, score: int) -> bool:
    return (score > 0) if label == "bullish" else (score < 0) if label == "bearish" else score == 0


# --- gold -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldCheck:
    id: str
    output: GoldOutput | None
    codes: tuple[str, ...]
    body: str | None = None
    title: str | None = None

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.codes

    @property
    def usable(self) -> bool:
        return self.ok and self.output is not None and self.output.usable


def _ranges(output: GoldOutput, n_lines: int) -> list[str]:
    codes: list[str] = []
    ranges = sorted(output.body_ranges)
    for start, end in ranges:
        if start > end:
            codes.append("range_inverted")
        elif start < 1 or end > n_lines:
            codes.append("range_out_of_bounds")
    if any(later[0] <= earlier[1] for earlier, later in pairwise(ranges)):
        codes.append("ranges_overlap")
    title = output.title_line
    if title is not None and not 1 <= title <= n_lines:
        codes.append("title_out_of_bounds")
    elif title is not None and any(start <= title <= end for start, end in ranges):
        codes.append("title_in_body")
    return sorted(set(codes))


def _key_facts(facts: Sequence[str], body: str) -> list[str]:
    codes: set[str] = set()
    if not KEY_FACTS_MIN <= len(facts) <= KEY_FACTS_MAX:
        codes.add("key_facts_count")
    grounded = number_set(body)
    body_runs = shingles(body)
    for fact in facts:
        if not fact.strip():
            codes.add("key_fact_empty")
        if word_count(fact) > KEY_FACT_MAX_WORDS:
            codes.add("key_fact_too_long")
        if not number_set(fact) <= grounded:
            codes.add("key_fact_number_ungrounded")
        if shares_run(fact, body_runs):
            codes.add("key_fact_verbatim")
    return sorted(codes)


def check_gold(article: str, raw: str | None, lines: Sequence[str] | None) -> GoldCheck:
    if raw is None:
        return GoldCheck(article, None, ("missing_output",))
    try:
        output = GoldOutput.model_validate_json(raw)
    except ValidationError as exc:
        return GoldCheck(article, None, tuple(schema_codes(exc)))
    codes: list[str] = ["id_mismatch"] if output.id != article else []
    if not output.usable:
        if output.reject_reason is None:
            codes.append("missing:reject_reason")
        return GoldCheck(article, output, tuple(codes))
    if lines is None:
        return GoldCheck(article, output, (*codes, "missing_packet_lines"))
    for name in ("title_line", "entity", "provisional_label"):
        value = getattr(output, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            codes.append(f"missing:{name}")
    if not output.body_ranges:
        codes.append("missing:body_ranges")
    if codes:
        return GoldCheck(article, output, tuple(codes))
    if range_codes := _ranges(output, len(lines)):
        return GoldCheck(article, output, tuple(range_codes))
    body = "\n\n".join(
        lines[i - 1] for start, end in sorted(output.body_ranges) for i in range(start, end + 1)
    )
    if word_count(body) < MIN_BODY_WORDS:
        codes.append("body_too_short")
    codes += _key_facts(output.key_facts, body)
    title = lines[output.title_line - 1] if output.title_line else None
    return GoldCheck(article, output, tuple(codes), body=body, title=title)


def load_lines(paths: ArticlePaths, article: str) -> list[str] | None:
    path = paths.lines(article)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(line) for line in payload["lines"]]


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def gold_check(paths: ArticlePaths, article: str) -> GoldCheck:
    return check_gold(article, _read(paths.out("gold", article)), load_lines(paths, article))


# --- labels and adjudication ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelCheck:
    id: str
    output: LabelOutput | None
    codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.codes


@dataclass(frozen=True, slots=True)
class AdjudicateCheck:
    id: str
    output: AdjudicateOutput | None
    codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.codes


def _rationale(rationale: str, label: Sentiment, score: int, body: str | None) -> list[str]:
    codes = []
    if not score_matches(label, score):
        codes.append("score_label_mismatch")
    if word_count(rationale) > RATIONALE_MAX_WORDS:
        codes.append("rationale_too_long")
    if body is None:
        codes.append("gold_missing")
    elif shares_run(rationale, shingles(body)):
        codes.append("rationale_verbatim")
    return codes


def check_label(article: str, raw: str | None, body: str | None) -> LabelCheck:
    if raw is None:
        return LabelCheck(article, None, ("missing_output",))
    try:
        output = LabelOutput.model_validate_json(raw)
    except ValidationError as exc:
        return LabelCheck(article, None, tuple(schema_codes(exc)))
    codes = ["id_mismatch"] if output.id != article else []
    codes += _rationale(output.rationale, output.label, output.score, body)
    return LabelCheck(article, output, tuple(codes))


def check_adjudicate(article: str, raw: str | None, body: str | None, requested: set[str]) -> AdjudicateCheck:
    if raw is None:
        return AdjudicateCheck(article, None, ("missing_output",))
    try:
        output = AdjudicateOutput.model_validate_json(raw)
    except ValidationError as exc:
        return AdjudicateCheck(article, None, tuple(schema_codes(exc)))
    codes = ["id_mismatch"] if output.id != article else []
    if article not in requested:
        codes.append("not_requested")
    codes += _rationale(output.rationale, output.label, output.score, body)
    return AdjudicateCheck(article, output, tuple(codes))


def gold_body(paths: ArticlePaths, article: str) -> str | None:
    return _read(paths.gold(article))


def label_check(paths: ArticlePaths, stage: Literal["label_a", "label_b"], article: str) -> LabelCheck:
    return check_label(article, _read(paths.out(stage, article)), gold_body(paths, article))


def adjudicate_requested(paths: ArticlePaths) -> set[str]:
    text = _read(paths.adjudicate_ids) or ""
    return {line.strip() for line in text.splitlines() if line.strip()}


def adjudicate_check(paths: ArticlePaths, article: str, requested: set[str]) -> AdjudicateCheck:
    return check_adjudicate(
        article, _read(paths.out("adjudicate", article)), gold_body(paths, article), requested
    )


# --- agreement --------------------------------------------------------------------------------------------


def needs_adjudication(a: LabelOutput, b: LabelOutput) -> bool:
    return a.label != b.label or a.ambiguous or b.ambiguous


def cohen_kappa(a: Sequence[Sentiment], b: Sequence[Sentiment]) -> float | None:
    """Chance-corrected agreement of two raters; 1.0 when both are constant and identical."""
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in SENTIMENTS)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True, slots=True)
class Agreement:
    n: int
    agreed: int
    raw: float | None
    kappa: float | None
    to_adjudicate: tuple[str, ...]


def agreement(pairs: Mapping[str, tuple[LabelOutput, LabelOutput]]) -> Agreement:
    ids = sorted(pairs)
    a = [pairs[i][0].label for i in ids]
    b = [pairs[i][1].label for i in ids]
    agreed = sum(x == y for x, y in zip(a, b, strict=True))
    flagged = tuple(i for i in ids if needs_adjudication(*pairs[i]))
    return Agreement(len(ids), agreed, agreed / len(ids) if ids else None, cohen_kappa(a, b), flagged)


# --- ingest -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ingested:
    id: str
    ok: bool
    codes: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class IngestReport:
    stage: Stage
    results: tuple[Ingested, ...]
    agreement: Agreement | None = None


def _output_ids(paths: ArticlePaths, stage: Stage, source: str | None) -> list[str]:
    folder = paths.out_dir(stage)
    ids = sorted(p.stem for p in folder.glob("art-*.json")) if folder.exists() else []
    if source is None:
        return ids
    return [i for i in ids if (record := read_fetched(paths, i)) is not None and record.source == source]


def ingest(paths: ArticlePaths, stage: Stage, source: str | None = None) -> IngestReport:
    """Validate one stage's outputs; gold writes bodies, label stages refresh the adjudication list."""
    results: list[Ingested] = []
    requested = adjudicate_requested(paths)
    for article in _output_ids(paths, stage, source):
        if stage == "gold":
            check = gold_check(paths, article)
            gold_path = paths.gold(article)
            if check.usable and check.body is not None and check.output is not None:
                gold_path.parent.mkdir(parents=True, exist_ok=True)
                gold_path.write_text(check.body, encoding="utf-8")
                note = f"usable provisional={check.output.provisional_label} words={word_count(check.body)}"
            else:
                gold_path.unlink(missing_ok=True)
                reason = check.output.reject_reason if check.output is not None else None
                note = f"rejected:{reason}" if check.ok else ""
            results.append(Ingested(article, check.ok, check.codes, note))
        elif stage == "adjudicate":
            adjudicated = adjudicate_check(paths, article, requested)
            decision = adjudicated.output.decision if adjudicated.output is not None else ""
            results.append(Ingested(article, adjudicated.ok, adjudicated.codes, decision))
        else:
            labelled = label_check(paths, stage, article)
            note = (
                f"{labelled.output.label} {labelled.output.score:+d}" if labelled.output is not None else ""
            )
            results.append(Ingested(article, labelled.ok, labelled.codes, note))
    if stage not in ("label_a", "label_b"):
        return IngestReport(stage, tuple(results))
    pairs: dict[str, tuple[LabelOutput, LabelOutput]] = {}
    for article in sorted(
        set(_output_ids(paths, "label_a", None)) & set(_output_ids(paths, "label_b", None))
    ):
        a, b = label_check(paths, "label_a", article), label_check(paths, "label_b", article)
        if a.ok and b.ok and a.output is not None and b.output is not None:
            pairs[article] = (a.output, b.output)
    stats = agreement(pairs)
    paths.adjudicate_ids.parent.mkdir(parents=True, exist_ok=True)
    paths.adjudicate_ids.write_text("".join(f"{i}\n" for i in stats.to_adjudicate), encoding="utf-8")
    return IngestReport(stage, tuple(results), stats)
