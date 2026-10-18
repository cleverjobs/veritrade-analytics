"""Opus-as-judge for summaries: blind packets, the output contract of 50-judge.md, validation, de-blinding.

Packets, the blinding map and judge outputs derive from publisher text, so they stay under
evals/data/private/summary/judge; only de-blinded numbers flow into the committed metrics.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from veritrade_evals.articles import SEED, rel, schema_codes
from veritrade_evals.articles_validate import shares_run, shingles, word_count
from veritrade_evals.paths import OPUS_PROMPTS_DIR

JUDGE_PROMPT = OPUS_PROMPTS_DIR / "50-judge.md"
HALLUCINATION_MAX_WORDS = 25
LABEL = re.compile(r"S[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class JudgePaths:
    root: Path

    def packet(self, article: str) -> Path:
        return self.root / "packets" / f"{article}.md"

    def blinding(self, article: str) -> Path:
        return self.root / "blinding" / f"{article}.json"

    def out(self, article: str) -> Path:
        return self.root / "out" / f"{article}.json"

    def packet_list(self, split: str) -> Path:
        return self.root / f"packets_{split}.txt"

    def ingested(self, split: str) -> Path:
        return self.root / "ingested" / f"{split}.jsonl"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgedSummary(_Model):
    label: str
    faithfulness: int = Field(ge=1, le=5)
    coverage: int = Field(ge=1, le=5)
    hallucinations: list[str]
    leads_with_market_moving_fact: bool


class JudgeOutput(_Model):
    id: str
    summaries: list[JudgedSummary]


class Blinding(_Model):
    """Which candidate hides behind each label, and a hash of the text the judge saw."""

    id: str
    split: str
    labels: dict[str, str]
    summary_sha: dict[str, str]


class JudgeRow(_Model):
    id: str
    candidate: str
    faithfulness: int
    coverage: int
    hallucinations: int
    leads: bool


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def blind(article: str, candidates: Sequence[str], seed: int = SEED) -> dict[str, str]:
    """Label -> candidate; depends only on the seed, the article and the candidate set, not on their order."""
    keys = sorted(set(candidates))
    digest = hashlib.sha256(f"{seed}:{article}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(keys)
    return {f"S{i}": key for i, key in enumerate(keys, start=1)}


def make_blinding(article: str, split: str, summaries: Mapping[str, str], seed: int = SEED) -> Blinding:
    labels = blind(article, list(summaries), seed)
    shas = {label: text_sha(summaries[key]) for label, key in labels.items()}
    return Blinding(id=article, split=split, labels=labels, summary_sha=shas)


def render_packet(blinding: Blinding, title: str, body: str, summaries: Mapping[str, str], out: Path) -> str:
    """`summaries` is keyed by candidate; the packet shows them only under their labels."""
    labels = list(blinding.labels)
    lines = [
        f"# Judge packet {blinding.id}",
        "",
        f"- id: {blinding.id}",
        f"- follow: {rel(JUDGE_PROMPT)}",
        f"- candidate summaries: {', '.join(labels)}",
        f"- write your output to: {rel(out)}",
        "",
        "## Title",
        "",
        title,
        "",
        "## Article",
        "",
    ]
    for paragraph in body.split("\n\n"):
        if paragraph.strip():
            lines += [paragraph.strip(), ""]
    lines += ["## Candidate summaries", ""]
    for label in labels:
        lines += [f"### {label}", "", summaries[blinding.labels[label]].strip(), ""]
    return "\n".join(lines)


# --- validation ---------------------------------------------------------------------------------------------


def _label_codes(output: JudgeOutput, expected: Sequence[str]) -> list[str]:
    seen = Counter(s.label for s in output.summaries)
    codes = [f"{label}:duplicate_label" for label, count in seen.items() if count > 1]
    codes += [f"{label}:unknown_label" for label in seen if label not in expected]
    codes += [f"{label}:missing_label" for label in expected if label not in seen]
    return codes


def _summary_codes(judged: JudgedSummary, runs: set[tuple[str, ...]]) -> list[str]:
    codes: list[str] = []
    for note in judged.hallucinations:
        if not note.strip():
            codes.append("hallucination_empty")
        elif word_count(note) > HALLUCINATION_MAX_WORDS:
            codes.append("hallucination_too_long")
        if shares_run(note, runs):
            codes.append("hallucination_verbatim")
    if judged.faithfulness == 5 and judged.hallucinations:
        codes.append("faithfulness_5_with_hallucinations")
    if judged.faithfulness <= 3 and not judged.hallucinations:
        codes.append("low_faithfulness_without_hallucinations")
    return [f"{judged.label}:{code}" for code in sorted(set(codes))]


def check_output(
    raw: str | None,
    blinding: Blinding,
    *,
    article_text: str,
    summaries: Mapping[str, str],
) -> tuple[JudgeOutput | None, tuple[str, ...]]:
    """Schema, label set, own-words notes and score/notes consistency; `summaries` keyed by candidate."""
    if raw is None:
        return None, ("missing_output",)
    try:
        output = JudgeOutput.model_validate_json(raw)
    except ValidationError as exc:
        return None, tuple(schema_codes(exc))
    codes = ["id_mismatch"] if output.id != blinding.id else []
    codes += _label_codes(output, list(blinding.labels))
    for label, key in blinding.labels.items():
        current = summaries.get(key)
        if current is None or text_sha(current) != blinding.summary_sha[label]:
            codes.append(f"{label}:stale_summary")
    runs = shingles(article_text)
    for text in summaries.values():
        runs |= shingles(text)
    for judged in output.summaries:
        if LABEL.fullmatch(judged.label) and judged.label in blinding.labels:
            codes += _summary_codes(judged, runs)
    return output, tuple(codes)


def deblind(output: JudgeOutput, blinding: Blinding) -> list[JudgeRow]:
    return [
        JudgeRow(
            id=output.id,
            candidate=blinding.labels[judged.label],
            faithfulness=judged.faithfulness,
            coverage=judged.coverage,
            hallucinations=len(judged.hallucinations),
            leads=judged.leads_with_market_moving_fact,
        )
        for judged in output.summaries
    ]
