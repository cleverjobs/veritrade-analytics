"""Confidence policies: raw model output -> Verdict, abstaining to neutral below the frozen threshold."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from veritrade.inference.schemas import LabelOut, OrdinalOut, score_sentiment
from veritrade.inference.types import ChatResult, Endpoint, PolicyName, Sentiment, TokenLogprob, Verdict
from veritrade.models_registry import ModelSpec

# Ties resolve to the safe class first; also the criteria order sent to decision models.
TIE_ORDER: tuple[Sentiment, ...] = ("neutral", "bullish", "bearish")
# Share of first-token mass allowed on tokens that could start several classes before we give up.
AMBIGUOUS_MASS_MAX = 0.05

_VALUE_START = {
    "label": re.compile(r'"label"\s*:\s*"'),
    "score": re.compile(r'"score"\s*:\s*'),
}
_SCORE_HEAD = re.compile(r"^(?:-|-[12](?:[\s,}][\s\S]*)?|[012](?:[\s,}][\s\S]*)?)$")
_SCORE_KEY = re.compile(r'"score"\s*:')


def winner(probabilities: Mapping[Sentiment, float]) -> Sentiment:
    best = TIE_ORDER[0]
    for sentiment in TIE_ORDER[1:]:
        if probabilities.get(sentiment, 0.0) > probabilities.get(best, 0.0):
            best = sentiment
    return best


def normalise(probabilities: Mapping[Sentiment, float]) -> dict[Sentiment, float]:
    total = sum(max(0.0, probabilities.get(s, 0.0)) for s in TIE_ORDER)
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return {s: max(0.0, probabilities.get(s, 0.0)) / total for s in TIE_ORDER}


def _threshold(
    name: PolicyName,
    probabilities: dict[Sentiment, float],
    min_p: float,
    score: int | None = None,
    tau: float | None = None,
) -> Verdict:
    top = winner(probabilities)
    if tau is not None and top == "neutral":
        side: Sentiment = "bullish" if probabilities["bullish"] >= probabilities["bearish"] else "bearish"
        if probabilities[side] >= tau:
            return Verdict(
                sentiment=side,
                abstained=False,
                policy=name,
                confidence=probabilities[side],
                score=score,
                probabilities=probabilities,
                threshold=min_p,
                directional_tau=tau,
            )
    confidence = probabilities[top]
    abstained = confidence < min_p
    return Verdict(
        sentiment="neutral" if abstained else top,
        abstained=abstained,
        policy=name,
        confidence=confidence,
        score=score,
        probabilities=probabilities,
        threshold=min_p,
        directional_tau=tau,
    )


def label_classes(rest: str) -> set[Sentiment]:
    """Labels a token's text could start; empty means the grammar would reject it."""
    return {s for s in TIE_ORDER if s.startswith(rest) or rest.startswith(s + '"')}


def sign_classes(rest: str) -> set[Sentiment]:
    """Class implied by a token that starts an enum score (-2..2); empty when the grammar rejects it."""
    if not rest:
        return set(TIE_ORDER)
    if not _SCORE_HEAD.match(rest):
        return set()
    return {"bearish" if rest[0] == "-" else "neutral" if rest[0] == "0" else "bullish"}


def first_value_token(
    content: str, logprobs: Sequence[TokenLogprob], key: str
) -> tuple[str, dict[str, float]] | None:
    """The generated token that starts the value of `key` (minus any text before the value) and its
    alternatives as {text after the same prefix: probability}; None if tokens do not tile the content."""
    match = _VALUE_START[key].search(content)
    if match is None:
        return None
    offset, pos = match.end(), 0
    for tok in logprobs:
        end = pos + len(tok.token)
        if content[pos:end] != tok.token:
            return None
        if end > offset:
            prefix = content[pos:offset]
            candidates: dict[str, float] = {}
            for text, logprob in ((tok.token, tok.logprob), *tok.top):
                if text.startswith(prefix):
                    candidates.setdefault(text[len(prefix) :], math.exp(logprob))
            return tok.token[len(prefix) :], candidates
        pos = end
    return None


def _first_token_distribution(
    content: str,
    logprobs: Sequence[TokenLogprob] | None,
    key: str,
    classes: Callable[[str], set[Sentiment]],
) -> dict[Sentiment, float] | None:
    found = first_value_token(content, logprobs or (), key)
    if found is None:
        return None
    chosen, candidates = found
    if len(classes(chosen)) != 1:
        return None
    mass = dict.fromkeys(TIE_ORDER, 0.0)
    ambiguous = 0.0
    for rest, p in candidates.items():
        # A token ending exactly where the value starts (e.g. a bare space) says nothing about the class.
        if not rest:
            continue
        hit = classes(rest)
        if len(hit) == 1:
            mass[next(iter(hit))] += p
        elif hit:
            ambiguous += p
    total = sum(mass.values())
    if total <= 0 or ambiguous > AMBIGUOUS_MASS_MAX * (total + ambiguous):
        return None
    return {s: mass[s] / total for s in TIE_ORDER}


def label_probabilities(
    content: str, logprobs: Sequence[TokenLogprob] | None
) -> dict[Sentiment, float] | None:
    """First label-token probabilities renormalised over tokens the grammar accepts (Ollama reports
    pre-mask logprobs, so "positive" is dropped). None when the generated first token is shared by
    two labels or ambiguous tokens carry real mass."""
    return _first_token_distribution(content, logprobs, "label", label_classes)


def score_sign_probabilities(
    content: str, logprobs: Sequence[TokenLogprob] | None
) -> dict[Sentiment, float] | None:
    """Class probabilities of the score's sign ("-" bearish, "0" neutral, "1"/"2" bullish) from the tokens
    after the colon, whitespace-insensitive because tokenizers fuse " -". A generated bare space defers to
    the next token; a bare space that was not generated hides its sign and counts as ambiguous."""
    match = _SCORE_KEY.search(content)
    if match is None or not logprobs:
        return None
    offset, pos = match.end(), 0
    mass = dict.fromkeys(TIE_ORDER, 0.0)
    hidden = 0.0
    path = 1.0
    for tok in logprobs:
        end = pos + len(tok.token)
        if content[pos:end] != tok.token:
            return None
        if end >= offset:
            prefix = content[pos:offset] if pos < offset else ""
            candidates: dict[str, float] = {}
            for text, logprob in ((tok.token, tok.logprob), *tok.top):
                if text.startswith(prefix):
                    candidates.setdefault(text[len(prefix) :], math.exp(logprob))
            chosen = tok.token[len(prefix) :]
            for rest, p in candidates.items():
                if rest == chosen:
                    continue
                if not rest.strip():
                    hidden += path * p
                elif hit := sign_classes(rest.lstrip()):
                    mass[next(iter(hit))] += path * p
            if chosen.strip():
                hit = sign_classes(chosen.lstrip())
                if len(hit) != 1:
                    return None
                mass[next(iter(hit))] += path * candidates[chosen]
                total = sum(mass.values())
                if total <= 0 or hidden > AMBIGUOUS_MASS_MAX * (total + hidden):
                    return None
                return {s: mass[s] / total for s in TIE_ORDER}
            path *= candidates[chosen]
        pos = end
    return None


@dataclass(frozen=True, slots=True)
class ProbabilityPolicy:
    """Argmax unless its probability is below min_p (then neutral, abstained). With directional_tau, a
    neutral argmax still yields the likelier directional class when that class reaches tau: models that
    park most of their mass on neutral are otherwise never pushed into a call."""

    min_p: float
    directional_tau: float | None = None
    name: ClassVar[PolicyName] = "probability"

    def decide(self, probabilities: Mapping[Sentiment, float]) -> Verdict:
        return _threshold(self.name, normalise(probabilities), self.min_p, tau=self.directional_tau)


@dataclass(frozen=True, slots=True)
class LabelLogprobPolicy:
    """The probability rule (min_p, directional_tau) on first-token label probabilities. On a label
    collision `decide` returns None and the adapter re-asks with the ordinal schema; when its sign is
    unreadable too, the chain ends at the ordinal policy."""

    min_p: float
    directional_tau: float | None = None
    name: ClassVar[PolicyName] = "label-logprob"

    def decide(self, out: LabelOut | OrdinalOut, chat: ChatResult) -> Verdict | None:
        if isinstance(out, LabelOut):
            probabilities = label_probabilities(chat.content, chat.logprobs)
            return None if probabilities is None else self.threshold(probabilities)
        probabilities = score_sign_probabilities(chat.content, chat.logprobs)
        return None if probabilities is None else self.threshold(probabilities, score=out.score)

    def threshold(self, probabilities: Mapping[Sentiment, float], score: int | None = None) -> Verdict:
        return _threshold(
            self.name, normalise(probabilities), self.min_p, score=score, tau=self.directional_tau
        )


@dataclass(frozen=True, slots=True)
class OrdinalPolicy:
    """Directional only when |score| >= t; a zero score is a confident neutral, not an abstention."""

    t: int
    name: ClassVar[PolicyName] = "ordinal"

    def decide(self, score: int) -> Verdict:
        raw = score_sentiment(score)
        abstained = raw != "neutral" and abs(score) < self.t
        return Verdict(
            sentiment="neutral" if abstained else raw,
            abstained=abstained,
            policy=self.name,
            score=score,
            threshold=float(self.t),
        )


Policy = ProbabilityPolicy | LabelLogprobPolicy | OrdinalPolicy


def _min_p(spec: ModelSpec) -> float:
    if spec.min_p is None:
        raise ValueError(f"{spec.name}: policy {spec.policy} needs min_p")
    return spec.min_p


def policy_for(spec: ModelSpec, streaming: bool = False, *, endpoint: Endpoint = "signal") -> Policy:
    """Streams and /analyze always threshold the ordinal score: it is read mid-JSON as soon as it
    closes, with no dependency on logprobs or on where the score sits in the output."""
    if spec.kind == "decision":
        return ProbabilityPolicy(_min_p(spec), spec.directional_tau)
    if streaming or endpoint == "analyze" or spec.policy == "ordinal":
        return OrdinalPolicy(spec.ordinal_t)
    return LabelLogprobPolicy(_min_p(spec), spec.directional_tau)
