from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

Sentiment = Literal["bullish", "bearish", "neutral"]
SENTIMENTS: tuple[Sentiment, ...] = get_args(Sentiment)
PolicyName = Literal["probability", "label-logprob", "ordinal"]
ModelKind = Literal["generative", "decision"]
Endpoint = Literal["signal", "summary", "analyze"]
FieldOrder = Literal["sentiment-first", "summary-first"]


@dataclass(frozen=True, slots=True)
class OllamaTimings:
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    total_ms: float | None = None


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    wall_ms: float
    timings: OllamaTimings = field(default_factory=OllamaTimings)
    logprobs: tuple[TokenLogprob, ...] | None = None
    ttft_ms: float | None = None


@dataclass(frozen=True, slots=True)
class DecisionResult:
    choice: Sentiment
    probabilities: dict[Sentiment, float]
    confidence: float
    wall_ms: float
    input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    sentiment: Sentiment
    abstained: bool
    policy: PolicyName
    confidence: float | None = None
    score: int | None = None
    probabilities: dict[Sentiment, float] | None = None
