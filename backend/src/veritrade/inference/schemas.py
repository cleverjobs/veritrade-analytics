"""Structured outputs. Property order is the generation order Ollama's grammar enforces."""

from __future__ import annotations

import copy
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from veritrade.inference.types import FieldOrder, Sentiment

Score = Literal[-2, -1, 0, 1, 2]


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LabelOut(_Out):
    label: Sentiment


class OrdinalOut(_Out):
    score: Score


class SummaryOut(_Out):
    summary: str = Field(min_length=1)


class AnalyzeSentimentFirst(_Out):
    score: Score
    entity: str
    summary: str = Field(min_length=1)


class AnalyzeSummaryFirst(_Out):
    summary: str = Field(min_length=1)
    entity: str
    score: Score


AnalyzeOut = AnalyzeSentimentFirst | AnalyzeSummaryFirst
StructuredOut = LabelOut | OrdinalOut | SummaryOut | AnalyzeSentimentFirst | AnalyzeSummaryFirst

ANALYZE_MODELS: dict[FieldOrder, type[AnalyzeSentimentFirst] | type[AnalyzeSummaryFirst]] = {
    "sentiment-first": AnalyzeSentimentFirst,
    "summary-first": AnalyzeSummaryFirst,
}


def _strip(node: Any) -> Any:
    # Titles and descriptions only cost grammar states; the prompt carries the semantics.
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in ("title", "description")}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


@cache
def _format_cached(model: type[BaseModel]) -> dict[str, Any]:
    schema = _strip(model.model_json_schema())
    schema["additionalProperties"] = False
    schema["required"] = list(model.model_fields)
    return dict(schema)


def ollama_format(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for Ollama's `format`; `properties` keeps the model's field order."""
    return copy.deepcopy(_format_cached(model))


def score_sentiment(score: int) -> Sentiment:
    return "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
