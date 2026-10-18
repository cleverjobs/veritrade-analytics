from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from veritrade.inference.schemas import (
    ANALYZE_MODELS,
    AnalyzeSentimentFirst,
    AnalyzeSummaryFirst,
    LabelOut,
    OrdinalOut,
    SummaryOut,
    ollama_format,
    score_sentiment,
)


def _walk(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [*node, *(k for v in node.values() for k in _walk(v))]
    if isinstance(node, list):
        return [k for v in node for k in _walk(v)]
    return []


@pytest.mark.parametrize(
    ("model", "order"),
    [
        (AnalyzeSentimentFirst, ["score", "entity", "summary"]),
        (AnalyzeSummaryFirst, ["summary", "entity", "score"]),
        (LabelOut, ["label"]),
        (OrdinalOut, ["score"]),
        (SummaryOut, ["summary"]),
    ],
)
def test_property_order_is_generation_order(model: type[Any], order: list[str]) -> None:
    schema = ollama_format(model)
    assert list(schema["properties"]) == order
    assert schema["required"] == order
    assert schema["additionalProperties"] is False
    assert "title" not in _walk(schema)
    # The wire format is what Ollama compiles into a grammar, so check the serialised order too.
    wire = json.dumps(schema)
    positions = [wire.index(f'"{name}":') for name in order]
    assert positions == sorted(positions)


def test_enums_constrain_label_and_score() -> None:
    assert ollama_format(LabelOut)["properties"]["label"]["enum"] == ["bullish", "bearish", "neutral"]
    score = ollama_format(OrdinalOut)["properties"]["score"]
    assert score == {"enum": [-2, -1, 0, 1, 2], "type": "integer"}


def test_format_is_a_fresh_copy() -> None:
    first = ollama_format(AnalyzeSentimentFirst)
    first["properties"]["score"]["enum"].append(9)
    assert 9 not in ollama_format(AnalyzeSentimentFirst)["properties"]["score"]["enum"]


def test_field_order_map() -> None:
    assert ANALYZE_MODELS["sentiment-first"] is AnalyzeSentimentFirst
    assert ANALYZE_MODELS["summary-first"] is AnalyzeSummaryFirst


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (LabelOut, '{"label": "positive"}'),
        (OrdinalOut, '{"score": 3}'),
        (OrdinalOut, '{"score": 1, "extra": true}'),
        (SummaryOut, '{"summary": ""}'),
        (AnalyzeSentimentFirst, '{"score": 1, "summary": "x"}'),
    ],
)
def test_outputs_are_validated(model: type[Any], payload: str) -> None:
    with pytest.raises(ValidationError):
        model.model_validate_json(payload)


def test_score_sentiment() -> None:
    assert [score_sentiment(s) for s in (-2, -1, 0, 1, 2)] == [
        "bearish",
        "bearish",
        "neutral",
        "bullish",
        "bullish",
    ]
