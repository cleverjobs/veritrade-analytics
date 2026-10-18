from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from veritrade.inference.confidence import (
    LabelLogprobPolicy,
    OrdinalPolicy,
    ProbabilityPolicy,
    label_probabilities,
    normalise,
    policy_for,
    score_sign_probabilities,
    winner,
)
from veritrade.inference.ollama import parse_logprobs
from veritrade.inference.schemas import LabelOut, OrdinalOut
from veritrade.inference.types import ChatResult, Sentiment, TokenLogprob
from veritrade.models_registry import ModelSpec

from .conftest import lp, plain


def _chat(content: str, entries: Sequence[dict[str, object]]) -> ChatResult:
    return ChatResult(content=content, wall_ms=1.0, logprobs=parse_logprobs(list(entries)))


def _label_chat(first: str, rest: Sequence[str], alternatives: Sequence[tuple[str, float]]) -> ChatResult:
    head = ['{"', "label", '":', ' "']
    content = "".join([*head, first, *rest])
    p = dict(alternatives)[first]
    return _chat(content, [*plain(head), lp(first, p, alternatives), *plain(rest)])


# --- probability policy ---------------------------------------------------------------------


def test_probability_policy_calls_above_threshold() -> None:
    probabilities: dict[Sentiment, float] = {"bullish": 0.7, "neutral": 0.2, "bearish": 0.1}
    verdict = ProbabilityPolicy(0.6).decide(probabilities)
    assert (verdict.sentiment, verdict.abstained, verdict.policy) == ("bullish", False, "probability")
    assert verdict.confidence == pytest.approx(0.7)


def test_probability_policy_abstains_to_neutral_but_keeps_the_distribution() -> None:
    probabilities: dict[Sentiment, float] = {"bullish": 0.45, "neutral": 0.35, "bearish": 0.2}
    verdict = ProbabilityPolicy(0.5).decide(probabilities)
    assert (verdict.sentiment, verdict.abstained) == ("neutral", True)
    assert verdict.probabilities == pytest.approx(probabilities)
    assert winner(verdict.probabilities or {}) == "bullish"


def test_threshold_is_inclusive() -> None:
    assert not ProbabilityPolicy(0.5).decide({"bearish": 0.5, "neutral": 0.3, "bullish": 0.2}).abstained


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"bullish": 0.4, "neutral": 0.4, "bearish": 0.2}, "neutral"),
        ({"bearish": 0.4, "neutral": 0.4, "bullish": 0.2}, "neutral"),
        ({"bullish": 0.45, "bearish": 0.45, "neutral": 0.1}, "bullish"),
        ({"bullish": 1 / 3, "bearish": 1 / 3, "neutral": 1 / 3}, "neutral"),
    ],
)
def test_ties_resolve_neutral_first(probabilities: dict[str, float], expected: str) -> None:
    assert winner(probabilities) == expected  # type: ignore[arg-type]


def test_normalise() -> None:
    assert normalise({"bullish": 2.0, "neutral": 1.0, "bearish": 1.0}) == {
        "neutral": 0.25,
        "bullish": 0.5,
        "bearish": 0.25,
    }
    with pytest.raises(ValueError, match="zero"):
        normalise({"bullish": 0.0})


# --- ordinal policy -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "score", "sentiment", "abstained"),
    [
        (1, 2, "bullish", False),
        (1, 1, "bullish", False),
        (1, 0, "neutral", False),
        (1, -1, "bearish", False),
        (2, 1, "neutral", True),
        (2, -1, "neutral", True),
        (2, -2, "bearish", False),
        (2, 0, "neutral", False),
    ],
)
def test_ordinal_policy(t: int, score: int, sentiment: str, abstained: bool) -> None:
    verdict = OrdinalPolicy(t).decide(score)
    assert (verdict.sentiment, verdict.abstained, verdict.score) == (sentiment, abstained, score)
    assert verdict.policy == "ordinal"
    assert verdict.probabilities is None


# --- label-logprob policy ------------------------------------------------------------------


def test_label_probabilities_drop_grammar_invalid_tokens_and_renormalise() -> None:
    chat = _label_chat(
        "bull",
        ["ish", '"}'],
        [("bull", 0.5), ("neutral", 0.3), ("bear", 0.1), ("positive", 0.1)],
    )
    probabilities = label_probabilities(chat.content, chat.logprobs)
    assert probabilities == pytest.approx({"neutral": 0.3 / 0.9, "bullish": 0.5 / 0.9, "bearish": 0.1 / 0.9})


def test_alternatives_for_the_same_label_add_up() -> None:
    chat = _label_chat("neutral", ['"}'], [("neutral", 0.3), ("ne", 0.2), ("bull", 0.4), ("bear", 0.1)])
    probabilities = label_probabilities(chat.content, chat.logprobs) or {}
    assert probabilities["neutral"] == pytest.approx(0.5)


def test_reported_argmax_wins_over_the_generated_label() -> None:
    # Seen live on qwen3.5:0.8b: greedy emitted "neutral" while the reported distribution favoured "bear".
    chat = _label_chat("neutral", ['"}'], [("bear", 0.59), ("neutral", 0.35), ("bull", 0.06)])
    verdict = LabelLogprobPolicy(0.5).decide(LabelOut(label="neutral"), chat)
    assert verdict is not None
    assert (verdict.sentiment, verdict.abstained) == ("bearish", False)


def test_value_token_that_also_carries_the_opening_quote() -> None:
    content = '{"label": "bearish"}'
    tokens = [
        *plain(['{"', "label", '":']),
        lp(' "bear', 0.7, [(' "bear', 0.7), (' "neutral', 0.2), (' "bull', 0.05), ('"bull', 0.05)]),
        *plain(['ish"}']),
    ]
    probabilities = label_probabilities(content, parse_logprobs(tokens))
    # '"bull' lacks the leading space the chosen token carries, so it is a different path and dropped.
    assert probabilities == pytest.approx(
        {"neutral": 0.2 / 0.95, "bullish": 0.05 / 0.95, "bearish": 0.7 / 0.95}
    )


def test_label_verdict_abstains_below_min_p() -> None:
    chat = _label_chat("bull", ["ish", '"}'], [("bull", 0.45), ("neutral", 0.35), ("bear", 0.2)])
    verdict = LabelLogprobPolicy(0.5).decide(LabelOut(label="bullish"), chat)
    assert verdict is not None
    assert (verdict.sentiment, verdict.abstained, verdict.policy) == ("neutral", True, "label-logprob")
    assert verdict.probabilities
    assert verdict.probabilities["bullish"] == pytest.approx(0.45)


def test_colliding_first_token_returns_none() -> None:
    chat = _label_chat("b", ["ullish", '"}'], [("b", 0.8), ("neutral", 0.2)])
    assert label_probabilities(chat.content, chat.logprobs) is None
    assert LabelLogprobPolicy(0.5).decide(LabelOut(label="bullish"), chat) is None


def test_ambiguous_alternatives_with_real_mass_return_none() -> None:
    heavy = _label_chat("bull", ["ish", '"}'], [("bull", 0.6), ("b", 0.3), ("neutral", 0.1)])
    assert label_probabilities(heavy.content, heavy.logprobs) is None
    light = _label_chat("bull", ["ish", '"}'], [("bull", 0.6), ("b", 0.01), ("neutral", 0.39)])
    assert label_probabilities(light.content, light.logprobs) == pytest.approx(
        {"neutral": 0.39 / 0.99, "bullish": 0.6 / 0.99, "bearish": 0.0}
    )


def test_alternatives_ending_at_the_value_start_carry_no_class() -> None:
    # Seen live on qwen3.5:0.8b: a bare " " held 27% of the mass next to " -" before "-2".
    content = '{"score": -2}'
    tokens = [
        *plain(['{"', "score", '":']),
        lp(" -", 0.7, [(" -", 0.7), (" ", 0.27), (" 0", 0.03)]),
        *plain(["2}"]),
    ]
    probabilities = score_sign_probabilities(content, parse_logprobs(tokens))
    assert probabilities == pytest.approx({"neutral": 0.03 / 0.73, "bullish": 0.0, "bearish": 0.7 / 0.73})


def test_tokens_that_do_not_tile_the_content_return_none() -> None:
    chat = _chat('{"label": "bullish"}', plain(['{"', "lab", '": "', "bullish", '"}']))
    assert label_probabilities(chat.content, chat.logprobs) is None
    assert label_probabilities('{"label": "bullish"}', None) is None


@pytest.mark.parametrize(
    ("score", "first", "alternatives", "expected"),
    [
        (-1, "-", [("-", 0.6), ("0", 0.3), ("1", 0.05), ("2", 0.05)], {"bearish": 0.6, "neutral": 0.3}),
        (2, "2", [("2", 0.5), ("1", 0.3), ("-", 0.1), ("3", 0.1)], {"bullish": 0.8 / 0.9}),
        (0, "0", [("0", 0.5), ("-1", 0.25), ("1", 0.25)], {"neutral": 0.5, "bearish": 0.25}),
    ],
)
def test_score_sign_probabilities(
    score: int, first: str, alternatives: list[tuple[str, float]], expected: dict[Sentiment, float]
) -> None:
    head = ['{"', "score", '":', " "]
    rest = [str(score)[len(first) :], "}"] if len(str(score)) > len(first) else ["}"]
    content = "".join([*head, first, *rest])
    chat = _chat(content, [*plain(head), lp(first, dict(alternatives)[first], alternatives), *plain(rest)])
    probabilities = score_sign_probabilities(chat.content, chat.logprobs) or {}
    for sentiment, p in expected.items():
        assert probabilities[sentiment] == pytest.approx(p)
    verdict = LabelLogprobPolicy(0.5).decide(OrdinalOut(score=score), chat)
    assert verdict is not None
    assert verdict.score == score


def test_policy_for(label_spec: ModelSpec, decision_spec: ModelSpec, ordinal_spec: ModelSpec) -> None:
    assert isinstance(policy_for(decision_spec), ProbabilityPolicy)
    assert isinstance(policy_for(decision_spec, True), ProbabilityPolicy)
    assert isinstance(policy_for(label_spec), LabelLogprobPolicy)
    assert policy_for(label_spec, True) == OrdinalPolicy(1)
    assert policy_for(label_spec, endpoint="analyze") == OrdinalPolicy(1)
    assert policy_for(ordinal_spec, endpoint="analyze") == OrdinalPolicy(2)


def test_parse_logprobs_keeps_alternatives() -> None:
    parsed = parse_logprobs([lp("bull", 0.5, [("bull", 0.5), ("bear", 0.25)])])
    assert parsed == (
        TokenLogprob("bull", math.log(0.5), (("bull", math.log(0.5)), ("bear", math.log(0.25)))),
    )
