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


def test_min_p_zero_never_abstains() -> None:
    verdict = ProbabilityPolicy(0.0).decide({"bullish": 0.34, "neutral": 0.33, "bearish": 0.33})
    assert (verdict.sentiment, verdict.abstained) == ("bullish", False)


@pytest.mark.parametrize(
    ("probabilities", "expected", "abstained"),
    [
        # Neutral argmax, directional class at or above tau: the directional class is called.
        ({"neutral": 0.6, "bullish": 0.3, "bearish": 0.1}, "bullish", False),
        ({"neutral": 0.55, "bullish": 0.1, "bearish": 0.35}, "bearish", False),
        # Below tau the argmax + min_p rule stands.
        ({"neutral": 0.7, "bullish": 0.2, "bearish": 0.1}, "neutral", False),
        ({"neutral": 0.4, "bullish": 0.29, "bearish": 0.31}, "bearish", False),
        ({"neutral": 0.45, "bullish": 0.28, "bearish": 0.27}, "neutral", True),
        # A directional argmax is never touched by tau, even under min_p.
        ({"bullish": 0.45, "neutral": 0.4, "bearish": 0.15}, "neutral", True),
    ],
)
def test_directional_tau_only_overrides_a_neutral_argmax(
    probabilities: dict[Sentiment, float], expected: str, abstained: bool
) -> None:
    verdict = ProbabilityPolicy(0.5, directional_tau=0.3).decide(probabilities)
    assert (verdict.sentiment, verdict.abstained, verdict.policy) == (expected, abstained, "probability")
    assert verdict.probabilities == pytest.approx(probabilities)


def test_directional_tau_ties_between_directions_go_bullish() -> None:
    verdict = ProbabilityPolicy(0.5, directional_tau=0.3).decide(
        {"neutral": 0.4, "bullish": 0.3, "bearish": 0.3}
    )
    assert verdict.sentiment == "bullish"


def test_policy_for_carries_tau_and_leaves_decision_models_alone_when_unset(
    decision_spec: ModelSpec, label_spec: ModelSpec
) -> None:
    neutral_lean: dict[Sentiment, float] = {"neutral": 0.62, "bullish": 0.33, "bearish": 0.05}
    plain = policy_for(decision_spec)
    assert plain == ProbabilityPolicy(0.5, None)
    assert isinstance(plain, ProbabilityPolicy)
    assert plain.decide(neutral_lean).sentiment == "neutral"
    with_tau = policy_for(decision_spec.model_copy(update={"directional_tau": 0.3}))
    assert isinstance(with_tau, ProbabilityPolicy)
    assert with_tau.decide(neutral_lean).sentiment == "bullish"
    label = policy_for(label_spec.model_copy(update={"directional_tau": 0.3, "min_p": 0.0}))
    assert label == LabelLogprobPolicy(0.0, 0.3)
    assert isinstance(label, LabelLogprobPolicy)
    chat = _label_chat("neutral", ['"}'], [("neutral", 0.62), ("bull", 0.33), ("bear", 0.05)])
    verdict = label.decide(LabelOut(label="neutral"), chat)
    assert verdict is not None
    assert (verdict.sentiment, verdict.abstained, verdict.policy) == ("bullish", False, "label-logprob")


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


def _score(content: str, tokens: Sequence[dict[str, object]]) -> dict[Sentiment, float] | None:
    return score_sign_probabilities(content, parse_logprobs(list(tokens)))


def test_fused_space_minus_with_a_heavy_bare_space_alternative_hides_the_sign() -> None:
    # Seen live on qwen3.5:0.8b: " -" 0.70 next to a bare " " 0.28, whose continuation is never observed.
    tokens = [*plain(['{"', "score", '":']), lp(" -", 0.7, [(" -", 0.7), (" ", 0.27), (" 0", 0.03)])]
    assert _score('{"score": -2}', [*tokens, *plain(["2}"])]) is None


def test_fused_space_minus_counts_as_negative() -> None:
    first = lp(" -", 0.95, [(" -", 0.95), (" 0", 0.03), (" ", 0.02)])
    probabilities = _score('{"score": -1}', [*plain(['{"', "score", '":']), first, *plain(["1", "}"])])
    assert probabilities == pytest.approx({"neutral": 0.03 / 0.98, "bullish": 0.0, "bearish": 0.95 / 0.98})


def test_a_generated_bare_space_defers_to_the_digit_and_keeps_its_signed_alternatives() -> None:
    # The bearish mass sits on " -" one token before the digit that decides the other classes.
    space = lp(" ", 0.9, [(" ", 0.9), (" -", 0.08), (' "', 0.02)])
    digit = lp("1", 0.6, [("1", 0.6), ("2", 0.3), ("0", 0.1)])
    probabilities = _score(
        '{"score": 1}', [*plain(['{"', "score"]), lp('":', 1.0), space, digit, *plain(["}"])]
    )
    total = 0.08 + 0.9 * 0.9 + 0.9 * 0.1
    assert probabilities == pytest.approx(
        {"neutral": 0.9 * 0.1 / total, "bullish": 0.9 * 0.9 / total, "bearish": 0.08 / total}
    )


def test_sign_fused_into_the_colon_token() -> None:
    colon = lp('":-', 0.9, [('":-', 0.9), ('":0', 0.06), ('": ', 0.04)])
    probabilities = _score('{"score":-2}', [*plain(['{"', "score"]), colon, *plain(["2", "}"])])
    assert probabilities == pytest.approx({"neutral": 0.06 / 0.96, "bullish": 0.0, "bearish": 0.9 / 0.96})


def test_colon_step_alternatives_that_already_carry_the_sign_count() -> None:
    colon = lp('":', 0.95, [('":', 0.95), ('":-', 0.05)])
    first = lp("0", 0.8, [("0", 0.8), ("1", 0.2)])
    probabilities = _score('{"score":0}', [*plain(['{"', "score"]), colon, first, *plain(["}"])])
    total = 0.05 + 0.95
    assert probabilities == pytest.approx(
        {"neutral": 0.95 * 0.8 / total, "bullish": 0.95 * 0.2 / total, "bearish": 0.05 / total}
    )


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
