from __future__ import annotations

import json
from typing import Any

import pytest

from veritrade.inference.types import Sentiment
from veritrade_evals.articles import LabelOutput
from veritrade_evals.articles_validate import (
    agreement,
    check_adjudicate,
    check_gold,
    check_label,
    cohen_kappa,
    needs_adjudication,
    number_set,
    shares_run,
    shingles,
)

ID = "art-000000000001"
BODY_LINE = (
    "Velmora Robotics said quarterly revenue rose 28% to $1,250 million as factory orders for its "
    "welding arms "
    "outpaced the company's own forecast, and management lifted the full-year outlook to $2.4 billion."
)
LINES = [
    "Menu",
    "Velmora lifts outlook",
    BODY_LINE,
    BODY_LINE.replace("28%", "31%"),
    BODY_LINE.replace("Velmora", "It"),
    BODY_LINE.replace("welding", "painting"),
    BODY_LINE.replace("factory", "export"),
]
FACTS = [
    "Velmora Robotics grew quarterly sales 28% to $1,250 million.",
    "Management raised the annual outlook to $2.4 billion.",
    "Orders for welding arms beat the internal forecast.",
]


def gold(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "id": ID,
        "usable": True,
        "title_line": 2,
        "body_ranges": [[3, 7]],
        "key_facts": FACTS,
        "entity": "Velmora Robotics",
        "ticker": "VLM",
        "provisional_label": "bullish",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_gold_builds_the_body_and_title() -> None:
    check = check_gold(ID, gold(), LINES)
    assert check.ok, check.codes
    assert check.usable
    assert check.title == "Velmora lifts outlook"
    assert check.body == "\n\n".join(LINES[2:7])


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"id": "art-ffffffffffff"}, "id_mismatch"),
        ({"body_ranges": [[7, 3]]}, "range_inverted"),
        ({"body_ranges": [[3, 99]]}, "range_out_of_bounds"),
        ({"body_ranges": [[0, 3]]}, "range_out_of_bounds"),
        ({"body_ranges": [[3, 5], [5, 7]]}, "ranges_overlap"),
        ({"title_line": 99}, "title_out_of_bounds"),
        ({"title_line": 3}, "title_in_body"),
        ({"body_ranges": [[3, 3]]}, "body_too_short"),
        ({"key_facts": FACTS[:2]}, "key_facts_count"),
        ({"key_facts": [*FACTS, "a", "b", "c"]}, "key_facts_count"),
        ({"key_facts": [*FACTS[:2], " ".join(["word"] * 26)]}, "key_fact_too_long"),
        ({"key_facts": [*FACTS[:2], "Revenue rose 29% in the quarter."]}, "key_fact_number_ungrounded"),
        (
            {
                "key_facts": [
                    *FACTS[:2],
                    "quarterly revenue rose 28% to $1,250 million as factory orders for its",
                ]
            },
            "key_fact_verbatim",
        ),
        ({"key_facts": [*FACTS[:2], "  "]}, "key_fact_empty"),
        ({"entity": None}, "missing:entity"),
        ({"title_line": None}, "missing:title_line"),
        ({"provisional_label": None}, "missing:provisional_label"),
        ({"body_ranges": []}, "missing:body_ranges"),
        ({"usable": False}, "missing:reject_reason"),
    ],
)
def test_gold_rules(overrides: dict[str, Any], code: str) -> None:
    check = check_gold(ID, gold(**overrides), LINES)
    assert code in check.codes
    assert not check.ok


def test_gold_schema_errors_report_location_not_text() -> None:
    check = check_gold(ID, gold(provisional_label="very bullish", extra_field="secret headline"), LINES)
    assert any(code.startswith("schema:provisional_label:") for code in check.codes)
    assert any(code.startswith("schema:extra_field:") for code in check.codes)
    assert not any("secret" in code or "very bullish" in code for code in check.codes)
    assert check_gold(ID, "not json", LINES).codes[0].startswith("schema:")


def test_gold_missing_inputs() -> None:
    assert check_gold(ID, None, LINES).codes == ("missing_output",)
    assert "missing_packet_lines" in check_gold(ID, gold(), None).codes


def test_rejected_gold_is_valid_but_unusable() -> None:
    check = check_gold(ID, gold(usable=False, reject_reason="live_blog", body_ranges=[]), LINES)
    assert check.ok
    assert not check.usable
    assert (
        "schema:reject_reason:literal_error"
        in check_gold(ID, gold(usable=False, reject_reason="meh"), LINES).codes
    )


def test_number_set_canonicalises() -> None:
    assert number_set("$1,250.50 and 1.5 and 28% in Q3 2026") == {"1250.5", "1.5", "28", "3", "2026"}
    assert number_set("1.50") == number_set("1.5")
    assert number_set("100") == {"100"}


def test_shingles_need_ten_consecutive_words() -> None:
    body = "one two three four five six seven eight nine ten eleven"
    runs = shingles(body)
    assert shares_run("One two three four five six seven eight nine ten", runs)
    assert not shares_run("one two three four five six seven eight nine", runs)
    assert not shares_run("two one three four five six seven eight nine ten", runs)


def label(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "id": ID,
        "label": "bullish",
        "score": 2,
        "entity": "Velmora Robotics",
        "ticker": None,
        "rationale": "Higher sales and a raised annual outlook point to stronger earnings.",
        "ambiguous": False,
        "confidence": 4,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"label": "bullish", "score": 0}, "score_label_mismatch"),
        ({"label": "bearish", "score": 1}, "score_label_mismatch"),
        ({"label": "neutral", "score": -1}, "score_label_mismatch"),
        ({"rationale": " ".join(["because"] * 51)}, "rationale_too_long"),
        (
            {"rationale": "So quarterly revenue rose 28% to $1,250 million as factory orders for its arms."},
            "rationale_verbatim",
        ),
        ({"id": "art-ffffffffffff"}, "id_mismatch"),
    ],
)
def test_label_rules(overrides: dict[str, Any], code: str) -> None:
    body = "\n\n".join(LINES[2:7])
    assert code in check_label(ID, label(**overrides), body).codes


def test_label_schema_and_missing_gold() -> None:
    assert check_label(ID, label(), "\n\n".join(LINES[2:7])).ok
    assert any(c.startswith("schema:score:") for c in check_label(ID, label(score=3), "body").codes)
    assert any(c.startswith("schema:confidence:") for c in check_label(ID, label(confidence=0), "body").codes)
    assert "gold_missing" in check_label(ID, label(), None).codes


def test_adjudication_must_be_requested() -> None:
    raw = json.dumps(
        {
            "id": ID,
            "decision": "keep",
            "label": "neutral",
            "score": 0,
            "entity": "Velmora",
            "rationale": "Balanced.",
        }
    )
    assert check_adjudicate(ID, raw, "body text", {ID}).ok
    assert "not_requested" in check_adjudicate(ID, raw, "body text", set()).codes
    bad = json.dumps({**json.loads(raw), "decision": "maybe"})
    assert any(c.startswith("schema:decision:") for c in check_adjudicate(ID, bad, "body", {ID}).codes)


def test_cohen_kappa_known_values() -> None:
    bull: Sentiment = "bullish"
    bear: Sentiment = "bearish"
    neut: Sentiment = "neutral"
    a = [bull] * 20 + [bear] * 15 + [neut] * 15
    b = [bull] * 15 + [neut] * 5 + [bear] * 15 + [neut] * 15
    # observed 0.9, expected (20*15 + 15*15 + 15*20) / 2500 = 0.33
    assert cohen_kappa(a, b) == pytest.approx((0.9 - 0.33) / 0.67)
    assert cohen_kappa(["neutral", "neutral"], ["neutral", "neutral"]) == 1.0
    assert cohen_kappa([], []) is None
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa(["neutral"], [])


def _out(label: Sentiment, score: int, ambiguous: bool = False) -> LabelOutput:
    return LabelOutput(
        id=ID, label=label, score=score, entity="X", rationale="r", ambiguous=ambiguous, confidence=3
    )


def test_agreement_flags_disagreement_and_ambiguity() -> None:
    pairs = {
        "art-1": (_out("bullish", 2), _out("bullish", 1)),
        "art-2": (_out("bullish", 1), _out("neutral", 0)),
        "art-3": (_out("bearish", -1), _out("bearish", -2, ambiguous=True)),
    }
    stats = agreement(pairs)
    assert stats.n == 3
    assert stats.agreed == 2
    assert stats.raw == pytest.approx(2 / 3)
    assert stats.to_adjudicate == ("art-2", "art-3")
    assert needs_adjudication(*pairs["art-3"])
    assert not needs_adjudication(*pairs["art-1"])
