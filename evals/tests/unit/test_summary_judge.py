from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from veritrade_evals.summary_judge import (
    Blinding,
    JudgePaths,
    blind,
    check_output,
    deblind,
    make_blinding,
    render_packet,
    text_sha,
)

ARTICLE = "art-000000000001"
BODY = (
    "Zentor Mills said on Tuesday that quarterly revenue climbed 12% to $410 million as demand for its "
    "industrial flour recovered across every region it serves.\n\n"
    "The company also lifted its full-year outlook and announced a new plant in Ohio."
)
SUMMARIES = {
    "summary/model-a": "Zentor Mills lifted revenue 12% to $410 million and raised its outlook.",
    "analyze/model-b": "Zentor Mills revenue rose 15% to $410 million.",
    "analyze/model-c": "Zentor Mills opened a plant.",
}


def judged(label: str, faithfulness: int = 5, notes: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "label": label,
        "faithfulness": faithfulness,
        "coverage": extra.get("coverage", 4),
        "hallucinations": notes or [],
        "leads_with_market_moving_fact": extra.get("leads", True),
    }


def output(blinding: Blinding, entries: list[dict[str, Any]] | None = None) -> str:
    body = entries if entries is not None else [judged(label) for label in blinding.labels]
    return json.dumps({"id": blinding.id, "summaries": body})


@pytest.fixture
def blinding() -> Blinding:
    return make_blinding(ARTICLE, "dev", SUMMARIES)


def check(raw: str | None, blinding: Blinding, summaries: dict[str, str] | None = None) -> tuple[str, ...]:
    return check_output(raw, blinding, article_text=BODY, summaries=summaries or SUMMARIES)[1]


# --- blinding -----------------------------------------------------------------------------------------------


def test_blinding_is_seeded_per_article_and_ignores_candidate_order() -> None:
    first = blind(ARTICLE, list(SUMMARIES))
    assert blind(ARTICLE, list(reversed(SUMMARIES))) == first
    assert list(first) == ["S1", "S2", "S3"]
    assert sorted(first.values()) == sorted(SUMMARIES)
    others = {tuple(blind(f"art-{i:012d}", list(SUMMARIES)).values()) for i in range(20)}
    assert len(others) > 1


def test_blinding_records_a_hash_of_each_shown_summary(blinding: Blinding) -> None:
    for label, key in blinding.labels.items():
        assert blinding.summary_sha[label] == text_sha(SUMMARIES[key])


def test_packet_shows_labels_but_never_the_candidates(blinding: Blinding, tmp_path: Path) -> None:
    out = JudgePaths(tmp_path).out(ARTICLE)
    packet = render_packet(blinding, "Zentor Mills beats", BODY, SUMMARIES, out)
    for label, key in blinding.labels.items():
        assert f"### {label}" in packet
        assert key not in packet
        assert SUMMARIES[key] in packet
    assert "model-" not in packet
    assert str(out) in packet
    assert "50-judge.md" in packet
    assert packet.index("### S1") < packet.index("### S2") < packet.index("### S3")


def test_deblind_maps_labels_back_to_candidates(blinding: Blinding) -> None:
    raw = output(blinding, [judged("S1"), judged("S2", 3, ["gives a growth rate the article does not"])])
    parsed, _ = check_output(raw, blinding, article_text=BODY, summaries=SUMMARIES)
    assert parsed is not None
    rows = {row.candidate: row for row in deblind(parsed, blinding)}
    assert rows[blinding.labels["S2"]].hallucinations == 1
    assert rows[blinding.labels["S2"]].faithfulness == 3
    assert rows[blinding.labels["S1"]].leads is True


# --- validation ---------------------------------------------------------------------------------------------


def test_valid_output_has_no_codes(blinding: Blinding) -> None:
    raw = output(
        blinding, [judged("S1"), judged("S2", 4), judged("S3", 2, ["names a plant opening as done"])]
    )
    assert check(raw, blinding) == ()


def test_missing_and_malformed_outputs(blinding: Blinding) -> None:
    assert check(None, blinding) == ("missing_output",)
    bad = json.dumps({"id": ARTICLE, "summaries": [judged("S1", faithfulness=7)]})
    assert any(code.startswith("schema:summaries.0.faithfulness") for code in check(bad, blinding))
    assert any(code.startswith("schema:") for code in check("not json", blinding))


def test_label_set_must_match_the_packet(blinding: Blinding) -> None:
    raw = output(blinding, [judged("S1"), judged("S1"), judged("S4")])
    codes = set(check(raw, blinding))
    assert {"S1:duplicate_label", "S4:unknown_label", "S2:missing_label", "S3:missing_label"} <= codes


def test_id_must_match(blinding: Blinding) -> None:
    raw = json.dumps({"id": "art-ffffffffffff", "summaries": [judged(label) for label in blinding.labels]})
    assert "id_mismatch" in check(raw, blinding)


def test_scores_and_notes_must_agree(blinding: Blinding) -> None:
    raw = output(blinding, [judged("S1", 5, ["adds a detail"]), judged("S2", 3), judged("S3", 4)])
    codes = set(check(raw, blinding))
    assert "S1:faithfulness_5_with_hallucinations" in codes
    assert "S2:low_faithfulness_without_hallucinations" in codes
    assert not any(code.startswith("S3:") for code in codes)


def test_notes_must_be_short_and_in_own_words(blinding: Blinding) -> None:
    quoted = "quarterly revenue climbed 12% to $410 million as demand for its industrial flour recovered"
    long = " ".join(["word"] * 26)
    from_summary = "Zentor Mills lifted revenue 12% to $410 million and raised its outlook."
    raw = output(blinding, [judged("S1", 2, [quoted, long]), judged("S2", 2, [from_summary]), judged("S3")])
    codes = set(check(raw, blinding))
    assert {"S1:hallucination_verbatim", "S1:hallucination_too_long", "S2:hallucination_verbatim"} <= codes


def test_a_changed_summary_makes_the_packet_stale(blinding: Blinding) -> None:
    changed = dict(SUMMARIES)
    changed["analyze/model-c"] = "Zentor Mills opened a plant in Ohio."
    stale = next(label for label, key in blinding.labels.items() if key == "analyze/model-c")
    assert check(output(blinding), blinding, changed) == (f"{stale}:stale_summary",)
    dropped = {k: v for k, v in SUMMARIES.items() if k != "analyze/model-c"}
    assert f"{stale}:stale_summary" in check(output(blinding), blinding, dropped)
