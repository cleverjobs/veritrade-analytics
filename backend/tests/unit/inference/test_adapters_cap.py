"""Served summaries never exceed 3 sentences and 60 words, whatever the model wrote."""

from __future__ import annotations

import json

import pytest
import respx

from veritrade.inference.adapters import (
    ELLIPSIS,
    SUMMARY_MAX_WORDS,
    AnalyzeResult,
    SummaryResult,
    analyze,
    analyze_stream,
    cap_summary,
    summarize,
    summarize_stream,
)
from veritrade.inference.incremental import SummaryText
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import ModelSpec
from veritrade.prompts import registry as prompts

from .conftest import chat_body, content_chunks, stream_response

ARTICLE = "Acme Corp raised its outlook after a strong quarter."
FOUR = "Acme raised its outlook. Sales rose 8%. Margins held at 31%. The stock rose 4%."
THREE = "Acme raised its outlook. Sales rose 8%. Margins held at 31%."
LONG_SENTENCE = "Acme " + " ".join(f"w{i}," for i in range(70)) + " end."


def words(n: int, tag: str) -> str:
    return " ".join([tag.capitalize(), *[tag] * (n - 2), f"{tag}."])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (THREE, THREE),
        (f"  {THREE}\n", THREE),
        (FOUR, THREE),
        (
            f"{words(30, 'alpha')} {words(25, 'beta')} {words(10, 'gamma')}",
            f"{words(30, 'alpha')} {words(25, 'beta')}",
        ),
        (f"{words(60, 'alpha')} {words(5, 'beta')}", words(60, "alpha")),
        (
            "Shares of U.S. Steel rose 5%. Profit fell. Guidance held. Debt fell.",
            "Shares of U.S. Steel rose 5%. Profit fell. Guidance held.",
        ),
        ("", ""),
    ],
    ids=["within", "strips", "four-sentences", "word-cap-drops-third", "exactly-60", "abbreviation", "empty"],
)
def test_cap_summary_keeps_whole_leading_sentences(text: str, expected: str) -> None:
    assert cap_summary(text) == expected


def test_an_overlong_first_sentence_is_cut_at_a_word_with_an_ellipsis() -> None:
    capped = cap_summary(LONG_SENTENCE + " Second sentence.")
    assert capped.endswith(ELLIPSIS)
    assert not capped.endswith("," + ELLIPSIS)
    assert len(capped.split()) == SUMMARY_MAX_WORDS
    assert capped.startswith("Acme w0, w1,")


def test_caps_are_parameters() -> None:
    assert cap_summary(FOUR, max_sentences=2) == "Acme raised its outlook. Sales rose 8%."
    assert cap_summary(FOUR, max_words=5) == "Acme raised its outlook."


async def test_summarize_caps_without_flagging_truncation(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    mock_ollama.post("/api/chat").respond(json=chat_body(json.dumps({"summary": FOUR})))
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == (THREE, False)
    assert FOUR in result.chat.content


async def test_summarize_stream_caps_the_done_result_only(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"summary": "', FOUR[:30], FOUR[30:], '"}']
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces)))
    items = [
        item
        async for item in summarize_stream(
            client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE
        )
    ]
    assert "The stock rose 4%" in "".join(i.delta for i in items if isinstance(i, SummaryText))
    final = items[-1]
    assert isinstance(final, SummaryResult)
    assert (final.output.summary, final.truncated) == (THREE, False)


async def test_analyze_caps_and_keeps_score_and_entity(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    content = json.dumps({"score": 2, "entity": "Acme Corp", "summary": FOUR})
    mock_ollama.post("/api/chat").respond(json=chat_body(content))
    result = await analyze(client, ordinal_spec, prompts.get("analyze", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.output.score, result.entity, result.truncated) == (
        THREE,
        2,
        "Acme Corp",
        False,
    )


async def test_analyze_stream_caps_the_done_result(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    pieces = ['{"score": 1, "entity": "Acme", "summary": "', LONG_SENTENCE, '"}']
    mock_ollama.post("/api/chat").mock(return_value=stream_response(content_chunks(pieces)))
    items = [
        item
        async for item in analyze_stream(
            client, ordinal_spec, prompts.get("analyze", ordinal_spec.name), ARTICLE
        )
    ]
    final = items[-1]
    assert isinstance(final, AnalyzeResult)
    assert final.output.summary.endswith(ELLIPSIS)
    assert len(final.output.summary.split()) == SUMMARY_MAX_WORDS
    assert final.output.score == 1


async def test_salvaged_output_is_capped_and_still_flagged(
    client: OllamaClient, mock_ollama: respx.MockRouter, ordinal_spec: ModelSpec
) -> None:
    cut = '{"summary": "' + FOUR + " Analysts were div"
    mock_ollama.post("/api/chat").respond(json=chat_body(cut, done_reason="length", eval_count=160))
    result = await summarize(client, ordinal_spec, prompts.get("summary", ordinal_spec.name), ARTICLE)
    assert (result.output.summary, result.truncated) == (THREE, True)
