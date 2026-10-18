from __future__ import annotations

import codecs
import json
from collections.abc import Iterable

import pytest

from veritrade.errors import SchemaViolation
from veritrade.inference.incremental import (
    EntityClosed,
    JsonFieldScanner,
    ScanEvent,
    ScoreClosed,
    SummaryText,
)

PAYLOADS = [
    # sentiment-first, compact
    '{"score":2,"entity":"Velmora Robotics","summary":"Revenue rose 28% to $2.4 billion."}',
    # summary-first, spaced
    '{"summary": "Harbridge Bank lost $1.2 billion.", "entity": "Harbridge Bank", "score": -2}',
    # every simple escape
    r'{"score": 0, "entity": "A \"B\" C", "summary": "q\" b\\ s\/ n\n t\t b\b f\f r\r end"}',
    # \u escapes incl. a surrogate pair (U+1F4C8) and an escaped quote
    '{"summary": "\\u20ac5m caf\\u00e9 \\ud83d\\udcc8 up \\u0022ok\\u0022", '
    '"entity": "\\u00c9toile", "score": 1}',
    # raw non-ASCII text
    '{"score": -1, "entity": "Zürich Re", "summary": "Shares fell 3% in Zürich 📉; costs rose €40m."}',
    # pretty-printed the way Ollama formats schema output, with an unknown key scanned and ignored
    '{\n  "score": 0,\n  "note": null,\n  "entity": "",\n  "summary": "Crestmoor Foods meets on May 14."\n}',
    # a 1-char summary and a negative score last
    '{"summary":"x","entity":"Y","score":-1}',
]


def _run(chunks: Iterable[str]) -> list[ScanEvent]:
    scanner = JsonFieldScanner()
    events: list[ScanEvent] = []
    for chunk in chunks:
        events.extend(scanner.feed(chunk))
    scanner.close()
    assert scanner.done
    return events


def _check(payload: str, events: list[ScanEvent]) -> None:
    expected = json.loads(payload)
    scores = [e.score for e in events if isinstance(e, ScoreClosed)]
    entities = [e.entity for e in events if isinstance(e, EntityClosed)]
    text = "".join(e.delta for e in events if isinstance(e, SummaryText))
    assert scores == [expected["score"]]
    assert entities == [expected["entity"]]
    assert text == expected["summary"]
    assert all(e.delta for e in events if isinstance(e, SummaryText))
    first_text = next(i for i, e in enumerate(events) if isinstance(e, SummaryText))
    last_text = max(i for i, e in enumerate(events) if isinstance(e, SummaryText))
    score_at = next(i for i, e in enumerate(events) if isinstance(e, ScoreClosed))
    keys = list(expected)
    if keys.index("score") < keys.index("summary"):
        assert score_at < first_text
    else:
        assert score_at > last_text


@pytest.mark.parametrize("payload", PAYLOADS)
def test_whole_payload(payload: str) -> None:
    _check(payload, _run([payload]))


@pytest.mark.parametrize("payload", PAYLOADS)
def test_split_at_every_offset(payload: str) -> None:
    for i in range(len(payload) + 1):
        _check(payload, _run([payload[:i], payload[i:]]))


@pytest.mark.parametrize("payload", PAYLOADS)
def test_one_character_per_chunk(payload: str) -> None:
    _check(payload, _run(list(payload)))


@pytest.mark.parametrize("payload", PAYLOADS)
def test_split_at_every_utf8_byte_offset(payload: str) -> None:
    raw = payload.encode()
    for i in range(len(raw) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        chunks = [decoder.decode(raw[:i]), decoder.decode(raw[i:], final=True)]
        _check(payload, _run(chunks))


def test_score_is_emitted_before_its_delimiter_arrives() -> None:
    scanner = JsonFieldScanner()
    assert scanner.feed('{"score": -') == []
    assert scanner.feed("2") == [ScoreClosed(-2)]
    assert scanner.feed(', "entity": "A"') == [EntityClosed("A")]


def test_without_a_range_the_score_waits_for_its_delimiter() -> None:
    scanner = JsonFieldScanner(score_range=None)
    assert scanner.feed('{"score": 1') == []
    assert scanner.feed("2}") == [ScoreClosed(12)]
    scanner.close()


def test_summary_deltas_are_coalesced_per_feed() -> None:
    scanner = JsonFieldScanner()
    assert scanner.feed('{"summary": "ab') == [SummaryText("ab")]
    assert scanner.feed(r"c\u00e") == [SummaryText("c")]
    assert scanner.feed('9 d"') == [SummaryText("é d")]


MALFORMED = [
    '{"score": 1.5, "entity": "A", "summary": "B"}',
    '{"score": "1"}',
    '{"score": 3}',
    '{"score": 01}',
    '{"score": 1e0}',
    '{"summary": "a\\x"}',
    '{"summary": "\\u12G4"}',
    '{"summary": "\\ud83d end"}',
    '{"summary": "\\ud83d\\n"}',
    '{"summary": "\\udcc8"}',
    '{"summary": "\\ud83d\\ud83d"}',
    '{"summary": "a\nb"}',
    '{"score": 1,}',
    '{"score" 1}',
    "{score: 1}",
    '{"a": {"b": 1}}',
    '{"a": [1]}',
    '{"score": 1}}',
    '{"score": 1} x',
    "[1]",
    '{"score": 1, "score": 2}',
    '{"summary": tru}',
    '{"entity": null}',
    '{"summary": 5}',
    '{"x": --1}',
    '{"score": 1',
    '{"summary": "unterminated',
    "",
]


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_input_raises(payload: str) -> None:
    for chunks in ([payload], list(payload)):
        with pytest.raises(SchemaViolation, match="Malformed streamed JSON"):
            _run(chunks)
