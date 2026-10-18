"""Incremental scanner for a flat streamed JSON object; emits score/entity/summary as soon as they exist."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from veritrade.errors import SchemaViolation


@dataclass(frozen=True, slots=True)
class ScoreClosed:
    score: int


@dataclass(frozen=True, slots=True)
class EntityClosed:
    entity: str


@dataclass(frozen=True, slots=True)
class SummaryText:
    delta: str


ScanEvent = ScoreClosed | EntityClosed | SummaryText


class _State(Enum):
    BEGIN = auto()
    KEY_OR_END = auto()
    KEY_REQUIRED = auto()
    KEY = auto()
    COLON = auto()
    VALUE = auto()
    STRING = auto()
    NUMBER = auto()
    LITERAL = auto()
    AFTER_VALUE = auto()
    END = auto()


_WS = frozenset(" \t\n\r")
_PLAIN = re.compile(r'[^"\\\x00-\x1f]+')
_HEX = frozenset("0123456789abcdefABCDEF")
_NUMBER_CHARS = frozenset("0123456789+-.eE")
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_JSON_INT = re.compile(r"-?(?:0|[1-9]\d*)")
_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_KNOWN = frozenset({"score", "entity", "summary"})


class JsonFieldScanner:
    """Feed raw content deltas; `close()` raises unless exactly one complete object was seen. With
    `score_range`, the score is emitted once no further digit could stay in range (a token before
    its delimiter arrives); the delimiter still validates it."""

    def __init__(self, *, score_range: tuple[int, int] | None = (-2, 2)) -> None:
        self._score_range = score_range
        self._state = _State.BEGIN
        self._pos = 0
        self._key: str | None = None
        self._seen: set[str] = set()
        self._buf: list[str] = []
        self._esc: str | None = None
        self._high: int | None = None
        self._num: list[str] = []
        self._score_emitted: int | None = None
        self._pending: list[str] = []

    @property
    def done(self) -> bool:
        return self._state is _State.END

    def _fail(self, why: str) -> SchemaViolation:
        return SchemaViolation(f"Malformed streamed JSON at char {self._pos}: {why}")

    def feed(self, chunk: str) -> list[ScanEvent]:
        events: list[ScanEvent] = []
        i, n = 0, len(chunk)
        while i < n:
            state = self._state
            if state is _State.KEY or state is _State.STRING:
                i = self._scan_string(chunk, i, events)
                continue
            c = chunk[i]
            if state is _State.NUMBER:
                if c in _NUMBER_CHARS:
                    self._num.append(c)
                    self._advance(1)
                    i += 1
                    self._maybe_emit_score_early(events)
                else:
                    self._finish_number(events)
                continue
            if state is _State.LITERAL:
                if c.isalpha():
                    self._num.append(c)
                    self._advance(1)
                    i += 1
                else:
                    self._finish_literal()
                continue
            self._advance(1)
            i += 1
            if c in _WS:
                continue
            self._structural(c)
        self._flush_summary(events)
        return events

    def close(self) -> None:
        if self._state is not _State.END:
            raise self._fail("stream ended before the JSON object closed")

    # --- structure ------------------------------------------------------------------------

    def _advance(self, count: int) -> None:
        self._pos += count

    def _structural(self, c: str) -> None:
        state = self._state
        if state is _State.BEGIN and c == "{":
            self._state = _State.KEY_OR_END
        elif state in (_State.KEY_OR_END, _State.KEY_REQUIRED) and c == '"':
            self._state = _State.KEY
            self._buf = []
        elif state is _State.KEY_OR_END and c == "}":
            self._state = _State.END
        elif state is _State.COLON and c == ":":
            self._state = _State.VALUE
        elif state is _State.VALUE:
            self._start_value(c)
        elif state is _State.AFTER_VALUE and c == ",":
            self._state = _State.KEY_REQUIRED
        elif state is _State.AFTER_VALUE and c == "}":
            self._state = _State.END
        else:
            raise self._fail(f"unexpected {c!r} in state {state.name}")

    def _start_value(self, c: str) -> None:
        if c == '"':
            self._state = _State.STRING
            self._buf = []
        elif c == "-" or c.isdigit():
            self._state = _State.NUMBER
            self._num = [c]
        elif c in "tfn":
            self._state = _State.LITERAL
            self._num = [c]
        elif c in "{[":
            raise self._fail("nested values are not part of any output schema")
        else:
            raise self._fail(f"unexpected {c!r} at the start of a value")

    # --- strings --------------------------------------------------------------------------

    def _scan_string(self, chunk: str, i: int, events: list[ScanEvent]) -> int:
        n = len(chunk)
        while i < n:
            if self._esc is not None:
                self._esc += chunk[i]
                self._advance(1)
                i += 1
                self._step_escape()
                continue
            run = _PLAIN.match(chunk, i)
            if run:
                self._append(run.group())
                self._advance(run.end() - i)
                i = run.end()
                continue
            c = chunk[i]
            self._advance(1)
            i += 1
            if c == '"':
                self._close_string(events)
                return i
            if c == "\\":
                self._esc = "\\"
                continue
            raise self._fail("unescaped control character in a string")
        return i

    def _step_escape(self) -> None:
        esc = self._esc or ""
        if len(esc) == 2:
            kind = esc[1]
            if kind in _SIMPLE_ESCAPES:
                self._esc = None
                self._append(_SIMPLE_ESCAPES[kind])
            elif kind != "u":
                raise self._fail(f"invalid escape \\{kind}")
            return
        if esc[-1] not in _HEX:
            raise self._fail(f"invalid \\u escape {esc!r}")
        if len(esc) < 6:
            return
        self._esc = None
        code = int(esc[2:], 16)
        if 0xD800 <= code <= 0xDBFF:
            if self._high is not None:
                raise self._fail("two high surrogates in a row")
            self._high = code
        elif 0xDC00 <= code <= 0xDFFF:
            if self._high is None:
                raise self._fail("low surrogate without a high surrogate")
            high, self._high = self._high, None
            self._emit_text(chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)))
        else:
            self._append(chr(code))

    def _append(self, text: str) -> None:
        if self._high is not None:
            raise self._fail("high surrogate not followed by a low surrogate")
        self._emit_text(text)

    def _emit_text(self, text: str) -> None:
        self._buf.append(text)
        if self._state is _State.STRING and self._key == "summary":
            self._pending.append(text)

    def _close_string(self, events: list[ScanEvent]) -> None:
        if self._high is not None:
            raise self._fail("string ended inside a surrogate pair")
        text = "".join(self._buf)
        self._buf = []
        if self._state is _State.KEY:
            if text in self._seen:
                raise self._fail(f"duplicate key {text!r}")
            self._seen.add(text)
            self._key = text
            self._state = _State.COLON
            return
        if self._key == "score":
            raise self._fail("score must be an integer, got a string")
        self._flush_summary(events)
        if self._key == "entity":
            events.append(EntityClosed(text))
        self._state = _State.AFTER_VALUE

    def _flush_summary(self, events: list[ScanEvent]) -> None:
        if self._pending:
            events.append(SummaryText("".join(self._pending)))
            self._pending = []

    # --- numbers and literals -------------------------------------------------------------

    def _maybe_emit_score_early(self, events: list[ScanEvent]) -> None:
        if self._key != "score" or self._score_range is None or self._score_emitted is not None:
            return
        text = "".join(self._num)
        if not _JSON_INT.fullmatch(text):
            return
        lo, hi = self._score_range
        value = int(text)
        extendable = text.lstrip("-") != "0" and any(lo <= int(text + d) <= hi for d in "0123456789")
        if lo <= value <= hi and not extendable:
            self._score_emitted = value
            events.append(ScoreClosed(value))

    def _finish_number(self, events: list[ScanEvent]) -> None:
        text = "".join(self._num)
        self._num = []
        self._state = _State.AFTER_VALUE
        if not _JSON_NUMBER.fullmatch(text):
            raise self._fail(f"invalid number {text!r}")
        if self._key in ("entity", "summary"):
            raise self._fail(f"{self._key} must be a string")
        if self._key != "score":
            return
        if not _JSON_INT.fullmatch(text):
            raise self._fail(f"score must be an integer, got {text!r}")
        value = int(text)
        if self._score_range and not self._score_range[0] <= value <= self._score_range[1]:
            raise self._fail(f"score {value} is outside {self._score_range}")
        if self._score_emitted is None:
            events.append(ScoreClosed(value))
        elif self._score_emitted != value:
            raise self._fail("score changed after it was emitted")

    def _finish_literal(self) -> None:
        text = "".join(self._num)
        self._num = []
        self._state = _State.AFTER_VALUE
        if text not in ("true", "false", "null"):
            raise self._fail(f"invalid literal {text!r}")
        if self._key in _KNOWN:
            raise self._fail(f"{self._key} cannot be {text}")
