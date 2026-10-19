"""Network-free stand-ins: publisher pages from the skeleton fixtures and a scripted Ollama on respx."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import respx

PUBLIC_IP = "93.184.215.14"
OLLAMA_URL = "http://ollama.test"
SKELETONS = Path(__file__).resolve().parents[1] / "fixtures" / "skeletons"
# Synthetic model output; the escaped quote and the non-ASCII sign exercise the stream scanner.
SUMMARY = 'Acme Corp raised its outlook after "record" demand lifted revenue 12% to €3.1 billion.'
ENTITY = "Acme Corp"


@dataclass(frozen=True)
class Page:
    source: str
    url: str
    html: str
    body_markers: tuple[str, ...]


@cache
def skeleton_pages() -> dict[str, Page]:
    pages = {}
    for path in sorted(SKELETONS.glob("*/*.expected.json")):
        expected = json.loads(path.read_text(encoding="utf-8"))
        html = path.with_name(f"{expected['slug']}.html").read_text(encoding="utf-8")
        source = expected["source"]
        pages[source] = Page(source, expected["fixture_url"], html, tuple(expected["body_markers"]))
    return pages


class FakeResolver:
    """Hosts resolve to one public address unless the table says otherwise; no DNS is used."""

    def __init__(self, table: Mapping[str, list[str]] | None = None) -> None:
        self.table = dict(table or {})

    async def __call__(self, host: str, port: int) -> list[str]:
        return self.table.get(host, [PUBLIC_IP])


class NdjsonStream(httpx.AsyncByteStream):
    """Ollama's NDJSON chat stream, one line per delay; records how far it got and whether it was closed."""

    def __init__(self, lines: list[bytes], delay: float) -> None:
        self.lines = lines
        self.delay = delay
        self.sent = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in self.lines:
            await asyncio.sleep(self.delay)
            self.sent += 1
            yield line

    async def aclose(self) -> None:
        self.closed = True


def _first_token(field_name: str, value: Any) -> str:
    if field_name == "label":
        return {"bullish": "bull", "bearish": "bear"}.get(value, value)
    return "-" if value < 0 else str(value)


def _logprob(token: str, p: float, alternatives: Mapping[str, float] | None = None) -> dict[str, Any]:
    top = [{"token": t, "logprob": math.log(q)} for t, q in (alternatives or {token: p}).items()]
    return {"token": token, "logprob": math.log(p), "top_logprobs": top}


def _done(**extra: Any) -> dict[str, Any]:
    return {
        "model": "m",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 400_000_000,
        "load_duration": 5_000_000,
        "prompt_eval_count": 512,
        "prompt_eval_duration": 120_000_000,
        "eval_count": 40,
        "eval_duration": 260_000_000,
    } | extra


@dataclass
class FakeOllama:
    """Answers follow each request's `format` schema in its property order, as Ollama's grammar does."""

    router: respx.MockRouter
    label: str = "bullish"
    label_alternatives: dict[str, float] = field(
        default_factory=lambda: {"bull": 0.9, "neutral": 0.08, "bear": 0.02}
    )
    score: int = 1
    entity: str = ENTITY
    summary: str = SUMMARY
    decision: dict[str, float] = field(
        default_factory=lambda: {"neutral": 0.06, "bullish": 0.9, "bearish": 0.04}
    )
    installed: tuple[str, ...] = ()
    # Loaded models as /api/ps lists them: name -> context length.
    resident: dict[str, int] = field(default_factory=dict)
    # Raised or returned by every /api/ps call when set.
    ps_failure: Exception | httpx.Response | None = None
    version: str = "0.40.6"
    delay: float = 0.001
    chunk_chars: int = 4
    # Raised or returned by every /api/chat call when set.
    chat_failure: Exception | httpx.Response | None = None
    # Emit an Ollama error line after this many content chunks.
    stream_error_after: int | None = None
    # End any output that contains this text right after it with done_reason "length", as num_predict does.
    cut_after: str | None = None
    down: bool = False
    streams: list[NdjsonStream] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = self.router.post(f"{OLLAMA_URL}/api/chat").mock(side_effect=self._chat)
        self.systemone = self.router.post(f"{OLLAMA_URL}/v1/systemone").mock(side_effect=self._systemone)
        self.tags = self.router.get(f"{OLLAMA_URL}/api/tags").mock(side_effect=self._tags)
        self.ps = self.router.get(f"{OLLAMA_URL}/api/ps").mock(side_effect=self._ps)
        self.version_route = self.router.get(f"{OLLAMA_URL}/api/version").mock(side_effect=self._version)

    @property
    def inference_calls(self) -> int:
        return self.chat.call_count + self.systemone.call_count

    def _check_up(self, request: httpx.Request) -> None:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)

    def _value(self, name: str) -> Any:
        return {"label": self.label, "score": self.score, "entity": self.entity, "summary": self.summary}[
            name
        ]

    def _chat(self, request: httpx.Request) -> httpx.Response:
        self._check_up(request)
        if isinstance(self.chat_failure, Exception):
            raise self.chat_failure
        if self.chat_failure is not None:
            return self.chat_failure
        body = json.loads(request.content)
        fields = list(body["format"]["properties"])
        content = json.dumps({name: self._value(name) for name in fields}, ensure_ascii=False)
        done_reason = "stop"
        if self.cut_after and self.cut_after in content:
            content = content[: content.index(self.cut_after) + len(self.cut_after)]
            done_reason = "length"
        if body["stream"]:
            return self._stream(content, done_reason)
        reply = _done(message={"role": "assistant", "content": content}, done_reason=done_reason)
        if body.get("logprobs") and fields in (["label"], ["score"]):
            reply["logprobs"] = self._logprobs(content, fields[0])
        return httpx.Response(200, json=reply)

    def _logprobs(self, content: str, name: str) -> list[dict[str, Any]]:
        """Three tokens that tile the content: everything up to the value, its first token, the rest."""
        first = _first_token(name, self._value(name))
        # json.dumps writes ": " after the key, then the label's opening quote or the bare score.
        start = content.index(":") + 2 + (name == "label")
        alternatives = self.label_alternatives if name == "label" else {first: 0.9, "0": 0.08, "-": 0.02}
        return [
            _logprob(content[:start], 0.99),
            _logprob(first, alternatives.get(first, 0.9), alternatives),
            _logprob(content[start + len(first) :], 0.99),
        ]

    def _stream(self, content: str, done_reason: str) -> httpx.Response:
        pieces = [content[i : i + self.chunk_chars] for i in range(0, len(content), self.chunk_chars)]
        chunks: list[dict[str, Any]] = [
            {"model": "m", "message": {"role": "assistant", "content": piece}, "done": False}
            for piece in pieces
        ]
        if self.stream_error_after is not None:
            chunks = [*chunks[: self.stream_error_after], {"error": "llama runner process has terminated"}]
        else:
            chunks.append(_done(eval_count=len(pieces), done_reason=done_reason))
        stream = NdjsonStream([(json.dumps(chunk) + "\n").encode() for chunk in chunks], self.delay)
        self.streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "application/x-ndjson"})

    def _systemone(self, request: httpx.Request) -> httpx.Response:
        self._check_up(request)
        body = json.loads(request.content)
        choice = max(self.decision, key=self.decision.__getitem__)
        answer = {
            "type": "choice",
            "choice": choice,
            "probabilities": self.decision,
            "confidence": self.decision[choice],
        }
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": {"sentiment": answer},
                "usage": {"input_tokens": 264, "output_tokens": 0},
            },
        )

    def _tags(self, request: httpx.Request) -> httpx.Response:
        self._check_up(request)
        models = [
            {"name": name, "model": name, "size": 1_000_000 * (i + 1)}
            for i, name in enumerate(self.installed)
        ]
        return httpx.Response(200, json={"models": models})

    def _ps(self, request: httpx.Request) -> httpx.Response:
        self._check_up(request)
        if isinstance(self.ps_failure, Exception):
            raise self.ps_failure
        if self.ps_failure is not None:
            return self.ps_failure
        models = [
            {"name": name, "model": name, "context_length": context}
            for name, context in self.resident.items()
        ]
        return httpx.Response(200, json={"models": models})

    def _version(self, request: httpx.Request) -> httpx.Response:
        self._check_up(request)
        return httpx.Response(200, json={"version": self.version})


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """(event, data) pairs from a text/event-stream body; comments and unnamed frames are skipped."""
    events = []
    for frame in text.split("\n\n"):
        name, data = None, []
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data.append(line.removeprefix("data: "))
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events
