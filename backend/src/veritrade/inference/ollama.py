"""Async Ollama client on httpx: the only Ollama path for the API, evals and probes."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from veritrade.config import Settings, get_settings
from veritrade.errors import (
    InferenceTimeout,
    LlmUnreachable,
    ModelUnavailable,
    SchemaViolation,
    VtError,
)
from veritrade.inference.schemas import ollama_format
from veritrade.inference.types import (
    SENTIMENTS,
    ChatResult,
    DecisionResult,
    OllamaTimings,
    Sentiment,
    TokenLogprob,
)

Message = Mapping[str, str]
# Greedy and seeded so evals and serving see the same output for the same prompt. Penalties are neutralised
# because Modelfiles set them (qwen3.5: presence_penalty 1.5), which skews outputs away from recent tokens.
FIXED_OPTIONS: dict[str, Any] = {
    "temperature": 0,
    "seed": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repeat_penalty": 1,
}
SENTIMENT_QUESTION = "sentiment"


class OutputTruncated(SchemaViolation):
    """Generation hit num_predict before the JSON closed; `chat` keeps the partial output."""

    def __init__(self, model: str, chat: ChatResult) -> None:
        super().__init__(f"{model} reached its output token limit before the JSON closed")
        self.chat = chat


class _ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choice: Sentiment
    probabilities: dict[Sentiment, float]
    confidence: float


def _ms(value: Any) -> float | None:
    return value / 1e6 if isinstance(value, int | float) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def parse_timings(body: Mapping[str, Any]) -> OllamaTimings:
    return OllamaTimings(
        load_ms=_ms(body.get("load_duration")),
        prompt_eval_ms=_ms(body.get("prompt_eval_duration")),
        eval_ms=_ms(body.get("eval_duration")),
        prompt_tokens=_int(body.get("prompt_eval_count")),
        eval_tokens=_int(body.get("eval_count")),
        total_ms=_ms(body.get("total_duration")),
    )


def parse_logprobs(raw: Any) -> tuple[TokenLogprob, ...] | None:
    if not isinstance(raw, list) or not raw:
        return None
    return tuple(
        TokenLogprob(
            token=entry["token"],
            logprob=float(entry["logprob"]),
            top=tuple((alt["token"], float(alt["logprob"])) for alt in entry.get("top_logprobs") or ()),
        )
        for entry in raw
    )


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, str) else response.text[:300]


def _status_error(status: int, text: str, model: str | None) -> VtError:
    name = model or "?"
    if status == 404 or "not found" in text.lower():
        return ModelUnavailable(
            f"Model {name} is not available in Ollama: {text}",
            hint=f"Pull it with `ollama pull {name}` or pick an installed model.",
            upstream_status=status,
        )
    return ModelUnavailable(
        f"Ollama rejected the request for {name} ({status}): {text}", upstream_status=status
    )


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 90.0,
        keep_alive: str | int = "30m",
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._http = http or httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=min(5.0, timeout))
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OllamaClient:
        settings = settings or get_settings()
        return cls(
            settings.ollama_base_url,
            timeout=settings.inference_timeout_s,
            keep_alive=settings.ollama_keep_alive,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    # --- transport ------------------------------------------------------------------------

    def _transport_error(self, exc: httpx.HTTPError, model: str | None) -> VtError:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return LlmUnreachable(
                f"Cannot reach Ollama at {self.base_url}: {exc}",
                hint="Start Ollama (`brew services start ollama`) or point OLLAMA_BASE_URL at it.",
            )
        if isinstance(exc, httpx.TimeoutException):
            return InferenceTimeout(f"Ollama did not answer in time for {model or 'request'}: {exc!r}")
        return LlmUnreachable(f"Ollama connection failed: {exc!r}")

    async def request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None, *, model: str | None = None
    ) -> dict[str, Any]:
        """One JSON round trip with Ollama errors mapped to typed VtErrors."""
        try:
            response = await self._http.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, model) from exc
        if response.status_code >= 400:
            raise _status_error(response.status_code, _error_text(response), model)
        try:
            body = response.json()
        except ValueError as exc:
            raise SchemaViolation(f"Ollama returned non-JSON for {path}") from exc
        if not isinstance(body, dict):
            raise SchemaViolation(f"Ollama returned an unexpected body for {path}")
        return body

    # --- chat -----------------------------------------------------------------------------

    def _chat_body(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        stream: bool,
        format: Mapping[str, Any] | str | None,
        options: Mapping[str, Any] | None,
        logprobs: bool,
        top_logprobs: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "stream": stream,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {**(options or {}), **FIXED_OPTIONS},
        }
        if format is not None:
            body["format"] = format
        if logprobs:
            body["logprobs"] = True
            if top_logprobs is not None:
                body["top_logprobs"] = top_logprobs
        return body

    async def chat_raw(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        format: Mapping[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> tuple[ChatResult, dict[str, Any]]:
        body = self._chat_body(
            model,
            messages,
            stream=False,
            format=format,
            options=options,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        t0 = time.perf_counter()
        raw = await self.request("POST", "/api/chat", body, model=model)
        wall_ms = (time.perf_counter() - t0) * 1000
        message = raw.get("message") or {}
        result = ChatResult(
            content=str(message.get("content") or ""),
            wall_ms=wall_ms,
            timings=parse_timings(raw),
            logprobs=parse_logprobs(raw.get("logprobs")),
            done_reason=_str(raw.get("done_reason")),
        )
        return result, raw

    async def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        format: Mapping[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> ChatResult:
        result, _ = await self.chat_raw(
            model, messages, format=format, options=options, logprobs=logprobs, top_logprobs=top_logprobs
        )
        return result

    async def chat_json[T: BaseModel](
        self,
        model: str,
        messages: Sequence[Message],
        out_type: type[T],
        *,
        options: Mapping[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        retry_cut: bool = True,
    ) -> tuple[T, ChatResult]:
        """Schema-constrained chat, validated, with one retry. Output cut at num_predict is retried with
        a doubled budget, or raised as OutputTruncated when `retry_cut` is off so the caller can salvage."""
        opts = dict(options or {})
        spent_ms = 0.0
        last_error: ValidationError | None = None
        for _attempt in range(2):
            result, _ = await self.chat_raw(
                model,
                messages,
                format=ollama_format(out_type),
                options=opts,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )
            spent_ms += result.wall_ms
            try:
                parsed = out_type.model_validate_json(result.content)
            except ValidationError as exc:
                last_error = exc
                if result.done_reason == "length":
                    if not retry_cut:
                        raise OutputTruncated(model, dataclasses.replace(result, wall_ms=spent_ms)) from exc
                    if isinstance(opts.get("num_predict"), int):
                        opts["num_predict"] = opts["num_predict"] * 2
                continue
            return parsed, dataclasses.replace(result, wall_ms=spent_ms)
        raise SchemaViolation(
            f"{model} returned output that does not match {out_type.__name__} twice",
            hint=str(last_error)[:300] if last_error else None,
        )

    async def chat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        format: Mapping[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> AsyncGenerator[str | ChatResult]:
        """Yield content deltas, then one final ChatResult; ttft is measured here, not by Ollama."""
        body = self._chat_body(
            model,
            messages,
            stream=True,
            format=format,
            options=options,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        parts: list[str] = []
        token_logprobs: list[TokenLogprob] = []
        try:
            async with self._http.stream("POST", "/api/chat", json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _status_error(response.status_code, _error_text(response), model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as exc:
                        raise SchemaViolation(f"Ollama sent a malformed stream line for {model}") from exc
                    if isinstance(chunk.get("error"), str):
                        raise _status_error(500, chunk["error"], model)
                    if parsed := parse_logprobs(chunk.get("logprobs")):
                        token_logprobs.extend(parsed)
                    delta = str((chunk.get("message") or {}).get("content") or "")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        parts.append(delta)
                        yield delta
                    if chunk.get("done"):
                        yield ChatResult(
                            content="".join(parts),
                            wall_ms=(time.perf_counter() - t0) * 1000,
                            timings=parse_timings(chunk),
                            logprobs=tuple(token_logprobs) or None,
                            ttft_ms=ttft_ms,
                            done_reason=_str(chunk.get("done_reason")),
                        )
                        return
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, model) from exc
        raise SchemaViolation(f"Ollama stream for {model} ended without a done message")

    # --- decision models ------------------------------------------------------------------

    async def systemone(
        self, model: str, state: str, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        body = {
            "model": model,
            "state": state,
            "questions": {k: dict(v) for k, v in questions.items()},
            "keep_alive": self.keep_alive,
        }
        return await self.request("POST", "/v1/systemone", body, model=model)

    async def decide_sentiment(
        self,
        model: str,
        state: str,
        instructions: str,
        criteria: Mapping[Sentiment, str | None],
    ) -> DecisionResult:
        """One `choice` question; criteria order is the tie order, so callers put neutral first."""
        if set(criteria) != set(SENTIMENTS):
            raise ValueError(f"criteria must name exactly {SENTIMENTS}, got {list(criteria)}")
        question = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
        t0 = time.perf_counter()
        raw = await self.systemone(model, state, {SENTIMENT_QUESTION: question})
        wall_ms = (time.perf_counter() - t0) * 1000
        try:
            answer = _ChoiceAnswer.model_validate((raw.get("answers") or {}).get(SENTIMENT_QUESTION))
        except ValidationError as exc:
            raise SchemaViolation(f"{model} returned an unexpected systemone answer") from exc
        if set(answer.probabilities) != set(SENTIMENTS):
            raise SchemaViolation(f"{model} returned probabilities for {sorted(answer.probabilities)}")
        usage = raw.get("usage") or {}
        return DecisionResult(
            choice=answer.choice,
            probabilities={s: answer.probabilities[s] for s in criteria},
            confidence=answer.confidence,
            wall_ms=wall_ms,
            input_tokens=_int(usage.get("input_tokens")),
        )

    # --- utilities ------------------------------------------------------------------------

    async def embed(self, model: str, inputs: Sequence[str]) -> list[list[float]]:
        body = {"model": model, "input": list(inputs), "keep_alive": self.keep_alive}
        raw = await self.request("POST", "/api/embed", body, model=model)
        embeddings = raw.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise SchemaViolation(f"{model} returned {len(embeddings or [])} embeddings for {len(inputs)}")
        return [[float(x) for x in row] for row in embeddings]

    async def tags(self) -> list[dict[str, Any]]:
        return list((await self.request("GET", "/api/tags")).get("models") or [])

    async def ps(self) -> list[dict[str, Any]]:
        return list((await self.request("GET", "/api/ps")).get("models") or [])

    async def show(self, model: str) -> dict[str, Any]:
        return await self.request("POST", "/api/show", {"model": model}, model=model)

    async def version(self) -> str:
        return str((await self.request("GET", "/api/version")).get("version") or "")

    async def unload(self, model: str) -> None:
        await self.request("POST", "/api/generate", {"model": model, "keep_alive": 0}, model=model)
