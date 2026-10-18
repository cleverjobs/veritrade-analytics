"""Server-sent event framing: service payloads become named events, late failures one `error` event."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing

from fastapi.sse import ServerSentEvent
from pydantic import BaseModel

from veritrade.api.contracts import (
    AnalyzeDoneEvent,
    ErrorBody,
    ErrorEvent,
    MetaEvent,
    SentimentEvent,
    SummaryDeltaEvent,
    SummaryDoneEvent,
)
from veritrade.errors import VtError
from veritrade.services.pipeline import StreamPayload

EVENT_NAMES: Mapping[type[BaseModel], str] = {
    MetaEvent: "meta",
    SentimentEvent: "sentiment",
    SummaryDeltaEvent: "summary_delta",
    SummaryDoneEvent: "done",
    AnalyzeDoneEvent: "done",
    ErrorEvent: "error",
}


def error_body(exc: VtError) -> ErrorBody:
    return ErrorBody(
        code=exc.code,
        message=exc.message,
        hint=exc.hint,
        retryable=exc.retryable,
        upstream_status=exc.upstream_status,
    )


async def event_stream(events: AsyncGenerator[StreamPayload]) -> AsyncGenerator[ServerSentEvent]:
    """Closing the payload generator on any exit is what stops the upstream Ollama stream on disconnect."""
    async with aclosing(events):
        try:
            async for payload in events:
                yield ServerSentEvent(event=EVENT_NAMES[type(payload)], data=payload)
        except VtError as exc:
            yield ServerSentEvent(event="error", data=ErrorEvent(error=error_body(exc)))
