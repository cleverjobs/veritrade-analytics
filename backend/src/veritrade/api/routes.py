"""HTTP routes: thin adapters from request bodies to services.

Stream routes take their events from a dependency, so every failure before the first byte (bad URL, model,
fetch, extraction) still reaches the client as a JSON error with its own HTTP status.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from veritrade.api.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    SignalRequest,
    SignalResponse,
    SummaryRequest,
    SummaryResponse,
)
from veritrade.api.sse import event_stream
from veritrade.services.analyze import open_analyze_stream, run_analyze
from veritrade.services.pipeline import StreamPayload
from veritrade.services.runtime import Runtime
from veritrade.services.signal import run_signal
from veritrade.services.summary import open_summary_stream, run_summary
from veritrade.services.system import check_health, list_models
from veritrade.timing import StageTimer

_ERROR_STATUSES = (400, 422, 502, 503, 504)
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in _ERROR_STATUSES
}
# A `model` entry would inherit the route's text/event-stream media type; stream errors are sent as JSON.
STREAM_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {
        "description": "Error raised before the stream opens",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
    }
    for status in _ERROR_STATUSES
}

router = APIRouter(prefix="/api")


async def request_clock() -> StageTimer:
    return StageTimer()


async def get_runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


# Declared first in every route so the clock starts before any other dependency or validation runs.
Clock = Annotated[StageTimer, Depends(request_clock)]
RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
Events = AsyncGenerator[StreamPayload]


async def summary_events(clock: Clock, runtime: RuntimeDep, body: SummaryRequest) -> Events:
    return await open_summary_stream(runtime, body, clock)


async def analyze_events(clock: Clock, runtime: RuntimeDep, body: AnalyzeRequest) -> Events:
    return await open_analyze_stream(runtime, body, clock)


@router.post("/signal", response_model=SignalResponse, responses=ERROR_RESPONSES)
async def signal(clock: Clock, runtime: RuntimeDep, body: SignalRequest) -> SignalResponse:
    return await run_signal(runtime, body, clock)


@router.post("/summary", response_model=SummaryResponse, responses=ERROR_RESPONSES)
async def summary(clock: Clock, runtime: RuntimeDep, body: SummaryRequest) -> SummaryResponse:
    return await run_summary(runtime, body, clock)


@router.post("/summary/stream", response_class=EventSourceResponse, responses=STREAM_ERROR_RESPONSES)
async def summary_stream(
    events: Annotated[Events, Depends(summary_events)],
) -> AsyncIterator[ServerSentEvent]:
    async with aclosing(event_stream(events)) as stream:
        async for event in stream:
            yield event


@router.post("/analyze", response_model=AnalyzeResponse, responses=ERROR_RESPONSES)
async def analyze(clock: Clock, runtime: RuntimeDep, body: AnalyzeRequest) -> AnalyzeResponse:
    return await run_analyze(runtime, body, clock)


@router.post("/analyze/stream", response_class=EventSourceResponse, responses=STREAM_ERROR_RESPONSES)
async def analyze_stream(
    events: Annotated[Events, Depends(analyze_events)],
) -> AsyncIterator[ServerSentEvent]:
    async with aclosing(event_stream(events)) as stream:
        async for event in stream:
            yield event


@router.get("/models", response_model=ModelsResponse)
async def models(runtime: RuntimeDep) -> ModelsResponse:
    return await list_models(runtime)


@router.get("/health", response_model=HealthResponse)
async def health(runtime: RuntimeDep) -> HealthResponse:
    return await check_health(runtime)
