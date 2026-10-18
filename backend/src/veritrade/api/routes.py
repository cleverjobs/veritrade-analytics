"""HTTP routes. Bodies are wired to services in the API milestone; P0 ships the contract only."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
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

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in (400, 422, 502, 503, 504)
}

router = APIRouter(prefix="/api")


@router.post("/signal", response_model=SignalResponse, responses=ERROR_RESPONSES)
async def signal(body: SignalRequest) -> SignalResponse:
    raise NotImplementedError


@router.post("/summary", response_model=SummaryResponse, responses=ERROR_RESPONSES)
async def summary(body: SummaryRequest) -> SummaryResponse:
    raise NotImplementedError


@router.post("/summary/stream", response_class=EventSourceResponse, responses=ERROR_RESPONSES)
async def summary_stream(body: SummaryRequest) -> AsyncIterator[ServerSentEvent]:
    raise NotImplementedError
    yield ServerSentEvent(data=None)


@router.post("/analyze", response_model=AnalyzeResponse, responses=ERROR_RESPONSES)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    raise NotImplementedError


@router.post("/analyze/stream", response_class=EventSourceResponse, responses=ERROR_RESPONSES)
async def analyze_stream(body: AnalyzeRequest) -> AsyncIterator[ServerSentEvent]:
    raise NotImplementedError
    yield ServerSentEvent(data=None)


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    raise NotImplementedError


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    raise NotImplementedError
