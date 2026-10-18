"""/analyze: sentiment, entity and summary from one generation, as JSON or as streamed events."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from veritrade.api.contracts import (
    AnalyzeDoneEvent,
    AnalyzeRequest,
    AnalyzeResponse,
    SentimentEvent,
    SummaryDeltaEvent,
)
from veritrade.inference.adapters import AnalyzeResult, analyze, analyze_stream, score_verdict
from veritrade.inference.incremental import EntityClosed, ScoreClosed, SummaryText
from veritrade.services.pipeline import (
    Job,
    Prepared,
    StreamPayload,
    cached_meta_event,
    inference_timings,
    signal_result,
)
from veritrade.services.runtime import Runtime
from veritrade.timing import StageTimer


def _job(runtime: Runtime, request: AnalyzeRequest, clock: StageTimer) -> Job:
    return Job.resolve(runtime, "analyze", request, scope=request.scope, clock=clock)


async def run_analyze(runtime: Runtime, request: AnalyzeRequest, clock: StageTimer) -> AnalyzeResponse:
    job = _job(runtime, request, clock)
    if (hit := job.cached(AnalyzeResponse)) is not None:
        return hit
    prepared = await job.prepare()
    result = await analyze(runtime.ollama, job.spec, job.prompt, prepared.text)
    response = _response(prepared, result, ttft_ms=None)
    job.remember(response)
    return response


async def open_analyze_stream(
    runtime: Runtime, request: AnalyzeRequest, clock: StageTimer
) -> AsyncGenerator[StreamPayload]:
    """Runs every step that can fail before the stream opens; the returned events only do inference."""
    job = _job(runtime, request, clock)
    if (hit := job.cached(AnalyzeResponse)) is not None:
        return _replay(hit, clock)
    return _stream(await job.prepare())


async def _stream(prepared: Prepared) -> AsyncGenerator[StreamPayload]:
    """Events follow generation order, so the frozen field order decides where `sentiment` lands."""
    job = prepared.job
    yield prepared.meta_event()
    entity: str | None = None
    ttft_ms: float | None = None
    events = analyze_stream(job.runtime.ollama, job.spec, job.prompt, prepared.text)
    async with aclosing(events):
        async for event in events:
            match event:
                case ScoreClosed(score=score):
                    yield SentimentEvent(
                        signal=signal_result(score_verdict(job.spec, score)),
                        entity=entity,
                        elapsed_ms=job.clock.elapsed_ms(),
                    )
                case EntityClosed(entity=name):
                    entity = name.strip() or None
                case SummaryText(delta=delta):
                    if ttft_ms is None:
                        ttft_ms = job.clock.elapsed_ms()
                    yield SummaryDeltaEvent(text=delta)
                case AnalyzeResult():
                    response = _response(prepared, event, ttft_ms=ttft_ms)
                    job.remember(response)
                    yield AnalyzeDoneEvent(result=response)


async def _replay(hit: AnalyzeResponse, clock: StageTimer) -> AsyncGenerator[StreamPayload]:
    yield cached_meta_event(hit)
    yield SentimentEvent(signal=hit.signal, entity=hit.entity, elapsed_ms=clock.elapsed_ms())
    yield SummaryDeltaEvent(text=hit.summary)
    yield AnalyzeDoneEvent(result=hit)


def _response(prepared: Prepared, result: AnalyzeResult, *, ttft_ms: float | None) -> AnalyzeResponse:
    return AnalyzeResponse(
        meta=prepared.meta,
        signal=signal_result(result.verdict),
        entity=result.entity,
        summary=result.output.summary,
        truncated=result.truncated,
        timings=prepared.timings(inference_timings(result.chat, ttft_ms=ttft_ms)),
    )
