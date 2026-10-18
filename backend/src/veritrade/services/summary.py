"""/summary: a short summary of the full article, as one JSON body or as streamed deltas."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from veritrade.api.contracts import SummaryDeltaEvent, SummaryDoneEvent, SummaryRequest, SummaryResponse
from veritrade.inference.adapters import SummaryResult, summarize, summarize_stream
from veritrade.inference.incremental import SummaryText
from veritrade.services.pipeline import Job, Prepared, StreamPayload, cached_meta_event, inference_timings
from veritrade.services.runtime import Runtime
from veritrade.timing import StageTimer


def _job(runtime: Runtime, request: SummaryRequest, clock: StageTimer) -> Job:
    return Job.resolve(runtime, "summary", request, scope="full", clock=clock)


async def run_summary(runtime: Runtime, request: SummaryRequest, clock: StageTimer) -> SummaryResponse:
    job = _job(runtime, request, clock)
    if (hit := job.cached(SummaryResponse)) is not None:
        return hit
    prepared = await job.prepare()
    result = await summarize(runtime.ollama, job.spec, job.prompt, prepared.text)
    response = _response(prepared, result, ttft_ms=None)
    job.remember(response)
    return response


async def open_summary_stream(
    runtime: Runtime, request: SummaryRequest, clock: StageTimer
) -> AsyncGenerator[StreamPayload]:
    """Runs every step that can fail before the stream opens; the returned events only do inference."""
    job = _job(runtime, request, clock)
    if (hit := job.cached(SummaryResponse)) is not None:
        return _replay(hit)
    return _stream(await job.prepare())


async def _stream(prepared: Prepared) -> AsyncGenerator[StreamPayload]:
    job = prepared.job
    yield prepared.meta_event()
    ttft_ms: float | None = None
    events = summarize_stream(job.runtime.ollama, job.spec, job.prompt, prepared.text)
    async with aclosing(events):
        async for event in events:
            match event:
                case SummaryText(delta=delta):
                    if ttft_ms is None:
                        ttft_ms = job.clock.elapsed_ms()
                    yield SummaryDeltaEvent(text=delta)
                case SummaryResult():
                    response = _response(prepared, event, ttft_ms=ttft_ms)
                    job.remember(response)
                    yield SummaryDoneEvent(result=response)


async def _replay(hit: SummaryResponse) -> AsyncGenerator[StreamPayload]:
    yield cached_meta_event(hit)
    yield SummaryDeltaEvent(text=hit.summary)
    yield SummaryDoneEvent(result=hit)


def _response(prepared: Prepared, result: SummaryResult, *, ttft_ms: float | None) -> SummaryResponse:
    return SummaryResponse(
        meta=prepared.meta,
        summary=result.output.summary,
        timings=prepared.timings(inference_timings(result.chat, ttft_ms=ttft_ms)),
    )
