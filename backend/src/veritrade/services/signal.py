"""/signal: one non-streaming classification of the lede (or the full article)."""

from __future__ import annotations

from veritrade.api.contracts import SignalRequest, SignalResponse
from veritrade.inference.adapters import classify_signal
from veritrade.services.pipeline import Job, inference_timings, signal_result
from veritrade.services.runtime import Runtime
from veritrade.timing import StageTimer


async def run_signal(runtime: Runtime, request: SignalRequest, clock: StageTimer) -> SignalResponse:
    job = Job.resolve(runtime, "signal", request, scope=request.scope, clock=clock)
    if (hit := job.cached(SignalResponse)) is not None:
        return hit
    prepared = await job.prepare()
    verdict, raw = await classify_signal(runtime.ollama, job.spec, job.prompt, prepared.text)
    response = SignalResponse(
        meta=prepared.meta,
        signal=signal_result(verdict),
        timings=prepared.timings(inference_timings(raw)),
    )
    job.remember(response)
    return response
