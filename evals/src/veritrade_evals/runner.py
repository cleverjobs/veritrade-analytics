"""Eval runner: drives the serving adapters over examples and caches raw outputs for offline thresholds."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO, get_args

from pydantic import BaseModel, ValidationError

from veritrade.errors import LlmUnreachable, ModelUnavailable, SchemaViolation, VtError
from veritrade.inference.adapters import TOP_LOGPROBS, analyze_stream, chat_options, classify_signal
from veritrade.inference.confidence import score_sign_probabilities
from veritrade.inference.incremental import ScoreClosed
from veritrade.inference.ollama import FIXED_OPTIONS, OllamaClient
from veritrade.inference.schemas import ANALYZE_MODELS, LabelOut, OrdinalOut, ollama_format
from veritrade.inference.types import ChatResult, DecisionResult, Endpoint, Verdict
from veritrade.models_registry import ModelSpec
from veritrade.prompts.registry import PromptSpec
from veritrade_evals.datasets import Example
from veritrade_evals.paths import PREDS_DIR
from veritrade_evals.predcache import PredCache, PredError, canonical_json

# Part of every cache key: bump when the runner changes what it records for an unchanged config.
RUNNER_VERSION = "1"
DEFAULT_CONCURRENCY = 4
# Rendering the prompt with its own placeholder hashes everything the model sees except the input.
TEXT_SENTINEL = "{text}"
# Synthetic input that loads the model and warms the prompt cache before any timed call.
WARMUP_TEXT = "Ostrava Logistics appoints a new chief financial officer effective January 1."

Variant = Literal["decision", "label", "ordinal", "sentiment-first"]
VARIANTS: tuple[Variant, ...] = get_args(Variant)
VARIANT_ENDPOINT: dict[Variant, Endpoint] = {
    "decision": "signal",
    "label": "signal",
    "ordinal": "signal",
    "sentiment-first": "analyze",
}
Pred = dict[str, Any]
Timing = dict[str, float]


class RunAborted(RuntimeError):
    pass


def serving_spec(base: ModelSpec, variant: Variant) -> ModelSpec:
    """The allowlisted spec switched to the variant under test; thresholds are applied offline."""
    if (variant == "decision") != (base.kind == "decision"):
        raise ValueError(f"{base.name} is a {base.kind} model and cannot run the {variant} variant")
    update: dict[str, Any] = {}
    if variant == "label":
        update = {"signal_schema": "label", "policy": "label-logprob", "min_p": base.min_p or 0.5}
    elif variant == "ordinal":
        update = {"signal_schema": "ordinal", "policy": "ordinal"}
    elif variant == "sentiment-first":
        update = {"field_order": "sentiment-first"}
    return ModelSpec.model_validate({**base.model_dump(), **update})


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One grid cell: a model asked for one output variant under one prompt version."""

    name: str
    spec: ModelSpec
    prompt: PromptSpec
    variant: Variant
    dims: Mapping[str, str] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.spec.name

    @property
    def endpoint(self) -> Endpoint:
        return VARIANT_ENDPOINT[self.variant]

    def run_config(self, split: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "runner": RUNNER_VERSION,
            "split": split,
            "endpoint": self.endpoint,
            "variant": self.variant,
            "prompt_version": self.prompt.version,
            "max_input_tokens": self.spec.max_input_tokens,
        }
        if self.variant != "decision":
            # Exactly what the client sends, so a change to the fixed decoding options re-keys the cache.
            config["options"] = {**chat_options(self.spec, self.endpoint), **FIXED_OPTIONS}
        if self.variant in ("label", "ordinal"):
            config["top_logprobs"] = TOP_LOGPROBS
        if self.variant == "sentiment-first":
            config["stop_after_score"] = True
        return config

    def prompt_sha(self) -> str:
        payload: dict[str, Any]
        if self.variant == "decision":
            payload = {
                "state": self.prompt.state(TEXT_SENTINEL),
                "instructions": self.prompt.instructions,
                "criteria": self.prompt.criteria,
            }
        else:
            payload = {
                "messages": self.prompt.messages(TEXT_SENTINEL, self.variant),
                "format": ollama_format(_out_type(self.variant)),
            }
            if self.variant == "label":
                payload["fallback"] = {
                    "messages": self.prompt.messages(TEXT_SENTINEL, "ordinal"),
                    "format": ollama_format(OrdinalOut),
                }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def cache(self, split: str, model_digest: str, root: Path = PREDS_DIR) -> PredCache:
        return PredCache(
            self.model,
            self.run_config(split),
            model_digest=model_digest,
            prompt_sha=self.prompt_sha(),
            root=root,
        )


def _out_type(variant: Variant) -> type[BaseModel]:
    if variant == "label":
        return LabelOut
    if variant == "ordinal":
        return OrdinalOut
    return ANALYZE_MODELS["sentiment-first"]


# --- one call --------------------------------------------------------------------------------


def _chat_timing(chat: ChatResult) -> Timing:
    timing: Timing = {"wall_ms": chat.wall_ms}
    timing.update({k: float(v) for k, v in asdict(chat.timings).items() if v is not None})
    return timing


def _parsed_label(content: str) -> str | None:
    try:
        return LabelOut.model_validate_json(content).label
    except ValidationError:
        return None


def _label_pred(verdict: Verdict, chat: ChatResult) -> Pred:
    """Records which branch of the serving fallback chain produced the probabilities, if any did."""
    if verdict.policy == "ordinal":
        fallback = "ordinal"
    elif verdict.score is not None:
        fallback = "score-sign"
    else:
        fallback = "none"
    return {
        "kind": "label",
        "label": _parsed_label(chat.content) if fallback == "none" else None,
        "probs": dict(verdict.probabilities) if verdict.probabilities is not None else None,
        "fallback": fallback,
        "score": verdict.score,
    }


async def _score_first(client: OllamaClient, run: RunSpec, text: str) -> tuple[Pred, Timing]:
    """Stops at the score: later tokens cannot change it, and closing the stream frees Ollama early."""
    t0 = time.perf_counter()
    stream = analyze_stream(client, run.spec, run.prompt, text)
    async with aclosing(stream):
        async for item in stream:
            if isinstance(item, ScoreClosed):
                return {"kind": "analyze", "score": item.score}, {
                    "wall_ms": (time.perf_counter() - t0) * 1000
                }
    raise SchemaViolation(f"{run.model} finished the stream without a score")


async def call_once(client: OllamaClient, run: RunSpec, text: str) -> tuple[Pred, Timing]:
    """One example through the serving adapters; returns raw outputs, never the input."""
    if run.variant == "sentiment-first":
        return await _score_first(client, run, text)
    verdict, raw = await classify_signal(client, run.spec, run.prompt, text)
    if isinstance(raw, DecisionResult):
        timing: Timing = {"wall_ms": raw.wall_ms}
        if raw.input_tokens is not None:
            timing["prompt_tokens"] = float(raw.input_tokens)
        pred: Pred = {
            "kind": "decision",
            "choice": raw.choice,
            "probs": dict(verdict.probabilities or {}),
            "model_confidence": raw.confidence,
        }
        return pred, timing
    if run.variant == "label":
        return _label_pred(verdict, raw), _chat_timing(raw)
    pred = {
        "kind": "ordinal",
        "score": verdict.score,
        "sign_probs": score_sign_probabilities(raw.content, raw.logprobs),
    }
    return pred, _chat_timing(raw)


# --- model residency -------------------------------------------------------------------------


async def model_digest(client: OllamaClient, model: str) -> str:
    for entry in await client.tags():
        if model in (entry.get("name"), entry.get("model")) and isinstance(entry.get("digest"), str):
            return str(entry["digest"])
    raise ModelUnavailable(f"{model} is not installed in Ollama", hint=f"Pull it with `ollama pull {model}`.")


async def make_resident(client: OllamaClient, run: RunSpec, *, log: TextIO = sys.stderr) -> None:
    """Unload every other model, then warm this one so the first timed call is not a cold load."""
    for entry in await client.ps():
        name = str(entry.get("name") or entry.get("model") or "")
        if name and name != run.model:
            try:
                await client.unload(name)
            except VtError as exc:
                print(f"warning: could not unload {name}: {exc.message}", file=log)
    await call_once(client, run, WARMUP_TEXT)


# --- cell run --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunStats:
    key: str
    path: Path
    total: int
    cached: int
    called: int
    errors: int


class Progress:
    """Counts only: neither inputs nor outputs reach the console."""

    def __init__(self, label: str, total: int, cached: int, *, every_s: float, out: TextIO) -> None:
        self.label, self.total, self.cached = label, total, cached
        self.every_s, self.out = every_s, out
        self.called = self.errors = 0
        self._t0 = self._last = time.monotonic()

    def tick(self, *, error: bool) -> None:
        self.called += 1
        self.errors += error
        now = time.monotonic()
        if now - self._last >= self.every_s:
            self._last = now
            self.emit()

    def emit(self, final: bool = False) -> None:
        rate = self.called / max(time.monotonic() - self._t0, 1e-9)
        status = "done" if final else "running"
        print(
            f"[{self.label}] {status} {self.cached + self.called}/{self.total} "
            f"(cached {self.cached}, new {self.called}, errors {self.errors}, {rate:.1f}/s)",
            file=self.out,
            flush=True,
        )


def _pred_error(exc: VtError) -> PredError:
    # Greedy seeded decoding repeats a schema failure, so it is final; transport failures retry on resume.
    retryable = not isinstance(exc, SchemaViolation) and (exc.retryable or isinstance(exc, ModelUnavailable))
    return PredError(exc.code.value, exc.message[:300], retryable)


async def _pool(workers: int, job: Callable[[], Coroutine[Any, Any, None]]) -> None:
    tasks = [asyncio.create_task(job()) for _ in range(workers)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_cell(
    client: OllamaClient,
    run: RunSpec,
    examples: Sequence[Example],
    *,
    split: str,
    model_digest: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    root: Path = PREDS_DIR,
    progress_every_s: float = 30.0,
    log: TextIO = sys.stderr,
) -> RunStats:
    """Run every example not yet in the cache; an unreachable Ollama aborts, other failures are recorded."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    with run.cache(split, model_digest, root) as cache:
        done = cache.done_ids()
        todo = deque(e for e in examples if e.id not in done)
        cached = len(examples) - len(todo)
        progress = Progress(
            f"{run.endpoint} {run.model} {run.name}", len(examples), cached, every_s=progress_every_s, out=log
        )

        async def worker() -> None:
            while todo:
                example = todo.popleft()
                t0 = time.perf_counter()
                try:
                    pred, timing = await call_once(client, run, example.text)
                except LlmUnreachable as exc:
                    raise RunAborted(f"{run.model} {run.name}: {exc.message}") from exc
                except VtError as exc:
                    wall = {"wall_ms": (time.perf_counter() - t0) * 1000}
                    cache.append(example.id, example.label, error=_pred_error(exc), t=wall)
                    progress.tick(error=True)
                else:
                    cache.append(example.id, example.label, pred=pred, t=timing)
                    progress.tick(error=False)

        if todo:
            await _pool(min(concurrency, len(todo)), worker)
        progress.emit(final=True)
        return RunStats(cache.key, cache.path, len(examples), cached, progress.called, progress.errors)
