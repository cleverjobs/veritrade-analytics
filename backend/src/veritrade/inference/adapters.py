"""Thin async adapters from (model spec, prompt, article text) to validated outputs and verdicts."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from veritrade.errors import SchemaViolation
from veritrade.inference.confidence import (
    LabelLogprobPolicy,
    OrdinalPolicy,
    ProbabilityPolicy,
    policy_for,
)
from veritrade.inference.incremental import JsonFieldScanner, ScanEvent
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.schemas import (
    ANALYZE_MODELS,
    AnalyzeOut,
    LabelOut,
    OrdinalOut,
    SummaryOut,
    ollama_format,
)
from veritrade.inference.types import ChatResult, DecisionResult, Endpoint, OllamaTimings, Verdict
from veritrade.models_registry import ModelSpec, SignalSchema
from veritrade.prompts.registry import PromptSpec

CHARS_PER_TOKEN = 4
TOP_LOGPROBS = 20


@dataclass(frozen=True, slots=True)
class SummaryResult:
    output: SummaryOut
    chat: ChatResult


@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    output: AnalyzeOut
    verdict: Verdict
    chat: ChatResult

    @property
    def entity(self) -> str | None:
        return self.output.entity.strip() or None


def fit_input(text: str, max_input_tokens: int) -> str:
    """Cap at max_input_tokens (chars/4), cutting at the last whitespace so no word is split."""
    limit = max_input_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = max(text.rfind(" ", 0, limit), text.rfind("\n", 0, limit))
    return text[: cut if cut > limit // 2 else limit].rstrip()


def chat_options(spec: ModelSpec, endpoint: Endpoint) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if spec.num_ctx is not None:
        options["num_ctx"] = spec.num_ctx
    if endpoint in spec.num_predict:
        options["num_predict"] = spec.num_predict[endpoint]
    return options


def score_verdict(spec: ModelSpec, score: int) -> Verdict:
    """Verdict for a streamed or /analyze score; both always use the ordinal policy."""
    return OrdinalPolicy(spec.ordinal_t).decide(score)


def _sum(a: float | None, b: float | None) -> float | None:
    return None if a is None and b is None else (a or 0) + (b or 0)


def _sum_int(a: int | None, b: int | None) -> int | None:
    return None if a is None and b is None else (a or 0) + (b or 0)


def _combine(first: ChatResult, second: ChatResult) -> ChatResult:
    # The fallback call's output decides; the reported cost covers both calls.
    t1, t2 = first.timings, second.timings
    return ChatResult(
        content=second.content,
        wall_ms=first.wall_ms + second.wall_ms,
        timings=OllamaTimings(
            load_ms=_sum(t1.load_ms, t2.load_ms),
            prompt_eval_ms=_sum(t1.prompt_eval_ms, t2.prompt_eval_ms),
            eval_ms=_sum(t1.eval_ms, t2.eval_ms),
            prompt_tokens=_sum_int(t1.prompt_tokens, t2.prompt_tokens),
            eval_tokens=_sum_int(t1.eval_tokens, t2.eval_tokens),
            total_ms=_sum(t1.total_ms, t2.total_ms),
        ),
        logprobs=second.logprobs,
    )


async def _signal_call(
    client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str, schema: SignalSchema
) -> tuple[LabelOut | OrdinalOut, ChatResult]:
    out_type: type[LabelOut] | type[OrdinalOut] = LabelOut if schema == "label" else OrdinalOut
    return await client.chat_json(
        spec.name,
        prompt.messages(text, schema),
        out_type,
        options=chat_options(spec, "signal"),
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )


async def classify_signal(
    client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str
) -> tuple[Verdict, ChatResult | DecisionResult]:
    """Label-logprob falls back label -> ordinal call (score sign) -> ordinal policy on the score."""
    text = fit_input(text, spec.max_input_tokens)
    policy = policy_for(spec, endpoint="signal")
    if isinstance(policy, ProbabilityPolicy):
        decision = await client.decide_sentiment(
            spec.name, prompt.state(text), prompt.instructions, prompt.criteria_map
        )
        return policy.decide(decision.probabilities), decision
    out, chat = await _signal_call(client, spec, prompt, text, spec.signal_schema or "ordinal")
    if isinstance(policy, LabelLogprobPolicy):
        verdict = policy.decide(out, chat)
        if verdict is None and isinstance(out, LabelOut):
            out, second = await _signal_call(client, spec, prompt, text, "ordinal")
            chat = _combine(chat, second)
            verdict = policy.decide(out, chat)
        if verdict is not None:
            return verdict, chat
    if not isinstance(out, OrdinalOut):
        raise SchemaViolation(f"{spec.name}: a label output needs the label-logprob policy")
    return OrdinalPolicy(spec.ordinal_t).decide(out.score), chat


def _validate[T: BaseModel](out_type: type[T], content: str, model: str) -> T:
    try:
        return out_type.model_validate_json(content)
    except ValidationError as exc:
        raise SchemaViolation(
            f"{model} streamed output that does not match {out_type.__name__}", hint=str(exc)[:300]
        ) from exc


async def summarize(client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str) -> SummaryResult:
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), "default")
    out, chat = await client.chat_json(spec.name, messages, SummaryOut, options=chat_options(spec, "summary"))
    return SummaryResult(out, chat)


async def summarize_stream(
    client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str
) -> AsyncGenerator[ScanEvent | SummaryResult]:
    """Summary text deltas as they decode, then the validated result (no retry once tokens are out)."""
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), "default")
    scanner = JsonFieldScanner()
    stream = client.chat_stream(
        spec.name, messages, format=ollama_format(SummaryOut), options=chat_options(spec, "summary")
    )
    async with aclosing(stream):
        async for item in stream:
            if isinstance(item, ChatResult):
                scanner.close()
                yield SummaryResult(_validate(SummaryOut, item.content, spec.name), item)
                return
            for event in scanner.feed(item):
                yield event


async def analyze(client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str) -> AnalyzeResult:
    order = spec.field_order or "sentiment-first"
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), order)
    out, chat = await client.chat_json(
        spec.name, messages, ANALYZE_MODELS[order], options=chat_options(spec, "analyze")
    )
    return AnalyzeResult(out, score_verdict(spec, out.score), chat)


async def analyze_stream(
    client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str
) -> AsyncGenerator[ScanEvent | AnalyzeResult]:
    """Scanner events in generation order (ScoreClosed first when sentiment-first), then the result."""
    order = spec.field_order or "sentiment-first"
    out_type = ANALYZE_MODELS[order]
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), order)
    scanner = JsonFieldScanner()
    stream = client.chat_stream(
        spec.name, messages, format=ollama_format(out_type), options=chat_options(spec, "analyze")
    )
    async with aclosing(stream):
        async for item in stream:
            if isinstance(item, ChatResult):
                scanner.close()
                out = _validate(out_type, item.content, spec.name)
                yield AnalyzeResult(out, score_verdict(spec, out.score), item)
                return
            for event in scanner.feed(item):
                yield event
