"""Thin async adapters from (model spec, prompt, article text) to validated outputs and verdicts."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, overload

from pydantic import BaseModel, ValidationError

from veritrade.errors import SchemaViolation
from veritrade.extract.dom import split_sentences
from veritrade.inference.confidence import (
    LabelLogprobPolicy,
    OrdinalPolicy,
    ProbabilityPolicy,
    policy_for,
)
from veritrade.inference.incremental import (
    EntityClosed,
    JsonFieldScanner,
    ScanEvent,
    ScoreClosed,
    SummaryText,
)
from veritrade.inference.ollama import OllamaClient, OutputTruncated
from veritrade.inference.schemas import (
    ANALYZE_MODELS,
    AnalyzeOut,
    AnalyzeSentimentFirst,
    AnalyzeSummaryFirst,
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
# Product spec for every served summary; models overshoot it, so serving enforces it.
SUMMARY_MAX_SENTENCES = 3
SUMMARY_MAX_WORDS = 60
ELLIPSIS = "…"
# Terminal punctuation, closing quotes or brackets, then whitespace. The text end is no boundary:
# a cut right after "3." may sit inside "3.5".
_COMPLETE_SENTENCES = re.compile(r"(.*[.!?][\"'”’)\]]*)\s", re.DOTALL)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SummaryResult:
    output: SummaryOut
    chat: ChatResult
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    output: AnalyzeOut
    verdict: Verdict
    chat: ChatResult
    truncated: bool = False

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
        done_reason=second.done_reason,
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


def cut_to_sentence(text: str) -> str:
    """The text up to its last complete sentence, or all of it when no sentence has ended."""
    match = _COMPLETE_SENTENCES.match(text)
    return (match.group(1) if match else text).strip()


def cap_summary(
    text: str, max_sentences: int = SUMMARY_MAX_SENTENCES, max_words: int = SUMMARY_MAX_WORDS
) -> str:
    """Whole leading sentences within both caps; a first sentence over the word cap is cut at a word
    and ends with an ellipsis. Text already within the caps is returned unchanged."""
    sentences = split_sentences(text)
    kept: list[str] = []
    words = 0
    for sentence in sentences[:max_sentences]:
        count = len(sentence.split())
        if words + count > max_words:
            break
        kept.append(sentence)
        words += count
    if len(kept) == len(sentences):
        return text.strip()
    if kept:
        return " ".join(kept)
    head = " ".join(sentences[0].split()[:max_words]).rstrip(",;:")
    return head + ELLIPSIS


@overload
def _capped(out: SummaryOut) -> SummaryOut: ...
@overload
def _capped(out: AnalyzeOut) -> AnalyzeOut: ...
def _capped(out: SummaryOut | AnalyzeOut) -> SummaryOut | AnalyzeOut:
    summary = cap_summary(out.summary)
    return out if summary == out.summary else out.model_copy(update={"summary": summary})


def _salvage(spec: ModelSpec, endpoint: Endpoint, chat: ChatResult) -> tuple[int | None, str, str]:
    """Score, entity and sentence-cut summary decoded from JSON that num_predict cut off."""
    log.warning(
        "%s output from %s hit num_predict after %s eval tokens; salvaging the partial JSON",
        endpoint,
        spec.name,
        chat.timings.eval_tokens,
    )
    score: int | None = None
    entity = ""
    parts: list[str] = []
    for event in JsonFieldScanner().feed(chat.content):
        match event:
            case ScoreClosed(score=value):
                score = value
            case EntityClosed(entity=value):
                entity = value
            case SummaryText(delta=delta):
                parts.append(delta)
    summary = cut_to_sentence("".join(parts))
    if not summary:
        raise SchemaViolation(f"{spec.name} reached its output token limit before writing a summary")
    return score, entity, summary


def _salvage_summary(spec: ModelSpec, chat: ChatResult) -> SummaryResult:
    _, _, summary = _salvage(spec, "summary", chat)
    return SummaryResult(_capped(SummaryOut(summary=summary)), chat, truncated=True)


def _salvage_analyze(
    spec: ModelSpec, out_type: type[AnalyzeSentimentFirst] | type[AnalyzeSummaryFirst], chat: ChatResult
) -> AnalyzeResult:
    """A cut generation still needs its score; summary-first output cut before it stays a violation."""
    score, entity, summary = _salvage(spec, "analyze", chat)
    if score is None:
        raise SchemaViolation(f"{spec.name} reached its output token limit before writing the score")
    out = _capped(out_type.model_validate({"score": score, "entity": entity, "summary": summary}))
    return AnalyzeResult(out, score_verdict(spec, out.score), chat, truncated=True)


def _was_cut(chat: ChatResult, scanner: JsonFieldScanner) -> bool:
    return chat.done_reason == "length" and not scanner.done


async def summarize(client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str) -> SummaryResult:
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), "default")
    try:
        out, chat = await client.chat_json(
            spec.name, messages, SummaryOut, options=chat_options(spec, "summary"), retry_cut=False
        )
    except OutputTruncated as cut:
        return _salvage_summary(spec, cut.chat)
    return SummaryResult(_capped(out), chat)


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
                if _was_cut(item, scanner):
                    yield _salvage_summary(spec, item)
                    return
                scanner.close()
                yield SummaryResult(_capped(_validate(SummaryOut, item.content, spec.name)), item)
                return
            for event in scanner.feed(item):
                yield event


async def analyze(client: OllamaClient, spec: ModelSpec, prompt: PromptSpec, text: str) -> AnalyzeResult:
    order = spec.field_order or "sentiment-first"
    out_type = ANALYZE_MODELS[order]
    messages = prompt.messages(fit_input(text, spec.max_input_tokens), order)
    try:
        out, chat = await client.chat_json(
            spec.name, messages, out_type, options=chat_options(spec, "analyze"), retry_cut=False
        )
    except OutputTruncated as cut:
        return _salvage_analyze(spec, out_type, cut.chat)
    out = _capped(out)
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
                if _was_cut(item, scanner):
                    yield _salvage_analyze(spec, out_type, item)
                    return
                scanner.close()
                out = _capped(_validate(out_type, item.content, spec.name))
                yield AnalyzeResult(out, score_verdict(spec, out.score), item)
                return
            for event in scanner.feed(item):
                yield event
