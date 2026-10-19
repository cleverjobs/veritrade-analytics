"""Public API contract: request/response bodies, SSE event payloads and the error envelope."""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from veritrade.errors import ErrorCode
from veritrade.extract.types import ExtractorName, ParserName, Scope
from veritrade.inference.types import Endpoint, FieldOrder, ModelKind, PolicyName, Sentiment


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- requests -------------------------------------------------------------------------------


class _ArticleRequest(_Strict):
    url: AnyHttpUrl
    model: str | None = Field(None, description="Allowlisted Ollama tag; endpoint default when omitted.")
    parser: ParserName = "selectolax"
    extractor: ExtractorName = "cascade"
    no_cache: bool = Field(False, description="Bypass fetch and result caches (used by latency benchmarks).")


class SignalRequest(_ArticleRequest):
    scope: Scope = Field("lede", description="lede = headline + opening paragraphs; full = whole article.")


class SummaryRequest(_ArticleRequest):
    pass


class AnalyzeRequest(_ArticleRequest):
    scope: Scope = "full"


# --- shared response parts ------------------------------------------------------------------


class CacheInfo(_Strict):
    fetch: Literal["hit", "miss", "joined", "bypass"]
    result: Literal["hit", "miss", "bypass"]


class InferenceTimings(_Strict):
    wall_ms: float
    ttft_ms: float | None = None
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    cold_start: bool | None = Field(
        None,
        description="True when the model was not loaded in Ollama (or was loaded with another context size) "
        "as the request arrived, so the inference time includes loading it; null when unknown.",
    )


class Timings(_Strict):
    total_ms: float
    fetch_ms: float
    consent_ms: float | None = None
    parse_ms: float
    extract_ms: float
    inference: InferenceTimings | None = None
    cache: CacheInfo


class ArticleMeta(_Strict):
    url: str
    final_url: str
    title: str
    source: str | None = Field(None, description="Key from sources.toml when the domain is a known source.")
    published_at: str | None = None
    endpoint: Endpoint
    model: str
    model_kind: ModelKind
    prompt_version: str
    parser: ParserName
    extractor: ExtractorName
    extraction_method: str = Field(description="Strategy that produced the body (cascade resolves to one).")
    scope: Scope
    input_chars: int
    word_count: int
    field_order: FieldOrder | None = None
    cached: bool
    backend: str = Field(description="Inference backend label, e.g. native-metal or docker-cpu.")


class SignalResult(_Strict):
    sentiment: Sentiment
    abstained: bool = Field(description="True when the call fell below the frozen confidence threshold.")
    policy: PolicyName
    confidence: float | None = None
    score: int | None = Field(None, ge=-2, le=2)
    probabilities: dict[Sentiment, float] | None = None
    threshold: float | None = Field(
        None,
        description="Frozen cutoff this call was held to: the minimum winning-class probability "
        "(probability, label-logprob) or the minimum |score| (ordinal).",
    )
    directional_tau: float | None = Field(
        None,
        description="When set, a neutral lead still yields the likelier directional class once that class "
        "reaches this probability.",
    )


# --- responses ------------------------------------------------------------------------------


class SignalResponse(_Strict):
    meta: ArticleMeta
    signal: SignalResult
    timings: Timings


class SummaryResponse(_Strict):
    meta: ArticleMeta
    summary: str
    truncated: bool = Field(False, description="True when the summary was cut at the model's output limit.")
    timings: Timings


class AnalyzeResponse(_Strict):
    meta: ArticleMeta
    signal: SignalResult
    entity: str | None = None
    summary: str
    truncated: bool = Field(False, description="True when the summary was cut at the model's output limit.")
    timings: Timings


class ErrorBody(_Strict):
    code: ErrorCode
    message: str
    hint: str | None = None
    retryable: bool = False
    upstream_status: int | None = None


class ErrorResponse(_Strict):
    error: ErrorBody


class SignalRule(_Strict):
    policy: PolicyName
    threshold: float
    directional_tau: float | None = None


class ModelInfo(_Strict):
    name: str
    kind: ModelKind
    endpoints: list[Endpoint]
    installed: bool
    size_bytes: int | None = None
    evaluated: bool = Field(True, description="False for plumbing-only models (e.g. the CI decision model).")
    signal_rules: dict[Endpoint, SignalRule] = Field(
        default_factory=dict,
        description="Frozen confidence rule for each endpoint of this model that returns a signal (signal, "
        "analyze). A label-logprob rule still falls back to ordinal on a label collision; the listed rule "
        "is the primary one.",
    )


class ModelsResponse(_Strict):
    models: list[ModelInfo]
    defaults: dict[Endpoint, str]
    parsers: list[ParserName]
    extractors: list[ExtractorName]
    scopes: list[Scope]


class HealthResponse(_Strict):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    ollama_version: str | None = None
    backend: str
    version: str


# --- SSE event payloads (event name -> data model) ------------------------------------------
# /summary/stream: meta -> summary_delta* -> done
# /analyze/stream: meta -> sentiment (exactly once, position follows field order) + summary_delta* -> done
# Any failure after the stream opens emits a single `error` event and closes.


class MetaEvent(_Strict):
    meta: ArticleMeta
    fetch_ms: float
    parse_ms: float
    extract_ms: float
    cache: CacheInfo
    cold_start: bool | None = Field(
        None, description="True when the model has to be loaded before its first token; null when unknown."
    )


class SentimentEvent(_Strict):
    signal: SignalResult
    entity: str | None = None
    elapsed_ms: float = Field(description="Milliseconds since the request was received.")


class SummaryDeltaEvent(_Strict):
    text: str


class SummaryDoneEvent(_Strict):
    result: SummaryResponse


class AnalyzeDoneEvent(_Strict):
    result: AnalyzeResponse


class ErrorEvent(_Strict):
    error: ErrorBody


SSE_EVENT_NAMES = ("meta", "sentiment", "summary_delta", "done", "error")
