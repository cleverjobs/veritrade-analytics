"""Steps every article endpoint shares: resolve, check the result cache, load, describe."""

from __future__ import annotations

from dataclasses import dataclass

from veritrade.api.contracts import (
    AnalyzeDoneEvent,
    AnalyzeRequest,
    AnalyzeResponse,
    ArticleMeta,
    CacheInfo,
    ErrorEvent,
    InferenceTimings,
    MetaEvent,
    SentimentEvent,
    SignalRequest,
    SignalResponse,
    SignalResult,
    SummaryDeltaEvent,
    SummaryDoneEvent,
    SummaryRequest,
    SummaryResponse,
    Timings,
)
from veritrade.extract.scope import build_input
from veritrade.extract.types import Scope
from veritrade.fetch import normalize_url
from veritrade.inference.types import ChatResult, DecisionResult, Endpoint, Verdict
from veritrade.models_registry import ModelSpec
from veritrade.prompts.registry import PromptSpec
from veritrade.services.article import LoadedArticle, load_article
from veritrade.services.cache import ResultKey
from veritrade.services.runtime import Runtime
from veritrade.timing import StageTimer

ArticleRequest = SignalRequest | SummaryRequest | AnalyzeRequest
StreamPayload = (
    MetaEvent | SentimentEvent | SummaryDeltaEvent | SummaryDoneEvent | AnalyzeDoneEvent | ErrorEvent
)


@dataclass(frozen=True, slots=True)
class Job:
    """A request with its model and prompt resolved; nothing has touched the network yet."""

    runtime: Runtime
    endpoint: Endpoint
    request: ArticleRequest
    url: str
    scope: Scope
    spec: ModelSpec
    prompt: PromptSpec
    key: ResultKey | None
    clock: StageTimer

    @classmethod
    def resolve(
        cls, runtime: Runtime, endpoint: Endpoint, request: ArticleRequest, *, scope: Scope, clock: StageTimer
    ) -> Job:
        spec = runtime.registry.resolve(endpoint, request.model)
        prompt = runtime.prompt(endpoint, spec)
        url = str(request.url)
        key = None
        if not request.no_cache:
            key = ResultKey(
                endpoint=endpoint,
                url=normalize_url(url),
                model=spec.name,
                prompt_version=prompt.version,
                parser=request.parser,
                extractor=request.extractor,
                scope=scope,
            )
        return cls(runtime, endpoint, request, url, scope, spec, prompt, key, clock)

    def cached[R: (SignalResponse, SummaryResponse, AnalyzeResponse)](self, kind: type[R]) -> R | None:
        """The stored response marked as cached, with timings that describe this request."""
        hit = self.runtime.results.get(self.key) if self.key is not None else None
        if not isinstance(hit, kind):
            return None
        # A result hit runs no stage, so stage timings are zero and the article counts as a fetch hit.
        timings = Timings(
            total_ms=self.clock.elapsed_ms(),
            fetch_ms=0.0,
            parse_ms=0.0,
            extract_ms=0.0,
            cache=CacheInfo(fetch="hit", result="hit"),
        )
        meta = hit.meta.model_copy(update={"cached": True})
        return hit.model_copy(update={"meta": meta, "timings": timings})

    def remember(self, response: SignalResponse | SummaryResponse | AnalyzeResponse) -> None:
        if self.key is not None:
            self.runtime.results.put(self.key, response)

    async def prepare(self) -> Prepared:
        loaded = await load_article(
            self.runtime,
            self.url,
            parser=self.request.parser,
            extractor=self.request.extractor,
            no_cache=self.request.no_cache,
        )
        text = build_input(loaded.article, self.scope, self.spec.max_input_tokens)
        return Prepared(
            job=self,
            loaded=loaded,
            text=text,
            meta=self._meta(loaded, text),
            cache=CacheInfo(fetch=loaded.fetch_status, result="bypass" if self.key is None else "miss"),
        )

    def _meta(self, loaded: LoadedArticle, text: str) -> ArticleMeta:
        article = loaded.article
        return ArticleMeta(
            url=self.url,
            final_url=loaded.page.final_url,
            title=article.title,
            source=article.source,
            published_at=article.published_at,
            endpoint=self.endpoint,
            model=self.spec.name,
            model_kind=self.spec.kind,
            prompt_version=self.prompt.version,
            parser=self.request.parser,
            extractor=self.request.extractor,
            extraction_method=article.method,
            scope=self.scope,
            input_chars=len(text),
            word_count=article.word_count,
            field_order=self.spec.field_order if self.endpoint == "analyze" else None,
            cached=False,
            backend=self.runtime.settings.ollama_backend_label,
        )


@dataclass(frozen=True, slots=True)
class Prepared:
    """A job with its article loaded and the model input built; only inference is left."""

    job: Job
    loaded: LoadedArticle
    text: str
    meta: ArticleMeta
    cache: CacheInfo

    def timings(self, inference: InferenceTimings | None) -> Timings:
        loaded = self.loaded
        return Timings(
            total_ms=self.job.clock.elapsed_ms(),
            fetch_ms=loaded.fetch_ms,
            consent_ms=loaded.consent_ms,
            parse_ms=loaded.article.parse_ms,
            extract_ms=loaded.article.extract_ms,
            inference=inference,
            cache=self.cache,
        )

    def meta_event(self) -> MetaEvent:
        article = self.loaded.article
        return MetaEvent(
            meta=self.meta,
            fetch_ms=self.loaded.fetch_ms,
            parse_ms=article.parse_ms,
            extract_ms=article.extract_ms,
            cache=self.cache,
        )


def cached_meta_event(response: SummaryResponse | AnalyzeResponse) -> MetaEvent:
    timings = response.timings
    return MetaEvent(
        meta=response.meta,
        fetch_ms=timings.fetch_ms,
        parse_ms=timings.parse_ms,
        extract_ms=timings.extract_ms,
        cache=timings.cache,
    )


def signal_result(verdict: Verdict) -> SignalResult:
    return SignalResult(
        sentiment=verdict.sentiment,
        abstained=verdict.abstained,
        policy=verdict.policy,
        confidence=verdict.confidence,
        score=verdict.score,
        probabilities=verdict.probabilities,
    )


def inference_timings(raw: ChatResult | DecisionResult, *, ttft_ms: float | None = None) -> InferenceTimings:
    """ttft_ms is passed in by streams, which measure it from request receipt to the first summary text."""
    if isinstance(raw, DecisionResult):
        return InferenceTimings(wall_ms=raw.wall_ms, prompt_tokens=raw.input_tokens)
    t = raw.timings
    return InferenceTimings(
        wall_ms=raw.wall_ms,
        ttft_ms=ttft_ms,
        load_ms=t.load_ms,
        prompt_eval_ms=t.prompt_eval_ms,
        eval_ms=t.eval_ms,
        prompt_tokens=t.prompt_tokens,
        eval_tokens=t.eval_tokens,
    )
