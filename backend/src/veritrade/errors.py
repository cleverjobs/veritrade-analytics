from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class ErrorCode(StrEnum):
    INVALID_URL = "invalid_url"
    BLOCKED_DESTINATION = "blocked_destination"
    MODEL_NOT_ALLOWED = "model_not_allowed"
    PAYWALLED = "paywalled"
    CONSENT_REQUIRED = "consent_required"
    EXTRACTION_EMPTY = "extraction_empty"
    UNSUPPORTED_CONTENT = "unsupported_content"
    TOO_LARGE = "too_large"
    SOURCE_BLOCKED = "source_blocked"
    UPSTREAM_HTTP = "upstream_http"
    SCHEMA_VIOLATION = "schema_violation"
    MODEL_UNAVAILABLE = "model_unavailable"
    LLM_UNREACHABLE = "llm_unreachable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    INFERENCE_TIMEOUT = "inference_timeout"


ERROR_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_URL: 400,
    ErrorCode.BLOCKED_DESTINATION: 400,
    ErrorCode.MODEL_NOT_ALLOWED: 400,
    ErrorCode.PAYWALLED: 422,
    ErrorCode.CONSENT_REQUIRED: 422,
    ErrorCode.EXTRACTION_EMPTY: 422,
    ErrorCode.UNSUPPORTED_CONTENT: 422,
    ErrorCode.TOO_LARGE: 422,
    ErrorCode.SOURCE_BLOCKED: 502,
    ErrorCode.UPSTREAM_HTTP: 502,
    ErrorCode.SCHEMA_VIOLATION: 502,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.LLM_UNREACHABLE: 503,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.INFERENCE_TIMEOUT: 504,
}


class VtError(Exception):
    code: ClassVar[ErrorCode]
    retryable_default: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool | None = None,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.retryable = self.retryable_default if retryable is None else retryable
        self.upstream_status = upstream_status

    @property
    def http_status(self) -> int:
        return ERROR_STATUS[self.code]


class InvalidUrl(VtError):
    code = ErrorCode.INVALID_URL


class BlockedDestination(VtError):
    code = ErrorCode.BLOCKED_DESTINATION


class ModelNotAllowed(VtError):
    code = ErrorCode.MODEL_NOT_ALLOWED


class Paywalled(VtError):
    code = ErrorCode.PAYWALLED


class ConsentRequired(VtError):
    code = ErrorCode.CONSENT_REQUIRED


class ExtractionEmpty(VtError):
    code = ErrorCode.EXTRACTION_EMPTY


class UnsupportedContent(VtError):
    code = ErrorCode.UNSUPPORTED_CONTENT


class TooLarge(VtError):
    code = ErrorCode.TOO_LARGE


class SourceBlocked(VtError):
    code = ErrorCode.SOURCE_BLOCKED


class UpstreamHttp(VtError):
    code = ErrorCode.UPSTREAM_HTTP
    retryable_default = True


class SchemaViolation(VtError):
    code = ErrorCode.SCHEMA_VIOLATION
    retryable_default = True


class ModelUnavailable(VtError):
    code = ErrorCode.MODEL_UNAVAILABLE


class LlmUnreachable(VtError):
    code = ErrorCode.LLM_UNREACHABLE
    retryable_default = True


class UpstreamTimeout(VtError):
    code = ErrorCode.UPSTREAM_TIMEOUT
    retryable_default = True


class InferenceTimeout(VtError):
    code = ErrorCode.INFERENCE_TIMEOUT
    retryable_default = True
