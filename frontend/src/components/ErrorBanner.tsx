import type { ApiErrorCode } from "@/lib/api";

import styles from "./ErrorBanner.module.css";

export interface ErrorLike {
  code: ApiErrorCode;
  message: string;
  hint?: string | null;
  status?: number | null;
  retryable?: boolean;
  upstreamStatus?: number | null;
}

export const ERROR_TITLES: Record<ApiErrorCode, string> = {
  invalid_url: "Invalid URL",
  blocked_destination: "Destination not allowed",
  model_not_allowed: "Model not allowed",
  paywalled: "Paywalled article",
  consent_required: "Consent wall",
  extraction_empty: "No article text found",
  unsupported_content: "Unsupported content",
  too_large: "Page too large",
  source_blocked: "Blocked by the source",
  upstream_http: "Source returned an error",
  schema_violation: "Malformed model output",
  model_unavailable: "Model unavailable",
  llm_unreachable: "Ollama unreachable",
  upstream_timeout: "Source timed out",
  inference_timeout: "Model timed out",
  network_error: "API unreachable",
  invalid_request: "Invalid request",
  http_error: "Unexpected HTTP error",
  bad_response: "Unexpected response",
};

// Used only when the server sent no hint of its own.
const FALLBACK_HINTS: Partial<Record<ApiErrorCode, string>> = {
  invalid_url: "Paste a full http(s) article URL.",
  blocked_destination: "Only public hosts on ports 80 and 443 can be fetched.",
  model_not_allowed: "Pick a model from the list; it mirrors the server allowlist.",
  paywalled: "The article is behind a paywall. Try a free source.",
  consent_required: "The site demands a consent choice the fetcher could not make. Try another source.",
  extraction_empty: "Try another extractor or parser.",
  unsupported_content: "The URL must point to an HTML article page.",
  too_large: "The page is larger than the fetch cap.",
  source_blocked: "The publisher refused automated access. Try another source.",
  upstream_http: "The source answered with an error status. Retry later.",
  schema_violation: "Retry; the model output did not match the expected schema.",
  model_unavailable: "Pull the model in Ollama or pick another one.",
  llm_unreachable: "Start Ollama or check OLLAMA_BASE_URL on the backend.",
  upstream_timeout: "The source was too slow to answer. Retry.",
  inference_timeout: "The model took too long. Try a smaller model.",
  network_error: "Start the backend (http://localhost:8000 in development) and retry.",
};

interface ErrorBannerProps {
  error: ErrorLike;
  /** Which request(s) failed, e.g. "Signal and summary". */
  context?: string;
}

export function ErrorBanner({ error, context }: ErrorBannerProps) {
  const hint = error.hint || FALLBACK_HINTS[error.code];
  const status = error.upstreamStatus ?? null;
  return (
    <div role="alert" className={styles.banner} data-code={error.code} data-testid="error-banner">
      <p className={styles.title}>
        {context ? <span className={styles.context}>{context}: </span> : null}
        {ERROR_TITLES[error.code]} <code className={styles.code}>{error.code}</code>
        {error.status ? <span className={styles.status}>HTTP {error.status}</span> : null}
      </p>
      <p className={styles.message}>{error.message}</p>
      {hint ? <p className={styles.hint}>{hint}</p> : null}
      {status !== null || error.retryable ? (
        <p className={styles.extra}>
          {status !== null ? `Upstream status ${status}. ` : ""}
          {error.retryable ? "Retrying may help." : ""}
        </p>
      ) : null}
    </div>
  );
}
