import type { components } from "./api-schema";

type Schemas = components["schemas"];

export type SignalRequest = Schemas["SignalRequest"];
export type SummaryRequest = Schemas["SummaryRequest"];
export type AnalyzeRequest = Schemas["AnalyzeRequest"];
export type SignalResponse = Schemas["SignalResponse"];
export type SummaryResponse = Schemas["SummaryResponse"];
export type AnalyzeResponse = Schemas["AnalyzeResponse"];
export type ModelsResponse = Schemas["ModelsResponse"];
export type HealthResponse = Schemas["HealthResponse"];
export type ModelInfo = Schemas["ModelInfo"];
export type ArticleMeta = Schemas["ArticleMeta"];
export type SignalResult = Schemas["SignalResult"];
export type Timings = Schemas["Timings"];
export type InferenceTimings = Schemas["InferenceTimings"];
export type CacheInfo = Schemas["CacheInfo"];
export type ErrorBody = Schemas["ErrorBody"];
export type ErrorCode = Schemas["ErrorCode"];

export type ParserName = SignalRequest["parser"];
export type ExtractorName = SignalRequest["extractor"];
export type Scope = SignalRequest["scope"];
export type Endpoint = ArticleMeta["endpoint"];
export type Sentiment = SignalResult["sentiment"];
export type PolicyName = SignalResult["policy"];

// The generated request types mark defaulted fields as required; callers only need the url.
type Body<T extends { url: string }> = Pick<T, "url"> & Partial<Omit<T, "url">>;
export type SignalBody = Body<SignalRequest>;
export type SummaryBody = Body<SummaryRequest>;
export type AnalyzeBody = Body<AnalyzeRequest>;

export const PARSERS: readonly ParserName[] = ["selectolax", "lxml", "bs4", "bs4-lxml"];
export const EXTRACTORS: readonly ExtractorName[] = [
  "cascade",
  "jsonld",
  "profile",
  "density",
  "trafilatura",
  "trafilatura-precision",
  "readability",
  "newspaper",
  "justext",
  "strip-tags",
];
export const SCOPES: readonly Scope[] = ["lede", "full"];

const SERVER_ERROR_CODES: Record<ErrorCode, true> = {
  invalid_url: true,
  blocked_destination: true,
  model_not_allowed: true,
  paywalled: true,
  consent_required: true,
  extraction_empty: true,
  unsupported_content: true,
  too_large: true,
  source_blocked: true,
  upstream_http: true,
  schema_violation: true,
  model_unavailable: true,
  llm_unreachable: true,
  upstream_timeout: true,
  inference_timeout: true,
};
export const ERROR_CODES = Object.keys(SERVER_ERROR_CODES) as ErrorCode[];

/** Failures detected in the browser rather than reported by the API envelope. */
export type ClientErrorCode = "network_error" | "invalid_request" | "http_error" | "bad_response";
export type ApiErrorCode = ErrorCode | ClientErrorCode;

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && Object.hasOwn(SERVER_ERROR_CODES, value);
}

export interface ApiErrorInit {
  code: ApiErrorCode;
  message: string;
  hint?: string | null;
  status?: number | null;
  retryable?: boolean;
  upstreamStatus?: number | null;
}

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly hint: string | null;
  readonly status: number | null;
  readonly retryable: boolean;
  readonly upstreamStatus: number | null;

  constructor(init: ApiErrorInit) {
    super(init.message);
    this.name = "ApiError";
    this.code = init.code;
    this.hint = init.hint ?? null;
    this.status = init.status ?? null;
    this.retryable = init.retryable ?? false;
    this.upstreamStatus = init.upstreamStatus ?? null;
  }

  static fromBody(body: ErrorBody, status: number | null = null): ApiError {
    return new ApiError({
      code: body.code,
      message: body.message,
      hint: body.hint,
      status,
      retryable: body.retryable,
      upstreamStatus: body.upstream_status,
    });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asErrorBody(value: unknown): ErrorBody | null {
  if (!isRecord(value) || !isErrorCode(value.code) || typeof value.message !== "string") return null;
  return {
    code: value.code,
    message: value.message,
    hint: typeof value.hint === "string" ? value.hint : null,
    retryable: value.retryable === true,
    upstream_status: typeof value.upstream_status === "number" ? value.upstream_status : null,
  };
}

/** Maps any error payload (our envelope, FastAPI validation detail, or junk) to an ApiError. */
export function errorFromPayload(status: number, payload: unknown): ApiError {
  const envelope = isRecord(payload) ? asErrorBody(payload.error) : null;
  if (envelope) return ApiError.fromBody(envelope, status);

  // FastAPI request validation runs before our handlers and answers {"detail": [...]}.
  if (isRecord(payload) && Array.isArray(payload.detail)) {
    const first: unknown = payload.detail[0];
    const msg = isRecord(first) && typeof first.msg === "string" ? first.msg : "Request validation failed.";
    const loc = isRecord(first) && Array.isArray(first.loc) ? first.loc.slice(1).join(".") : "";
    return new ApiError({
      code: "invalid_request",
      message: loc ? `${loc}: ${msg}` : msg,
      status,
    });
  }

  return new ApiError({
    code: "http_error",
    message: `The API answered with HTTP ${status}.`,
    status,
    retryable: status >= 500,
  });
}

export async function readError(res: Response): Promise<ApiError> {
  let text = "";
  try {
    text = await res.text();
  } catch (err) {
    if (isAbortError(err)) throw err;
  }
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = null;
  }
  return errorFromPayload(res.status, payload);
}

export function isAbortError(err: unknown): boolean {
  return typeof err === "object" && err !== null && (err as { name?: unknown }).name === "AbortError";
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new ApiError({ code: "bad_response", message });
}

/** Empty base means same origin, which is how FastAPI serves the static export in production. */
export function apiUrl(path: string): string {
  const base = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/+$/, "");
  if (base) return `${base}${path}`;
  return typeof window === "undefined" ? path : new URL(path, window.location.origin).toString();
}

/** fetch() that turns network failures and non-2xx answers into ApiError; aborts pass through. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch (err) {
    if (isAbortError(err)) throw err;
    throw new ApiError({
      code: "network_error",
      message: "Could not reach the API.",
      retryable: true,
    });
  }
  if (!res.ok) throw await readError(res);
  return res;
}

async function readJson<T>(res: Response): Promise<T> {
  try {
    return (await res.json()) as T;
  } catch (err) {
    if (isAbortError(err)) throw err;
    throw new ApiError({
      code: "bad_response",
      message: "The API returned invalid JSON.",
      status: res.status,
    });
  }
}

export function jsonPost(body: unknown, signal?: AbortSignal, accept = "application/json"): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: accept },
    body: JSON.stringify(body),
    signal,
  };
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return readJson<T>(await apiFetch(path, jsonPost(body, signal)));
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return readJson<HealthResponse>(await apiFetch("/api/health", { signal }));
}

export async function getModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return readJson<ModelsResponse>(await apiFetch("/api/models", { signal }));
}

export function postSignal(body: SignalBody, signal?: AbortSignal): Promise<SignalResponse> {
  return postJson<SignalResponse>("/api/signal", body, signal);
}

export function postSummary(body: SummaryBody, signal?: AbortSignal): Promise<SummaryResponse> {
  return postJson<SummaryResponse>("/api/summary", body, signal);
}

export function postAnalyze(body: AnalyzeBody, signal?: AbortSignal): Promise<AnalyzeResponse> {
  return postJson<AnalyzeResponse>("/api/analyze", body, signal);
}
