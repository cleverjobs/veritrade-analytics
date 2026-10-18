// Synthetic API payloads for tests and mocked dev runs; the article text is invented.
import type {
  AnalyzeResponse,
  ArticleMeta,
  Endpoint,
  ExtractorName,
  HealthResponse,
  ModelsResponse,
  ParserName,
  Scope,
  SignalResponse,
  SignalResult,
  SummaryResponse,
  Timings,
} from "@/lib/api";
import type { MetaEventData, SentimentEventData } from "@/lib/sse";

export const FIXTURE_URL = "https://news.example.com/markets/acme-q3-results";
export const SUMMARY_CHUNKS = [
  "Acme Corp beat third-quarter revenue estimates by 6% ",
  "and raised its full-year outlook, ",
  "sending shares up 8% in premarket trading.",
];
export const SUMMARY_TEXT = SUMMARY_CHUNKS.join("");
export const ENTITY = "Acme Corp (ACME)";

export const models: ModelsResponse = {
  models: [
    { name: "clef-flash:9b-q8_0", kind: "decision", endpoints: ["signal"], installed: true, evaluated: true },
    { name: "nimble:9b-q4_K_M", kind: "decision", endpoints: ["signal"], installed: true, evaluated: true },
    { name: "tev1:0.8b-q8_0", kind: "decision", endpoints: ["signal"], installed: true, evaluated: false },
    {
      name: "qwen3.5:0.8b-q8_0",
      kind: "generative",
      endpoints: ["signal", "summary", "analyze"],
      installed: true,
      evaluated: true,
    },
    {
      name: "qwen3.5:2b-q4_K_M",
      kind: "generative",
      endpoints: ["summary"],
      installed: true,
      evaluated: true,
    },
    {
      name: "qwen3.5:4b-q4_K_M",
      kind: "generative",
      endpoints: ["summary", "analyze"],
      installed: true,
      evaluated: true,
    },
    {
      name: "qwen3:4b-instruct-2507-q4_K_M",
      kind: "generative",
      endpoints: ["analyze"],
      installed: true,
      evaluated: true,
    },
    { name: "ministral-3:3b", kind: "generative", endpoints: ["analyze"], installed: true, evaluated: true },
    {
      name: "qwen3.5:9b-q4_K_M",
      kind: "generative",
      endpoints: ["analyze"],
      installed: false,
      evaluated: true,
    },
  ],
  defaults: { signal: "nimble:9b-q4_K_M", summary: "qwen3.5:4b-q4_K_M", analyze: "qwen3.5:4b-q4_K_M" },
  parsers: ["selectolax", "lxml", "bs4", "bs4-lxml"],
  extractors: [
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
  ],
  scopes: ["lede", "full"],
};

export const health: HealthResponse = {
  status: "ok",
  ollama_reachable: true,
  ollama_version: "0.40.6",
  backend: "native-metal",
  version: "0.1.0",
};

export interface MetaOptions {
  endpoint: Endpoint;
  model?: string | null;
  parser?: ParserName;
  extractor?: ExtractorName;
  scope?: Scope;
}

export function articleMeta({ endpoint, model, parser, extractor, scope }: MetaOptions): ArticleMeta {
  const resolvedModel = model || models.defaults[endpoint];
  const kind = models.models.find((m) => m.name === resolvedModel)?.kind ?? "generative";
  const resolvedExtractor = extractor ?? "cascade";
  return {
    url: FIXTURE_URL,
    final_url: FIXTURE_URL,
    title: "Acme Corp beats estimates and lifts full-year outlook",
    source: null,
    published_at: "2026-10-16T12:30:00Z",
    endpoint,
    model: resolvedModel,
    model_kind: kind,
    prompt_version: "v1-fixture",
    parser: parser ?? "selectolax",
    extractor: resolvedExtractor,
    extraction_method: resolvedExtractor === "cascade" ? "jsonld" : resolvedExtractor,
    scope: scope ?? (endpoint === "signal" ? "lede" : "full"),
    input_chars: endpoint === "signal" ? 1180 : 4920,
    word_count: 812,
    field_order: endpoint === "analyze" ? "sentiment-first" : null,
    cached: false,
    backend: "native-metal",
  };
}

export const bullishDecision: SignalResult = {
  sentiment: "bullish",
  abstained: false,
  policy: "probability",
  confidence: 0.82,
  probabilities: { bullish: 0.82, bearish: 0.05, neutral: 0.13 },
};

export const bullishOrdinal: SignalResult = {
  sentiment: "bullish",
  abstained: false,
  policy: "ordinal",
  score: 2,
};

export function timings(inference: Timings["inference"], totalMs: number): Timings {
  return {
    total_ms: totalMs,
    fetch_ms: 182,
    parse_ms: 6.4,
    extract_ms: 3.1,
    inference,
    cache: { fetch: "miss", result: "miss" },
  };
}

export function signalResponse(options: Omit<MetaOptions, "endpoint"> = {}): SignalResponse {
  return {
    meta: articleMeta({ endpoint: "signal", ...options }),
    signal: bullishDecision,
    timings: timings({ wall_ms: 341 }, 535),
  };
}

export function summaryResponse(options: Omit<MetaOptions, "endpoint"> = {}): SummaryResponse {
  return {
    meta: articleMeta({ endpoint: "summary", ...options }),
    summary: SUMMARY_TEXT,
    truncated: false,
    timings: timings(
      { wall_ms: 2410, ttft_ms: 512, load_ms: 12, prompt_eval_ms: 420, eval_ms: 1930, eval_tokens: 46 },
      2610,
    ),
  };
}

export function analyzeResponse(options: Omit<MetaOptions, "endpoint"> = {}): AnalyzeResponse {
  return {
    meta: articleMeta({ endpoint: "analyze", ...options }),
    signal: bullishOrdinal,
    entity: ENTITY,
    summary: SUMMARY_TEXT,
    truncated: false,
    timings: timings(
      { wall_ms: 2980, ttft_ms: 560, load_ms: 15, prompt_eval_ms: 470, eval_ms: 2410, eval_tokens: 61 },
      3190,
    ),
  };
}

export function metaEvent(meta: ArticleMeta): MetaEventData {
  return { meta, fetch_ms: 182, parse_ms: 6.4, extract_ms: 3.1, cache: { fetch: "miss", result: "miss" } };
}

export const sentimentEvent: SentimentEventData = { signal: bullishOrdinal, entity: ENTITY, elapsed_ms: 742 };
