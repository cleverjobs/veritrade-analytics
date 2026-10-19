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
  SignalRule,
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

// Frozen rules as backend/src/veritrade/data/models.toml sets them.
const probabilityRule = (threshold: number): SignalRule => ({
  policy: "probability",
  threshold,
  directional_tau: null,
});
const ORDINAL_RULE: SignalRule = { policy: "ordinal", threshold: 1, directional_tau: null };

export const models: ModelsResponse = {
  models: [
    {
      name: "clef-flash:9b-q8_0",
      kind: "decision",
      endpoints: ["signal"],
      installed: true,
      evaluated: true,
      signal_rules: { signal: probabilityRule(0.8) },
    },
    {
      name: "nimble:9b-q4_K_M",
      kind: "decision",
      endpoints: ["signal"],
      installed: true,
      evaluated: true,
      signal_rules: { signal: probabilityRule(0.97) },
    },
    {
      name: "tev1:0.8b-q8_0",
      kind: "decision",
      endpoints: ["signal"],
      installed: true,
      evaluated: false,
      signal_rules: { signal: probabilityRule(0.5) },
    },
    {
      name: "qwen3.5:0.8b-q8_0",
      kind: "generative",
      endpoints: ["signal", "summary", "analyze"],
      installed: true,
      evaluated: true,
      signal_rules: {
        signal: { policy: "label-logprob", threshold: 0, directional_tau: 0.28 },
        analyze: ORDINAL_RULE,
      },
    },
    {
      name: "qwen3.5:2b-q4_K_M",
      kind: "generative",
      endpoints: ["summary"],
      installed: true,
      evaluated: true,
      signal_rules: {},
    },
    {
      name: "qwen3.5:4b-q4_K_M",
      kind: "generative",
      endpoints: ["summary", "analyze"],
      installed: true,
      evaluated: true,
      signal_rules: { analyze: ORDINAL_RULE },
    },
    {
      name: "qwen3:4b-instruct-2507-q4_K_M",
      kind: "generative",
      endpoints: ["analyze"],
      installed: true,
      evaluated: true,
      signal_rules: { analyze: ORDINAL_RULE },
    },
    {
      name: "ministral-3:3b",
      kind: "generative",
      endpoints: ["analyze"],
      installed: true,
      evaluated: true,
      signal_rules: { analyze: ORDINAL_RULE },
    },
    {
      name: "qwen3.5:9b-q4_K_M",
      kind: "generative",
      endpoints: ["analyze"],
      installed: false,
      evaluated: true,
      signal_rules: { analyze: ORDINAL_RULE },
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

// Clears every frozen signal rule above, so any picked model calls it Bullish.
export const bullishDecision: SignalResult = {
  sentiment: "bullish",
  abstained: false,
  policy: "probability",
  confidence: 0.984,
  probabilities: { bullish: 0.984, bearish: 0.004, neutral: 0.012 },
  threshold: 0.97,
  directional_tau: null,
};

// Bullish leads but stays under clef-flash's 0.80 rule, so the call abstains.
export const abstainedDecision: SignalResult = {
  sentiment: "neutral",
  abstained: true,
  policy: "probability",
  confidence: 0.73,
  probabilities: { bullish: 0.73, bearish: 0.06, neutral: 0.21 },
  threshold: 0.8,
  directional_tau: null,
};

export const bullishOrdinal: SignalResult = {
  sentiment: "bullish",
  abstained: false,
  policy: "ordinal",
  score: 2,
  threshold: 1,
  directional_tau: null,
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
  const meta = articleMeta({ endpoint: "signal", ...options });
  const rule = models.models.find((m) => m.name === meta.model)?.signal_rules?.signal;
  return {
    meta,
    signal: rule
      ? {
          ...bullishDecision,
          policy: rule.policy,
          threshold: rule.threshold,
          directional_tau: rule.directional_tau ?? null,
        }
      : bullishDecision,
    timings: timings({ wall_ms: 341, cold_start: false }, 535),
  };
}

export function summaryResponse(options: Omit<MetaOptions, "endpoint"> = {}): SummaryResponse {
  return {
    meta: articleMeta({ endpoint: "summary", ...options }),
    summary: SUMMARY_TEXT,
    truncated: false,
    timings: timings(
      {
        wall_ms: 2410,
        ttft_ms: 512,
        load_ms: 12,
        prompt_eval_ms: 420,
        eval_ms: 1930,
        eval_tokens: 46,
        cold_start: false,
      },
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
      {
        wall_ms: 2980,
        ttft_ms: 560,
        load_ms: 15,
        prompt_eval_ms: 470,
        eval_ms: 2410,
        eval_tokens: 61,
        cold_start: false,
      },
      3190,
    ),
  };
}

export function metaEvent(meta: ArticleMeta, coldStart = false): MetaEventData {
  return {
    meta,
    fetch_ms: 182,
    parse_ms: 6.4,
    extract_ms: 3.1,
    cache: { fetch: "miss", result: "miss" },
    cold_start: coldStart,
  };
}

export const sentimentEvent: SentimentEventData = { signal: bullishOrdinal, entity: ENTITY, elapsed_ms: 742 };
