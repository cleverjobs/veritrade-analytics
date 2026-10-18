// Shape of evals/results/benchmarks.json, written by `veritrade_evals report`.

export interface BenchmarkEnvironment {
  chip: string;
  ramGb: number;
  ollamaVersion: string;
  backend: string;
  gitSha: string | null;
}

export interface SignalBenchmark {
  endpoint: "signal" | "analyze";
  model: string;
  kind: "decision" | "generative";
  policy: string;
  scope: "lede" | "full";
  fieldOrder: string | null;
  promptVersion: string;
  dataset: string;
  split: string;
  n: number;
  f05Macro: number;
  precisionBullish: number;
  precisionBearish: number;
  recallBullish: number;
  recallBearish: number;
  inversionRate: number;
  coverage: number;
  ciLow: number | null;
  ciHigh: number | null;
}

export interface LatencyBenchmark {
  endpoint: string;
  model: string;
  scope: "lede" | "full";
  streaming: boolean;
  backend: string;
  n: number;
  p50Ms: number;
  p95Ms: number;
  ttftP50Ms: number | null;
  timeToSignalP50Ms: number | null;
}

export interface SummaryBenchmark {
  endpoint: "summary" | "analyze";
  model: string;
  n: number;
  deterministicPassRate: number;
  keyFactCoverage: number;
  judgeFaithfulness: number | null;
  judgeCoverage: number | null;
  hallucinationRate: number | null;
}

export interface ExtractionBenchmark {
  parser: string | null;
  extractor: string;
  n: number;
  p50Ms: number;
  p95Ms: number;
  precision: number;
  recall: number;
  f1: number;
  leakage: number;
  tokensToLlm: number;
  failureRate: number;
  installMb: number | null;
}

export type SelectionEndpoint = "signal" | "summary" | "analyze";

export interface Selection {
  model: string;
  reason: string;
}

export interface ArticleSetStats {
  n: number;
  perSource: Record<string, { bullish: number; bearish: number; neutral: number }>;
  agreement: number | null;
  kappa: number | null;
}

export interface BenchmarksV1 {
  version: 1;
  sample: boolean;
  generatedAt: string | null;
  environment: BenchmarkEnvironment;
  signal: SignalBenchmark[];
  latency: LatencyBenchmark[];
  summary: SummaryBenchmark[];
  extraction: ExtractionBenchmark[];
  selection: Record<SelectionEndpoint, Selection>;
  articles: ArticleSetStats;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Structural check at the top level only; the eval report owns row-level correctness. */
export function isBenchmarksV1(value: unknown): value is BenchmarksV1 {
  if (!isObject(value) || value.version !== 1 || typeof value.sample !== "boolean") return false;
  const arrays = ["signal", "latency", "summary", "extraction"] as const;
  return (
    isObject(value.environment) &&
    arrays.every((key) => Array.isArray(value[key])) &&
    isObject(value.selection) &&
    isObject(value.articles)
  );
}

/** Latency budgets settled in the design decisions (native, warm, excluding fetch). */
export const LATENCY_BUDGETS = {
  signalLedeP50Ms: 500,
  ttftMs: 1500,
  completeP50Ms: 4000,
} as const;

export function withinBudget(row: LatencyBenchmark): boolean | null {
  if (row.endpoint === "signal") {
    return row.scope === "lede" ? row.p50Ms <= LATENCY_BUDGETS.signalLedeP50Ms : null;
  }
  if (row.endpoint === "summary" || row.endpoint === "analyze") {
    const ttftOk = row.ttftP50Ms === null || row.ttftP50Ms <= LATENCY_BUDGETS.ttftMs;
    return ttftOk && row.p50Ms <= LATENCY_BUDGETS.completeP50Ms;
  }
  return null;
}
