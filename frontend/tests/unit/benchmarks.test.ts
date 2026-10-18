import { describe, expect, it } from "vitest";

import sample from "@/data/benchmarks.sample.json";
import { isBenchmarksV1, withinBudget, type LatencyBenchmark } from "@/lib/benchmarks";

const row = (overrides: Partial<LatencyBenchmark>): LatencyBenchmark => ({
  endpoint: "signal",
  model: "m",
  scope: "lede",
  streaming: false,
  backend: "native-metal",
  n: 30,
  p50Ms: 400,
  p95Ms: 600,
  ttftP50Ms: null,
  timeToSignalP50Ms: null,
  ...overrides,
});

describe("benchmarks", () => {
  it("accepts the committed sample and flags it as sample data", () => {
    expect(isBenchmarksV1(sample)).toBe(true);
    expect(sample.sample).toBe(true);
  });

  it("rejects other shapes", () => {
    expect(isBenchmarksV1({ generatedAt: null })).toBe(false);
    expect(isBenchmarksV1({ ...sample, version: 2 })).toBe(false);
    expect(isBenchmarksV1({ ...sample, signal: null })).toBe(false);
  });

  it("applies the latency budgets per endpoint", () => {
    expect(withinBudget(row({ p50Ms: 480 }))).toBe(true);
    expect(withinBudget(row({ p50Ms: 640 }))).toBe(false);
    expect(withinBudget(row({ scope: "full", p50Ms: 900 }))).toBeNull();
    expect(withinBudget(row({ endpoint: "summary", ttftP50Ms: 600, p50Ms: 3400 }))).toBe(true);
    expect(withinBudget(row({ endpoint: "analyze", ttftP50Ms: 2600, p50Ms: 3400 }))).toBe(false);
    expect(withinBudget(row({ endpoint: "analyze", ttftP50Ms: 600, p50Ms: 14800 }))).toBe(false);
  });
});
