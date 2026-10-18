import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api";
import { articleMeta, distinctErrors, initialRunState, runReducer, type RunAction } from "@/lib/run-state";
import { articleMeta as metaFixture, bullishDecision, summaryResponse } from "@/mocks/fixtures";

const start: RunAction = {
  type: "start",
  runId: 1,
  mode: "split",
  streaming: true,
  lanes: ["signal", "summary"],
};

function play(actions: RunAction[]) {
  return actions.reduce(runReducer, initialRunState);
}

describe("runReducer", () => {
  it("tracks deltas, first token and the authoritative final summary", () => {
    const done = summaryResponse();
    const state = play([
      start,
      { type: "meta", runId: 1, lane: "summary", atMs: 50, meta: done.meta },
      { type: "delta", runId: 1, lane: "summary", atMs: 300, text: "Acme " },
      { type: "delta", runId: 1, lane: "summary", atMs: 340, text: "beat" },
      {
        type: "done",
        runId: 1,
        lane: "summary",
        atMs: 900,
        meta: done.meta,
        timings: done.timings,
        summary: "Final.",
      },
    ]);
    const lane = state.lanes.find((l) => l.id === "summary");
    expect(lane).toMatchObject({ status: "done", firstTokenAtMs: 300, endAtMs: 900 });
    expect(state.summary).toBe("Final.");
    expect(state.summaryTruncated).toBe(false);
    expect(state.running).toBe(true);
    expect(articleMeta(state)?.title).toBe(done.meta.title);
  });

  it("keeps the truncation flag from the final summary until the next run", () => {
    const done = summaryResponse();
    const truncated = play([
      start,
      {
        type: "done",
        runId: 1,
        lane: "summary",
        atMs: 900,
        meta: done.meta,
        timings: done.timings,
        summary: "Cut.",
        truncated: true,
      },
      { type: "done", runId: 1, lane: "signal", atMs: 950, meta: done.meta, timings: done.timings },
    ]);
    expect(truncated.summaryTruncated).toBe(true);
    expect(runReducer(truncated, { ...start, runId: 2 }).summaryTruncated).toBe(false);
  });

  it("records time to signal from a JSON signal response and finishes the run", () => {
    const state = play([
      start,
      {
        type: "done",
        runId: 1,
        lane: "signal",
        atMs: 420,
        meta: metaFixture({ endpoint: "signal" }),
        timings: summaryResponse().timings,
        signal: bullishDecision,
      },
      {
        type: "error",
        runId: 1,
        lane: "summary",
        atMs: 500,
        error: new ApiError({ code: "llm_unreachable", message: "x" }),
      },
    ]);
    expect(state.signal).toEqual(bullishDecision);
    expect(state.lanes.find((l) => l.id === "signal")?.signalAtMs).toBe(420);
    expect(state.running).toBe(false);
  });

  it("ignores events from a superseded run", () => {
    const state = play([
      start,
      { ...start, runId: 2 },
      { type: "delta", runId: 1, lane: "summary", atMs: 10, text: "stale" },
    ]);
    expect(state.summary).toBe("");
    expect(state.runId).toBe(2);
  });

  it("marks active lanes cancelled", () => {
    const state = play([start, { type: "cancel", runId: 1, atMs: 120 }]);
    expect(state.lanes.map((l) => l.status)).toEqual(["cancelled", "cancelled"]);
    expect(state.running).toBe(false);
  });

  it("merges identical failures across lanes", () => {
    const err = () => new ApiError({ code: "paywalled", message: "Paywall" });
    const state = play([
      start,
      { type: "error", runId: 1, lane: "signal", atMs: 1, error: err() },
      { type: "error", runId: 1, lane: "summary", atMs: 2, error: err() },
    ]);
    expect(distinctErrors(state).map((e) => e.lanes)).toEqual([["signal", "summary"]]);
  });
});
