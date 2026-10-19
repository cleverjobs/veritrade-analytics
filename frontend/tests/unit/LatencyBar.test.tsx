import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LatencyBar, niceStep, stageSegments, type LatencyLane } from "@/components/LatencyBar";
import type { Timings } from "@/lib/api";

const base: Omit<Timings, "inference" | "total_ms"> = {
  fetch_ms: 200,
  parse_ms: 10,
  extract_ms: 5,
  cache: { fetch: "miss", result: "miss" },
};

const generative: Timings = {
  ...base,
  total_ms: 1900,
  inference: {
    wall_ms: 1680,
    ttft_ms: 400,
    load_ms: 20,
    prompt_eval_ms: 360,
    eval_ms: 1280,
    cold_start: true,
  },
};

const decision: Timings = { ...base, total_ms: 560, inference: { wall_ms: 340, cold_start: false } };

describe("stageSegments", () => {
  it("lays stages end to end and fills the client remainder with other", () => {
    const segments = stageSegments(generative, 2000);
    expect(segments.map((s) => [s.key, s.startMs, s.ms])).toEqual([
      ["fetch", 0, 200],
      ["parse", 200, 10],
      ["extract", 210, 5],
      ["load", 215, 20],
      ["prompt", 235, 360],
      ["generation", 595, 1280],
      ["other", 1875, 125],
    ]);
  });

  it("uses one inference segment when only wall time is known", () => {
    const keys = stageSegments(decision, null).map((s) => s.key);
    expect(keys).toEqual(["fetch", "parse", "extract", "inference", "other"]);
  });

  it("skips zero stages and never emits a negative remainder", () => {
    const cached: Timings = { ...decision, fetch_ms: 0, total_ms: 300 };
    expect(stageSegments(cached, 100).map((s) => s.key)).toEqual(["parse", "extract", "inference"]);
  });
});

describe("niceStep", () => {
  it("rounds to 1, 2, 2.5 or 5 times a power of ten", () => {
    expect(niceStep(430)).toBe(500);
    expect(niceStep(180)).toBe(200);
    expect(niceStep(2400)).toBe(2500);
    expect(niceStep(7)).toBe(10);
  });
});

describe("LatencyBar", () => {
  const lanes: LatencyLane[] = [
    {
      id: "signal",
      label: "Signal",
      status: "done",
      timings: decision,
      endMs: 600,
      marks: [{ kind: "signal", label: "time to signal", atMs: 600 }],
    },
    {
      id: "summary",
      label: "Summary",
      status: "done",
      timings: generative,
      endMs: 2000,
      marks: [{ kind: "token", label: "time to first token", atMs: 650 }],
    },
  ];

  it("draws two lanes on one shared client time axis", () => {
    render(<LatencyBar lanes={lanes} />);
    expect(screen.getByTestId("lane-signal")).toBeInTheDocument();
    expect(screen.getByTestId("lane-summary")).toBeInTheDocument();
    // Axis spans the slowest lane (2000 ms), so the 600 ms signal lane ends at 30%.
    expect(screen.getByTestId("segment-signal-fetch")).toHaveStyle({ left: "0%", width: "10%" });
    expect(screen.getByTestId("mark-signal-signal")).toHaveStyle({ left: "30%" });
    expect(screen.getByTestId("segment-summary-generation")).toHaveStyle({ left: "29.75%", width: "64%" });
    expect(screen.getByTestId("mark-summary-token")).toHaveStyle({ left: "32.5%" });
  });

  it("lists every value as text for screen readers", () => {
    render(<LatencyBar lanes={lanes} />);
    const summary = within(screen.getByRole("heading", { name: "Summary" }).parentElement as HTMLElement);
    expect(summary.getByText("time to first token").nextSibling).toHaveTextContent("650 ms");
    expect(summary.getByText("TTFT (server)").nextSibling).toHaveTextContent("400 ms");
    expect(summary.getByText("prompt eval").nextSibling).toHaveTextContent("360 ms");
    expect(summary.getByText("generation").nextSibling).toHaveTextContent("1.28 s");
    expect(summary.getByText("complete (client)").nextSibling).toHaveTextContent("2.00 s");
    expect(summary.getByText("cold start").nextSibling).toHaveTextContent("yes, model loaded");
    const signal = within(screen.getByRole("heading", { name: "Signal" }).parentElement as HTMLElement);
    expect(signal.getByText("inference").nextSibling).toHaveTextContent("340 ms");
    expect(signal.getByText("cache").nextSibling).toHaveTextContent("fetch miss, result miss");
    expect(signal.getByText("cold start").nextSibling).toHaveTextContent("no, already loaded");
  });

  it("leaves out the cold start row when it is unknown", () => {
    const unknown: Timings = { ...decision, inference: { wall_ms: 340, cold_start: null } };
    render(<LatencyBar lanes={[{ ...lanes[0], timings: unknown }]} />);
    expect(screen.queryByText("cold start")).not.toBeInTheDocument();
  });

  it("shows a running lane before timings arrive", () => {
    render(
      <LatencyBar
        lanes={[
          {
            id: "analyze",
            label: "Analyze",
            status: "streaming",
            timings: null,
            endMs: null,
            marks: [{ kind: "signal", label: "time to signal", atMs: 700 }],
          },
        ]}
      />,
    );
    expect(screen.getByText("running...")).toBeInTheDocument();
    expect(screen.getByTestId("mark-analyze-signal")).toHaveStyle({ left: "100%" });
  });
});
