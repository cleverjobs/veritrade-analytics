import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import BenchmarksPage from "@/app/benchmarks/page";
import { BenchmarksView } from "@/components/benchmarks/BenchmarksView";
import sample from "@/data/benchmarks.sample.json";
import { isBenchmarksV1, type BenchmarksV1 } from "@/lib/benchmarks";

// Pin the page to the committed sample even after real eval results are copied in.
vi.mock("@/data/benchmarks.json", async () => ({
  default: (await import("@/data/benchmarks.sample.json")).default,
}));

function sampleData(): BenchmarksV1 {
  if (!isBenchmarksV1(sample)) throw new Error("benchmarks.sample.json does not match BenchmarksV1");
  return sample;
}

describe("Benchmarks page", () => {
  it("renders the sample with a visible sample-data banner and every table", () => {
    render(<BenchmarksPage />);
    expect(screen.getByRole("heading", { level: 1, name: "Benchmarks" })).toBeInTheDocument();
    expect(screen.getByTestId("sample-banner")).toHaveTextContent("Sample data.");

    for (const name of [
      "Selected models",
      "Signal accuracy",
      "Latency",
      "Summary quality",
      "Extraction matrix",
      "Article set",
    ]) {
      expect(screen.getByRole("heading", { level: 2, name })).toBeInTheDocument();
    }

    const data = sampleData();
    const signal = screen.getByRole("table", { name: /Signal accuracy by model/ });
    expect(within(signal).getAllByRole("row")).toHaveLength(data.signal.length + 1);
    const latency = screen.getByRole("table", { name: /Latency by endpoint/ });
    expect(within(latency).getAllByRole("row")).toHaveLength(data.latency.length + 1);
    expect(within(latency).getAllByText("over").length).toBeGreaterThan(0);
    const extraction = screen.getByRole("table", { name: /Extraction speed and quality/ });
    expect(within(extraction).getAllByRole("row")).toHaveLength(data.extraction.length + 1);
    const articles = screen.getByRole("table", { name: /Articles per source/ });
    expect(within(articles).getByText("Yahoo Finance")).toBeInTheDocument();

    const picks = screen.getByRole("heading", { name: "Selected models" }).parentElement as HTMLElement;
    expect(within(picks).getByText("/api/signal")).toBeInTheDocument();
    expect(within(picks).getByText(data.selection.signal.reason)).toBeInTheDocument();
  });

  it("hides the banner for measured results", () => {
    render(<BenchmarksView data={{ ...sampleData(), sample: false, generatedAt: "2026-10-20T10:00:00Z" }} />);
    expect(screen.queryByTestId("sample-banner")).not.toBeInTheDocument();
    expect(screen.getByText(/generated 2026-10-20T10:00:00Z/)).toBeInTheDocument();
  });
});
