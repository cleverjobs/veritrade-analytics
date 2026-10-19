import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SummaryPanel } from "@/components/SummaryPanel";

const TEXT = "Acme Corp cut its full-year guidance.";

describe("SummaryPanel", () => {
  it("notes a summary the server cut at the model's output limit", () => {
    render(<SummaryPanel status="done" text={TEXT} truncated streaming model="qwen3.5:0.8b-q8_0" />);
    expect(screen.getByTestId("summary-text")).toHaveTextContent(TEXT);
    expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete");
    expect(screen.getByTestId("summary-truncated")).toHaveTextContent(/output limit.*cut short/);
  });

  it("says the model is loading on a cold start, then reports the load time", () => {
    const { rerender } = render(<SummaryPanel status="streaming" text="" streaming coldStart />);
    expect(screen.getByTestId("summary-status")).toHaveTextContent(
      "Loading the model into memory (cold start)...",
    );
    expect(screen.queryByTestId("summary-cold-start")).not.toBeInTheDocument();

    rerender(<SummaryPanel status="streaming" text={TEXT} streaming coldStart />);
    expect(screen.getByTestId("summary-status")).toHaveTextContent("Streaming...");
    expect(screen.getByTestId("summary-cold-start")).toHaveTextContent(
      "cold start: includes loading the model into memory",
    );

    rerender(
      <SummaryPanel
        status="done"
        text={TEXT}
        streaming
        model="qwen3.5:4b-q4_K_M"
        firstTokenMs={9_600}
        coldStart
        loadMs={8_400}
      />,
    );
    expect(screen.getByTestId("summary-cold-start").parentElement).toHaveTextContent(
      "Model qwen3.5:4b-q4_K_M · first token 9.60 s · cold start: model loaded for this request (8.40 s)",
    );
  });

  it("stops saying the model is loading once the same generation returned the signal", () => {
    render(<SummaryPanel status="streaming" text="" streaming coldStart modelAnswered />);
    expect(screen.getByTestId("summary-status")).toHaveTextContent("Waiting for the first token...");
    expect(screen.getByTestId("summary-cold-start")).toHaveTextContent(
      "cold start: includes loading the model into memory",
    );
  });

  it("keeps the usual wait message when the model is already loaded", () => {
    render(<SummaryPanel status="streaming" text="" streaming coldStart={false} />);
    expect(screen.getByTestId("summary-status")).toHaveTextContent("Waiting for the first token...");
    expect(screen.queryByTestId("summary-cold-start")).not.toBeInTheDocument();
  });

  it("shows no note for a complete summary or when there is no text", () => {
    const { rerender } = render(<SummaryPanel status="done" text={TEXT} truncated={false} streaming />);
    expect(screen.queryByTestId("summary-truncated")).not.toBeInTheDocument();
    rerender(<SummaryPanel status="done" text="" truncated streaming />);
    expect(screen.queryByTestId("summary-truncated")).not.toBeInTheDocument();
    expect(screen.getByText("The model returned an empty summary.")).toBeInTheDocument();
  });
});
