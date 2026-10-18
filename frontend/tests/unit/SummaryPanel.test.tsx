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

  it("shows no note for a complete summary or when there is no text", () => {
    const { rerender } = render(<SummaryPanel status="done" text={TEXT} truncated={false} streaming />);
    expect(screen.queryByTestId("summary-truncated")).not.toBeInTheDocument();
    rerender(<SummaryPanel status="done" text="" truncated streaming />);
    expect(screen.queryByTestId("summary-truncated")).not.toBeInTheDocument();
    expect(screen.getByText("The model returned an empty summary.")).toBeInTheDocument();
  });
});
