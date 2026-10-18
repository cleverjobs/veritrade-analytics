import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SignalCard, leaning } from "@/components/SignalCard";
import type { SignalResult } from "@/lib/api";

const decision: SignalResult = {
  sentiment: "bearish",
  abstained: false,
  policy: "probability",
  confidence: 0.71,
  probabilities: { bullish: 0.09, bearish: 0.71, neutral: 0.2 },
};

describe("SignalCard", () => {
  it("renders the badge, confidence, probabilities, policy and timing", () => {
    render(<SignalCard status="done" signal={decision} model="nimble:9b-q4_K_M" timeToSignalMs={412.4} />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveTextContent("Bearish");
    expect(badge).toHaveAttribute("data-sentiment", "bearish");
    expect(screen.queryByText("Abstained")).not.toBeInTheDocument();
    expect(screen.getByText("Confidence").nextSibling).toHaveTextContent("71.0%");
    expect(screen.getByText("probability")).toBeInTheDocument();
    expect(screen.getByText("nimble:9b-q4_K_M")).toBeInTheDocument();
    expect(screen.getByText("Time to signal").nextSibling).toHaveTextContent("412 ms");

    const probs = within(screen.getByRole("list", { name: "Class probabilities" })).getAllByRole("listitem");
    expect(probs.map((li) => li.textContent)).toEqual(["Bullish9.0%", "Bearish71.0%", "Neutral20.0%"]);
  });

  it("explains an abstained call as below the confidence threshold", () => {
    const abstained: SignalResult = {
      sentiment: "neutral",
      abstained: true,
      policy: "probability",
      confidence: 0.44,
      probabilities: { bullish: 0.44, bearish: 0.2, neutral: 0.36 },
    };
    render(<SignalCard status="done" signal={abstained} />);
    expect(screen.getByTestId("signal-badge")).toHaveTextContent("Neutral");
    expect(screen.getByTestId("signal-badge")).toHaveAttribute("data-abstained", "true");
    expect(screen.getByText("Abstained")).toBeInTheDocument();
    expect(screen.getByText(/below confidence threshold/i)).toHaveTextContent("The model leaned Bullish.");
  });

  it("shows the ordinal score with its sign and the server timing", () => {
    const ordinal: SignalResult = { sentiment: "bullish", abstained: false, policy: "ordinal", score: 1 };
    render(
      <SignalCard status="streaming" signal={ordinal} entity="Acme" timeToSignalMs={800} serverMs={742} />,
    );
    expect(screen.getByText("Score").nextSibling).toHaveTextContent("+1 on -2..+2");
    expect(screen.getByText("Acme")).toBeInTheDocument();
    expect(screen.getByText("Time to signal").nextSibling).toHaveTextContent("800 ms (server 742 ms)");
  });

  it("shows waiting, failed and cancelled states without a signal", () => {
    const { rerender } = render(<SignalCard status="pending" signal={null} />);
    expect(screen.getByRole("region", { name: "Signal" })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText(/waiting for the signal/i)).toBeInTheDocument();
    rerender(<SignalCard status="error" signal={null} />);
    expect(screen.getByText(/request failed/i)).toBeInTheDocument();
    rerender(<SignalCard status="cancelled" signal={null} />);
    expect(screen.getByText("Cancelled.")).toBeInTheDocument();
  });
});

describe("leaning", () => {
  it("uses probabilities, then score sign, then the label", () => {
    expect(leaning(decision)).toBe("bearish");
    expect(leaning({ sentiment: "neutral", abstained: true, policy: "ordinal", score: -1 })).toBe("bearish");
    expect(leaning({ sentiment: "neutral", abstained: true, policy: "ordinal", score: 0 })).toBe("neutral");
  });
});
