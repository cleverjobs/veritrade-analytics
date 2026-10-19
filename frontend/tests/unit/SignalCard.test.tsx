import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SignalCard, abstainReason, leaning, pctDigits, thresholdText } from "@/components/SignalCard";
import type { SignalResult } from "@/lib/api";

const decision: SignalResult = {
  sentiment: "bearish",
  abstained: false,
  policy: "probability",
  confidence: 0.71,
  probabilities: { bullish: 0.09, bearish: 0.71, neutral: 0.2 },
  threshold: 0.5,
  directional_tau: null,
};

// clef-flash's frozen 0.80 rule with Bullish leading below it.
const abstained: SignalResult = {
  sentiment: "neutral",
  abstained: true,
  policy: "probability",
  confidence: 0.73,
  probabilities: { bullish: 0.73, bearish: 0.06, neutral: 0.21 },
  threshold: 0.8,
  directional_tau: null,
};

// qwen3.5:0.8b's label-logprob rule: no floor, neutral leads but bullish clears tau.
const tauCall: SignalResult = {
  sentiment: "bullish",
  abstained: false,
  policy: "label-logprob",
  confidence: 0.31,
  probabilities: { bullish: 0.31, bearish: 0.09, neutral: 0.6 },
  threshold: 0,
  directional_tau: 0.28,
};

describe("SignalCard", () => {
  it("renders the badge, confidence, probabilities, policy, threshold and timing", () => {
    render(<SignalCard status="done" signal={decision} model="nimble:9b-q4_K_M" timeToSignalMs={412.4} />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveTextContent("Bearish");
    expect(badge).toHaveAttribute("data-sentiment", "bearish");
    expect(badge).not.toHaveAttribute("data-abstained");
    expect(screen.queryByTestId("abstain-note")).not.toBeInTheDocument();
    expect(screen.getByText("Confidence").nextSibling).toHaveTextContent("71.0%");
    expect(screen.getByText("probability")).toBeInTheDocument();
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent("call needs ≥ 50.0%");
    expect(screen.getByText("nimble:9b-q4_K_M")).toBeInTheDocument();
    expect(screen.getByText("Time to signal").nextSibling).toHaveTextContent("412 ms");
    expect(screen.queryByTestId("signal-cold-start")).not.toBeInTheDocument();

    const probs = within(screen.getByRole("list", { name: "Class probabilities" })).getAllByRole("listitem");
    expect(probs.map((li) => li.textContent)).toEqual(["Bullish9.0%", "Bearish71.0%", "Neutral20.0%"]);
    for (const s of ["bullish", "bearish", "neutral"]) {
      expect(screen.getByTestId(`mark-threshold-${s}`)).toHaveStyle({ left: "50%" });
    }
    expect(screen.queryByTestId("mark-tau-bullish")).not.toBeInTheDocument();
  });

  it("shows an abstention as No signal and explains it with the numbers", () => {
    render(<SignalCard status="done" signal={abstained} model="clef-flash:9b-q8_0" />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveTextContent("No signal");
    expect(badge).toHaveAttribute("data-sentiment", "neutral");
    expect(badge).toHaveAttribute("data-abstained", "true");
    expect(screen.queryByText("Abstained")).not.toBeInTheDocument();
    expect(screen.getByTestId("abstain-note")).toHaveTextContent(
      "Abstained: the top class (Bullish, 73.0%) is below this model's 80.0% threshold, so there is no directional call.",
    );
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent("call needs ≥ 80.0%");
    expect(screen.getByTestId("mark-threshold-bullish")).toHaveStyle({ left: "80%" });
  });

  it("adds decimals when a near miss would print as equal to the threshold", () => {
    // nimble's 0.97 rule: 0.9696 rounds to 97.0% at one decimal.
    const nearMiss: SignalResult = {
      sentiment: "neutral",
      abstained: true,
      policy: "probability",
      confidence: 0.9696,
      probabilities: { bullish: 0.9696, bearish: 0.01, neutral: 0.0204 },
      threshold: 0.97,
      directional_tau: null,
    };
    render(<SignalCard status="done" signal={nearMiss} />);
    expect(screen.getByTestId("abstain-note")).toHaveTextContent(
      "Abstained: the top class (Bullish, 96.96%) is below this model's 97.00% threshold",
    );
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent("call needs ≥ 97.00%");
    expect(screen.getByText("Confidence").nextSibling).toHaveTextContent("96.96%");
    const probs = within(screen.getByRole("list", { name: "Class probabilities" })).getAllByRole("listitem");
    expect(probs.map((li) => li.textContent)).toEqual(["Bullish96.96%", "Bearish1.00%", "Neutral2.04%"]);
  });

  it("explains an ordinal abstention with the score", () => {
    const ordinal: SignalResult = {
      sentiment: "neutral",
      abstained: true,
      policy: "ordinal",
      score: 1,
      threshold: 2,
      directional_tau: null,
    };
    render(<SignalCard status="done" signal={ordinal} />);
    expect(screen.getByTestId("signal-badge")).toHaveTextContent("No signal");
    expect(screen.getByTestId("abstain-note")).toHaveTextContent(
      "Abstained: |score| 1 (leaning Bullish) is below this model's threshold of 2, so there is no directional call.",
    );
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent("|score| ≥ 2");
    expect(screen.queryByRole("list", { name: "Class probabilities" })).not.toBeInTheDocument();
  });

  it("marks tau on the directional bars and skips a zero threshold", () => {
    render(<SignalCard status="done" signal={tauCall} />);
    expect(screen.getByTestId("signal-badge")).toHaveTextContent("Bullish");
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent(
      "top class wins; bullish/bearish called at ≥ 28.0% even when neutral leads",
    );
    expect(screen.getByTestId("mark-tau-bullish")).toHaveStyle({ left: "28%" });
    expect(screen.getByTestId("mark-tau-bearish")).toHaveStyle({ left: "28%" });
    expect(screen.queryByTestId("mark-tau-neutral")).not.toBeInTheDocument();
    expect(screen.queryByTestId("mark-threshold-bullish")).not.toBeInTheDocument();
  });

  it("omits the threshold stat for a payload without one", () => {
    const legacy: SignalResult = { ...abstained, threshold: undefined, directional_tau: undefined };
    render(<SignalCard status="done" signal={legacy} />);
    expect(screen.queryByText("Threshold")).not.toBeInTheDocument();
    expect(screen.queryByTestId("mark-threshold-bullish")).not.toBeInTheDocument();
    expect(screen.getByTestId("abstain-note")).toHaveTextContent(
      "Abstained: the call is below this model's confidence threshold (it leaned Bullish), so there is no directional call.",
    );
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

  it("flags a cold start next to the time to signal", () => {
    const { rerender } = render(
      <SignalCard status="done" signal={abstained} timeToSignalMs={12_300} coldStart />,
    );
    expect(screen.getByTestId("signal-cold-start")).toHaveTextContent(
      "Cold start includes loading the model into memory",
    );
    rerender(<SignalCard status="done" signal={tauCall} timeToSignalMs={9_100} coldStart loadMs={8_400} />);
    expect(screen.getByTestId("signal-cold-start")).toHaveTextContent(
      "Cold start model loaded for this request (8.40 s)",
    );
    rerender(<SignalCard status="done" signal={tauCall} timeToSignalMs={410} coldStart={false} />);
    expect(screen.queryByTestId("signal-cold-start")).not.toBeInTheDocument();
  });

  it("shows waiting, loading, failed and cancelled states without a signal", () => {
    const { rerender } = render(<SignalCard status="pending" signal={null} />);
    expect(screen.getByRole("region", { name: "Signal" })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Waiting for the signal...")).toBeInTheDocument();
    rerender(<SignalCard status="streaming" signal={null} coldStart />);
    expect(
      screen.getByText("Waiting for the signal while the model loads into memory (cold start)..."),
    ).toBeInTheDocument();
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

describe("thresholdText", () => {
  // The sign is glued to its number so a wrapped stat never splits them.
  const ge = "≥ ";

  it("words each frozen rule", () => {
    expect(thresholdText(decision)).toBe(`call needs ${ge}50.0%`);
    expect(thresholdText({ ...decision, threshold: 0 })).toBe("none (top class wins)");
    expect(thresholdText({ ...decision, threshold: 0.5, directional_tau: 0.3 })).toBe(
      `call needs ${ge}50.0%; bullish/bearish called at ${ge}30.0% even when neutral leads`,
    );
    expect(thresholdText({ sentiment: "bullish", abstained: false, policy: "ordinal", threshold: 1 })).toBe(
      `|score| ${ge}1`,
    );
    expect(thresholdText({ ...decision, threshold: null })).toBeNull();
  });
});

describe("pctDigits", () => {
  it("stays at one decimal unless a missed threshold or tau would print as met", () => {
    expect(pctDigits(decision)).toBe(1);
    expect(pctDigits(abstained)).toBe(1);
    expect(pctDigits(tauCall)).toBe(1);
    // A served neutral whose bullish side missed tau 0.28 by a hair.
    const tauMiss: SignalResult = {
      ...tauCall,
      sentiment: "neutral",
      confidence: 0.6204,
      probabilities: { bullish: 0.2796, bearish: 0.1, neutral: 0.6204 },
    };
    expect(pctDigits(tauMiss)).toBe(2);
    expect(thresholdText(tauMiss)).toMatch(/called at ≥\s28\.00% even when neutral leads$/);
  });
});

describe("abstainReason", () => {
  it("names tau when neither directional class reached it under a neutral lead", () => {
    const neutralLead: SignalResult = {
      sentiment: "neutral",
      abstained: true,
      policy: "probability",
      confidence: 0.46,
      probabilities: { bullish: 0.27, bearish: 0.27, neutral: 0.46 },
      threshold: 0.5,
      directional_tau: 0.28,
    };
    expect(abstainReason(neutralLead)).toBe(
      "the top class (Neutral, 46.0%) is below this model's 50.0% threshold, and neither Bullish nor Bearish reached 28.0%",
    );
  });
});
