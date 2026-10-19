import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { AnalyzeApp } from "@/components/AnalyzeApp";
import type { AnalyzeBody, SignalBody, SummaryBody } from "@/lib/api";
import {
  ENTITY,
  FIXTURE_URL,
  SUMMARY_CHUNKS,
  SUMMARY_TEXT,
  abstainedDecision,
  analyzeResponse,
  metaEvent,
  signalResponse,
  timings,
} from "@/mocks/fixtures";
import { API, analyzeStreamChunks, errorResponse, summaryStreamChunks } from "@/mocks/handlers";
import { gate, sseFrame, sseResponse } from "@/mocks/sse";

import { setupMswServer } from "./msw";

const server = setupMswServer();

async function submit(user: UserEvent, url = FIXTURE_URL) {
  const input = screen.getByLabelText("Article URL");
  await user.clear(input);
  await user.click(input);
  await user.paste(url);
  await user.click(screen.getByRole("button", { name: /^(Analyze|Restart)$/ }));
}

const optionLabels = (select: HTMLElement) =>
  within(select)
    .getAllByRole("option")
    .map((o) => o.textContent);

const tick = () => new Promise((resolve) => setTimeout(resolve, 20));

describe("Analyze page", () => {
  it("populates the pickers from /api/models and sends the chosen options", async () => {
    const bodies: { signal?: SignalBody; summary?: SummaryBody } = {};
    server.use(
      http.post(`${API}/signal`, async ({ request }) => {
        bodies.signal = (await request.json()) as SignalBody;
        return HttpResponse.json(signalResponse(bodies.signal));
      }),
      http.post(`${API}/summary/stream`, async ({ request }) => {
        bodies.summary = (await request.json()) as SummaryBody;
        return sseResponse(summaryStreamChunks(bodies.summary));
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);

    const signalSelect = screen.getByLabelText("Signal model");
    await waitFor(() =>
      expect(optionLabels(signalSelect)).toEqual([
        "Server default (nimble:9b-q4_K_M, p ≥ 0.97)",
        "clef-flash:9b-q8_0 (p ≥ 0.80)",
        "nimble:9b-q4_K_M (p ≥ 0.97) (default)",
        "tev1:0.8b-q8_0 (p ≥ 0.50) (CI plumbing, unevaluated)",
        "qwen3.5:0.8b-q8_0 (tau 0.28)",
      ]),
    );
    expect(signalSelect).toHaveAccessibleDescription(
      "In brackets, each model's frozen rule for a directional call. p ≥ x means the top class needs " +
        "probability x or more; tau x means bullish or bearish is called at x even when neutral leads.",
    );
    expect(screen.getByTestId("signal-model-rule")).toHaveTextContent("Rule for nimble:9b-q4_K_M: p ≥ 0.97");
    expect(screen.queryByTestId("summary-model-rule")).not.toBeInTheDocument();
    const summarySelect = screen.getByLabelText("Summary model");
    expect(optionLabels(summarySelect)).toEqual([
      "Server default (qwen3.5:4b-q4_K_M)",
      "qwen3.5:0.8b-q8_0",
      "qwen3.5:2b-q4_K_M",
      "qwen3.5:4b-q4_K_M (default)",
    ]);
    expect(summarySelect).not.toHaveAttribute("aria-describedby");

    await user.click(screen.getByRole("radio", { name: /Combined/ }));
    const analyzeSelect = screen.getByLabelText("Analyze model");
    expect(optionLabels(analyzeSelect)).toContain("Server default (qwen3.5:4b-q4_K_M, |score| ≥ 1)");
    expect(optionLabels(analyzeSelect)).toContain("ministral-3:3b (|score| ≥ 1)");
    expect(analyzeSelect).toHaveAccessibleDescription(
      /\|score\| ≥ n means the -2\.\.\+2 score needs magnitude n/,
    );
    expect(screen.getByTestId("analyze-model-rule")).toHaveTextContent(
      "Rule for qwen3.5:4b-q4_K_M: |score| ≥ 1",
    );
    expect(within(analyzeSelect).getByRole("option", { name: /qwen3.5:9b-q4_K_M/ })).toBeDisabled();
    expect(screen.queryByLabelText("Signal scope")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /Split/ }));
    await user.selectOptions(screen.getByLabelText("Signal model"), "clef-flash:9b-q8_0");
    expect(screen.getByTestId("signal-model-rule")).toHaveTextContent(
      "Rule for clef-flash:9b-q8_0: p ≥ 0.80",
    );
    await user.selectOptions(screen.getByLabelText("Signal scope"), "full");
    await user.selectOptions(screen.getByLabelText("Extractor"), "trafilatura");
    expect(screen.getByLabelText("Parser")).toBeDisabled();
    await submit(user);

    await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
    expect(bodies.signal).toEqual({
      url: FIXTURE_URL,
      parser: "selectolax",
      extractor: "trafilatura",
      scope: "full",
      model: "clef-flash:9b-q8_0",
    });
    expect(bodies.summary).toEqual({ url: FIXTURE_URL, parser: "selectolax", extractor: "trafilatura" });
    expect(screen.getByText("clef-flash:9b-q8_0", { selector: "dd code" })).toBeInTheDocument();
  });

  it("split mode shows the signal before the summary stream completes", async () => {
    const hold = gate();
    server.use(
      http.post(`${API}/summary/stream`, async ({ request }) => {
        const [meta, ping, first, ...rest] = summaryStreamChunks((await request.json()) as SummaryBody);
        return sseResponse(
          (async function* () {
            yield meta;
            yield ping;
            yield first;
            await hold.promise;
            yield* rest;
          })(),
        );
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    try {
      await submit(user);

      expect(await screen.findByTestId("signal-badge")).toHaveTextContent("Bullish");
      await waitFor(() =>
        expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY_CHUNKS[0].trim()),
      );
      expect(screen.getByTestId("summary-status")).toHaveTextContent("Streaming...");
      expect(screen.getByTestId("summary-text")).not.toHaveTextContent(SUMMARY_CHUNKS[2]);
      expect(screen.getByText("Acme Corp beats estimates and lifts full-year outlook")).toBeInTheDocument();

      hold.open();
      await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
      expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY_TEXT);

      const latency = within(screen.getByTestId("latency"));
      expect(latency.getByTestId("lane-signal")).toBeInTheDocument();
      expect(latency.getByTestId("lane-summary")).toBeInTheDocument();
      expect(latency.getByTestId("mark-signal-signal")).toBeInTheDocument();
      expect(latency.getByTestId("mark-summary-token")).toBeInTheDocument();
      expect(latency.getByTestId("segment-signal-inference")).toBeInTheDocument();
      expect(latency.getByTestId("segment-summary-generation")).toBeInTheDocument();
    } finally {
      hold.open();
    }
  });

  it("combined streaming renders sentiment, then summary deltas, then timings", async () => {
    const afterSentiment = gate();
    const afterDeltas = gate();
    server.use(
      http.post(`${API}/analyze/stream`, async ({ request }) => {
        const [meta, sentiment, ...rest] = analyzeStreamChunks((await request.json()) as AnalyzeBody);
        const done = rest.pop() as string;
        return sseResponse(
          (async function* () {
            yield meta;
            yield sentiment;
            await afterSentiment.promise;
            yield* rest;
            await afterDeltas.promise;
            yield done;
          })(),
        );
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    try {
      await user.click(screen.getByRole("radio", { name: /Combined/ }));
      await submit(user);

      expect(await screen.findByTestId("signal-badge")).toHaveTextContent("Bullish");
      expect(screen.getByText(ENTITY)).toBeInTheDocument();
      expect(screen.getByText("Score").nextSibling).toHaveTextContent("+2");
      expect(screen.queryByTestId("summary-text")).not.toBeInTheDocument();
      expect(screen.getByTestId("mark-analyze-signal")).toBeInTheDocument();
      expect(screen.queryByText("server total")).not.toBeInTheDocument();

      afterSentiment.open();
      await waitFor(() => expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY_TEXT));
      expect(screen.getByTestId("summary-status")).toHaveTextContent("Streaming...");
      expect(screen.getByTestId("mark-analyze-token")).toBeInTheDocument();
      expect(screen.queryByText("server total")).not.toBeInTheDocument();

      afterDeltas.open();
      expect(await screen.findByText("server total")).toBeInTheDocument();
      expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete");
      expect(screen.getByTestId("segment-analyze-prompt")).toBeInTheDocument();
      expect(screen.getByTestId("segment-analyze-generation")).toBeInTheDocument();
    } finally {
      afterSentiment.open();
      afterDeltas.open();
    }
  });

  it("shows a below-threshold call as No signal with the model's rule and a cold start", async () => {
    server.use(
      http.post(`${API}/signal`, async ({ request }) => {
        const res = signalResponse((await request.json()) as SignalBody);
        return HttpResponse.json({
          ...res,
          signal: abstainedDecision,
          timings: timings({ wall_ms: 12_100, cold_start: true }, 12_300),
        });
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    await waitFor(() => expect(optionLabels(screen.getByLabelText("Signal model"))).toHaveLength(5));
    await user.selectOptions(screen.getByLabelText("Signal model"), "clef-flash:9b-q8_0");
    await submit(user);

    const badge = await screen.findByTestId("signal-badge");
    expect(badge).toHaveTextContent("No signal");
    expect(badge).toHaveAttribute("data-sentiment", "neutral");
    expect(screen.getByTestId("abstain-note")).toHaveTextContent(
      "the top class (Bullish, 73.0%) is below this model's 80.0% threshold",
    );
    expect(screen.getByTestId("signal-threshold")).toHaveTextContent("call needs ≥ 80.0%");
    expect(screen.getByTestId("signal-cold-start")).toHaveTextContent(
      "Cold start includes loading the model into memory",
    );
    expect(screen.getByRole("status")).toHaveTextContent("No signal: below this model's threshold.");

    await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
    const laneDetail = (name: string) =>
      within(screen.getByRole("heading", { name, level: 3 }).parentElement as HTMLElement);
    const signalLane = laneDetail("Signal");
    expect(signalLane.getByText("cold start").nextSibling).toHaveTextContent("yes, model loaded");
    const summaryLane = laneDetail("Summary");
    expect(summaryLane.getByText("cold start").nextSibling).toHaveTextContent("no, already loaded");
    expect(screen.queryByTestId("summary-cold-start")).not.toBeInTheDocument();
  });

  it("combined streaming says the model is loading on a cold start, then reports the load time", async () => {
    const afterMeta = gate();
    const afterSentiment = gate();
    server.use(
      http.post(`${API}/analyze/stream`, async ({ request }) => {
        const body = (await request.json()) as AnalyzeBody;
        const [, sentiment, ...rest] = analyzeStreamChunks(body);
        rest.pop();
        const done = analyzeResponse(body);
        const inference = {
          wall_ms: 11_200,
          ttft_ms: 8_900,
          load_ms: 8_400,
          eval_ms: 2_100,
          cold_start: true,
        };
        return sseResponse(
          (async function* () {
            yield sseFrame("meta", metaEvent(done.meta, true));
            await afterMeta.promise;
            yield sentiment;
            await afterSentiment.promise;
            yield* rest;
            yield sseFrame("done", { result: { ...done, timings: timings(inference, 11_500) } });
          })(),
        );
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    try {
      await user.click(screen.getByRole("radio", { name: /Combined/ }));
      await submit(user);

      await waitFor(() =>
        expect(screen.getByTestId("summary-status")).toHaveTextContent(
          "Loading the model into memory (cold start)...",
        ),
      );
      expect(
        screen.getByText("Waiting for the signal while the model loads into memory (cold start)..."),
      ).toBeInTheDocument();

      // The score came from the same generation, so the model is loaded before the first summary token.
      afterMeta.open();
      expect(await screen.findByTestId("signal-badge")).toHaveTextContent("Bullish");
      expect(screen.getByTestId("summary-status")).toHaveTextContent("Waiting for the first token...");

      afterSentiment.open();
      await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
      expect(screen.getByTestId("signal-threshold")).toHaveTextContent("|score| ≥ 1");
      expect(screen.getByTestId("signal-cold-start")).toHaveTextContent(
        "Cold start model loaded for this request (8.40 s)",
      );
      expect(screen.getByTestId("summary-cold-start")).toHaveTextContent(
        "cold start: model loaded for this request (8.40 s)",
      );
      expect(screen.getByTestId("segment-analyze-load")).toBeInTheDocument();
    } finally {
      afterMeta.open();
      afterSentiment.open();
    }
  });

  it("uses the JSON endpoints when streaming is off", async () => {
    const streamed = vi.fn();
    server.use(
      http.post(`${API}/summary/stream`, () => {
        streamed();
        return HttpResponse.error();
      }),
      http.post(`${API}/analyze/stream`, () => {
        streamed();
        return HttpResponse.error();
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    await user.click(screen.getByRole("checkbox", { name: /Streaming/ }));

    await submit(user);
    await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
    expect(screen.getByTestId("signal-badge")).toHaveTextContent("Bullish");
    expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY_TEXT);
    expect(screen.queryByTestId("mark-summary-token")).not.toBeInTheDocument();
    expect(screen.getByTestId("segment-summary-generation")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /Combined/ }));
    await submit(user);
    expect(await screen.findByText(ENTITY)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
    expect(screen.getByTestId("lane-analyze")).toBeInTheDocument();
    expect(streamed).not.toHaveBeenCalled();
  });

  it("shows one error banner with code and hint when both requests hit a paywall", async () => {
    const paywalled = () =>
      errorResponse(422, {
        code: "paywalled",
        message: "The article is for subscribers only.",
        hint: "Try the company press release instead.",
        retryable: false,
        upstream_status: null,
      });
    server.use(http.post(`${API}/signal`, paywalled), http.post(`${API}/summary/stream`, paywalled));
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    await submit(user);

    const alert = await screen.findByRole("alert");
    await waitFor(() => expect(alert).toHaveTextContent("Signal and Summary"));
    expect(alert).toHaveTextContent("Paywalled article");
    expect(within(alert).getByText("paywalled")).toBeInTheDocument();
    expect(alert).toHaveTextContent("HTTP 422");
    expect(alert).toHaveTextContent("Try the company press release instead.");
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByText(/No signal: the request failed/)).toBeInTheDocument();
    expect(screen.getByTestId("summary-status")).toHaveTextContent("Failed");
  });

  it("aborts the previous run on resubmit and on cancel", async () => {
    const hold = gate();
    let calls = 0;
    server.use(
      http.post(`${API}/summary/stream`, async ({ request }) => {
        calls += 1;
        const chunks = summaryStreamChunks((await request.json()) as SummaryBody);
        if (calls > 1) return sseResponse(chunks);
        return sseResponse(
          (async function* () {
            yield chunks[0];
            yield chunks[2];
            await hold.promise;
            yield* chunks.slice(3);
          })(),
        );
      }),
    );
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    try {
      await submit(user);
      await waitFor(() =>
        expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY_CHUNKS[0].trim()),
      );

      await submit(user);
      await waitFor(() => expect(screen.getByTestId("summary-status")).toHaveTextContent("Complete"));
      hold.open();
      await tick();
      expect(screen.getByTestId("summary-text").textContent).toBe(SUMMARY_TEXT);
      expect(calls).toBe(2);
    } finally {
      hold.open();
    }

    const stuck = gate();
    server.use(
      http.post(`${API}/signal`, async () => {
        await stuck.promise;
        return HttpResponse.json(signalResponse());
      }),
    );
    try {
      await submit(user);
      await user.click(await screen.findByRole("button", { name: "Cancel" }));
      expect(screen.getByText("Cancelled.")).toBeInTheDocument();
      stuck.open();
      await tick();
      expect(screen.queryByTestId("signal-badge")).not.toBeInTheDocument();
    } finally {
      stuck.open();
    }
  });

  it("rejects a non-http URL without calling the API", async () => {
    const user = userEvent.setup();
    render(<AnalyzeApp />);
    await submit(user, "ftp://example.com/file");
    expect(screen.getByText(/Enter a full http\(s\) URL/)).toBeInTheDocument();
    expect(screen.getByLabelText("Article URL")).toHaveAttribute("aria-invalid", "true");
    expect(screen.queryByTestId("summary-status")).not.toBeInTheDocument();
  });
});
