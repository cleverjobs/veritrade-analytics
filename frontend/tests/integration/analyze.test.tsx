import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { AnalyzeApp } from "@/components/AnalyzeApp";
import type { AnalyzeBody, SignalBody, SummaryBody } from "@/lib/api";
import { ENTITY, FIXTURE_URL, SUMMARY_CHUNKS, SUMMARY_TEXT, signalResponse } from "@/mocks/fixtures";
import { API, analyzeStreamChunks, errorResponse, summaryStreamChunks } from "@/mocks/handlers";
import { gate, sseResponse } from "@/mocks/sse";

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
        "Server default (nimble:9b-q4_K_M)",
        "clef-flash:9b-q8_0",
        "nimble:9b-q4_K_M (default)",
        "tev1:0.8b-q8_0 (CI plumbing, unevaluated)",
        "qwen3.5:0.8b-q8_0",
      ]),
    );
    expect(optionLabels(screen.getByLabelText("Summary model"))).toEqual([
      "Server default (qwen3.5:4b-q4_K_M)",
      "qwen3.5:0.8b-q8_0",
      "qwen3.5:2b-q4_K_M",
      "qwen3.5:4b-q4_K_M (default)",
    ]);

    await user.click(screen.getByRole("radio", { name: /Combined/ }));
    const analyzeSelect = screen.getByLabelText("Analyze model");
    expect(optionLabels(analyzeSelect)).toContain("ministral-3:3b");
    expect(within(analyzeSelect).getByRole("option", { name: /qwen3.5:9b-q4_K_M/ })).toBeDisabled();
    expect(screen.queryByLabelText("Signal scope")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /Split/ }));
    await user.selectOptions(screen.getByLabelText("Signal model"), "clef-flash:9b-q8_0");
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
