import { http, HttpResponse, type HttpHandler } from "msw";

import type { AnalyzeBody, ErrorBody, SignalBody, SummaryBody } from "@/lib/api";

import {
  SUMMARY_CHUNKS,
  analyzeResponse,
  articleMeta,
  health,
  metaEvent,
  models,
  sentimentEvent,
  signalResponse,
  summaryResponse,
} from "./fixtures";
import { SSE_PING, sseFrame, sseResponse } from "./sse";

// "*/api" matches any origin, so the same handlers serve jsdom, next dev and Playwright.
export const API = "*/api";

export function summaryStreamChunks(body: SummaryBody): string[] {
  const done = summaryResponse(body);
  return [
    sseFrame("meta", metaEvent(done.meta)),
    SSE_PING,
    ...SUMMARY_CHUNKS.map((text) => sseFrame("summary_delta", { text })),
    sseFrame("done", { result: done }),
  ];
}

export function analyzeStreamChunks(body: AnalyzeBody): string[] {
  const done = analyzeResponse(body);
  return [
    sseFrame("meta", metaEvent(articleMeta({ endpoint: "analyze", ...body }))),
    sseFrame("sentiment", sentimentEvent),
    ...SUMMARY_CHUNKS.map((text) => sseFrame("summary_delta", { text })),
    sseFrame("done", { result: done }),
  ];
}

export function errorResponse(status: number, error: ErrorBody): HttpResponse<{ error: ErrorBody }> {
  return HttpResponse.json({ error }, { status });
}

export const handlers: HttpHandler[] = [
  http.get(`${API}/health`, () => HttpResponse.json(health)),
  http.get(`${API}/models`, () => HttpResponse.json(models)),
  http.post(`${API}/signal`, async ({ request }) =>
    HttpResponse.json(signalResponse((await request.json()) as SignalBody)),
  ),
  http.post(`${API}/summary`, async ({ request }) =>
    HttpResponse.json(summaryResponse((await request.json()) as SummaryBody)),
  ),
  http.post(`${API}/summary/stream`, async ({ request }) =>
    sseResponse(summaryStreamChunks((await request.json()) as SummaryBody)),
  ),
  http.post(`${API}/analyze`, async ({ request }) =>
    HttpResponse.json(analyzeResponse((await request.json()) as AnalyzeBody)),
  ),
  http.post(`${API}/analyze/stream`, async ({ request }) =>
    sseResponse(analyzeStreamChunks((await request.json()) as AnalyzeBody)),
  ),
];
