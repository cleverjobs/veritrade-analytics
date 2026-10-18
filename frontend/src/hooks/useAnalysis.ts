import { useCallback, useEffect, useReducer, useRef, type Dispatch } from "react";

import {
  ApiError,
  isAbortError,
  postAnalyze,
  postSignal,
  postSummary,
  toApiError,
  type AnalyzeBody,
  type AnalyzeResponse,
  type ExtractorName,
  type ParserName,
  type Scope,
  type SignalBody,
  type SummaryBody,
  type SummaryResponse,
} from "@/lib/api";
import { initialRunState, runReducer, type LaneId, type Mode, type RunAction } from "@/lib/run-state";
import { streamAnalyze, streamSummary, type StreamEvent } from "@/lib/sse";

export interface RunParams {
  url: string;
  mode: Mode;
  streaming: boolean;
  parser: ParserName;
  extractor: ExtractorName;
  /** Signal scope in split mode; /analyze keeps its server default. */
  scope: Scope;
  /** Empty string lets the server pick the endpoint default. */
  signalModel: string;
  summaryModel: string;
  analyzeModel: string;
}

interface LaneContext {
  runId: number;
  dispatch: Dispatch<RunAction>;
  signal: AbortSignal;
  at: () => number;
}

const withModel = (model: string) => (model ? { model } : {});

async function guarded(ctx: LaneContext, lane: LaneId, work: () => Promise<void>): Promise<void> {
  try {
    await work();
  } catch (err) {
    if (ctx.signal.aborted || isAbortError(err)) return;
    ctx.dispatch({ type: "error", runId: ctx.runId, lane, atMs: ctx.at(), error: toApiError(err) });
  }
}

function handleEvent(
  ctx: LaneContext,
  lane: LaneId,
  event: StreamEvent<SummaryResponse | AnalyzeResponse>,
): void {
  const tag = { runId: ctx.runId, lane, atMs: ctx.at() };
  switch (event.type) {
    case "meta":
      ctx.dispatch({ type: "meta", ...tag, meta: event.data.meta });
      return;
    case "sentiment":
      ctx.dispatch({
        type: "signal",
        ...tag,
        signal: event.data.signal,
        entity: event.data.entity ?? null,
        serverMs: event.data.elapsed_ms,
      });
      return;
    case "summary_delta":
      ctx.dispatch({ type: "delta", ...tag, text: event.data.text });
      return;
    case "done": {
      const result = event.data.result;
      const combined = "signal" in result ? result : null;
      ctx.dispatch({
        type: "done",
        ...tag,
        meta: result.meta,
        timings: result.timings,
        summary: result.summary,
        truncated: result.truncated,
        signal: combined?.signal,
        entity: combined ? (combined.entity ?? null) : undefined,
      });
      return;
    }
    case "error":
      ctx.dispatch({ type: "error", ...tag, error: ApiError.fromBody(event.data.error) });
      return;
  }
}

function runSignal(ctx: LaneContext, body: SignalBody) {
  return guarded(ctx, "signal", async () => {
    const res = await postSignal(body, ctx.signal);
    ctx.dispatch({
      type: "done",
      runId: ctx.runId,
      lane: "signal",
      atMs: ctx.at(),
      meta: res.meta,
      timings: res.timings,
      signal: res.signal,
    });
  });
}

function runSummary(ctx: LaneContext, body: SummaryBody, streaming: boolean) {
  return guarded(ctx, "summary", async () => {
    if (streaming) {
      for await (const event of streamSummary(body, { signal: ctx.signal }))
        handleEvent(ctx, "summary", event);
      return;
    }
    const res = await postSummary(body, ctx.signal);
    handleEvent(ctx, "summary", { type: "done", data: { result: res } });
  });
}

function runAnalyze(ctx: LaneContext, body: AnalyzeBody, streaming: boolean) {
  return guarded(ctx, "analyze", async () => {
    if (streaming) {
      for await (const event of streamAnalyze(body, { signal: ctx.signal }))
        handleEvent(ctx, "analyze", event);
      return;
    }
    const res = await postAnalyze(body, ctx.signal);
    handleEvent(ctx, "analyze", { type: "done", data: { result: res } });
  });
}

/** Owns one in-flight run; starting a new run aborts the previous one. */
export function useAnalysis() {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const controllerRef = useRef<AbortController | null>(null);
  const runIdRef = useRef(0);
  const startedAtRef = useRef(0);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const start = useCallback((params: RunParams) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const runId = ++runIdRef.current;
    const t0 = performance.now();
    startedAtRef.current = t0;
    const ctx: LaneContext = { runId, dispatch, signal: controller.signal, at: () => performance.now() - t0 };
    const base = { url: params.url, parser: params.parser, extractor: params.extractor };

    if (params.mode === "split") {
      dispatch({
        type: "start",
        runId,
        mode: "split",
        streaming: params.streaming,
        lanes: ["signal", "summary"],
      });
      // Fired together on purpose: the backend single-flight shares one fetch between them.
      void runSignal(ctx, { ...base, scope: params.scope, ...withModel(params.signalModel) });
      void runSummary(ctx, { ...base, ...withModel(params.summaryModel) }, params.streaming);
    } else {
      dispatch({ type: "start", runId, mode: "combined", streaming: params.streaming, lanes: ["analyze"] });
      void runAnalyze(ctx, { ...base, ...withModel(params.analyzeModel) }, params.streaming);
    }
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    dispatch({ type: "cancel", runId: runIdRef.current, atMs: performance.now() - startedAtRef.current });
  }, []);

  return { state, start, cancel };
}
