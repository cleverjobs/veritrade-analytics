import {
  ApiError,
  apiFetch,
  jsonPost,
  type AnalyzeBody,
  type AnalyzeResponse,
  type ArticleMeta,
  type CacheInfo,
  type ErrorBody,
  type SignalResult,
  type SummaryBody,
  type SummaryResponse,
} from "./api";

// Payloads mirror backend/src/veritrade/api/contracts.py; OpenAPI cannot describe SSE bodies.
export interface MetaEventData {
  meta: ArticleMeta;
  fetch_ms: number;
  parse_ms: number;
  extract_ms: number;
  cache: CacheInfo;
  /** True when the model has to be loaded before its first token; null when unknown. */
  cold_start?: boolean | null;
}

export interface SentimentEventData {
  signal: SignalResult;
  entity?: string | null;
  /** Server-side milliseconds since the request was received. */
  elapsed_ms: number;
}

export interface SummaryDeltaEventData {
  text: string;
}

export interface DoneEventData<R> {
  result: R;
}

export interface ErrorEventData {
  error: ErrorBody;
}

export type StreamEvent<R> =
  | { type: "meta"; data: MetaEventData }
  | { type: "sentiment"; data: SentimentEventData }
  | { type: "summary_delta"; data: SummaryDeltaEventData }
  | { type: "done"; data: DoneEventData<R> }
  | { type: "error"; data: ErrorEventData };

export type StreamEventType = StreamEvent<unknown>["type"];
const EVENT_TYPES: ReadonlySet<string> = new Set<StreamEventType>([
  "meta",
  "sentiment",
  "summary_delta",
  "done",
  "error",
]);

export interface SseFrame {
  event: string;
  data: string;
  id: string | null;
}

// Incremental text/event-stream parser (WHATWG rules): any of CRLF/LF/CR ends a line, a blank
// line dispatches, ":" lines are comments, repeated data lines join with "\n".
export class SseParser {
  private buffer = "";
  private skipLeadingLf = false;
  private eventType = "";
  private dataLines: string[] = [];
  private lastId: string | null = null;

  push(chunk: string): SseFrame[] {
    if (!chunk) return [];
    let text = chunk;
    // A CR that ended the previous chunk may be the first half of a CRLF.
    if (this.skipLeadingLf) {
      this.skipLeadingLf = false;
      if (text.startsWith("\n")) text = text.slice(1);
    }
    this.buffer += text;

    const frames: SseFrame[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch !== "\n" && ch !== "\r") continue;
      const line = this.buffer.slice(start, i);
      if (ch === "\r") {
        if (i + 1 === this.buffer.length) this.skipLeadingLf = true;
        else if (this.buffer[i + 1] === "\n") i++;
      }
      start = i + 1;
      const frame = this.processLine(line);
      if (frame) frames.push(frame);
    }
    this.buffer = this.buffer.slice(start);
    return frames;
  }

  private processLine(line: string): SseFrame | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null;

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") this.eventType = value;
    else if (field === "data") this.dataLines.push(value);
    else if (field === "id" && !value.includes("\0")) this.lastId = value;
    return null;
  }

  private dispatch(): SseFrame | null {
    const lines = this.dataLines;
    const event = this.eventType || "message";
    this.dataLines = [];
    this.eventType = "";
    if (lines.length === 0) return null;
    return { event, data: lines.join("\n"), id: this.lastId };
  }
}

/** Yields parsed frames; an unterminated trailing frame is dropped, as the SSE spec requires. */
export async function* readSseFrames(body: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  // lib.dom types the decoder's input as BufferSource, which TS will not accept for Uint8Array chunks.
  const decoder = new TextDecoderStream() as unknown as TransformStream<Uint8Array, string>;
  const reader = body.pipeThrough(decoder).getReader();
  const parser = new SseParser();
  let finished = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        finished = true;
        return;
      }
      yield* parser.push(value);
    }
  } finally {
    // Early exit (consumer break or abort) must release the network stream.
    if (!finished) await reader.cancel().catch(() => undefined);
  }
}

export function toStreamEvent<R>(frame: SseFrame): StreamEvent<R> | null {
  if (!EVENT_TYPES.has(frame.event)) return null;
  let data: unknown;
  try {
    data = JSON.parse(frame.data);
  } catch {
    throw new ApiError({ code: "bad_response", message: `Malformed "${frame.event}" event from the API.` });
  }
  return { type: frame.event, data } as StreamEvent<R>;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

// Yields typed events until `done` or `error`. Failures before the stream opens arrive as the
// JSON error envelope and are thrown as ApiError.
export async function* streamEvents<R>(
  path: string,
  body: unknown,
  { signal }: StreamOptions = {},
): AsyncGenerator<StreamEvent<R>> {
  const res = await apiFetch(path, jsonPost(body, signal, "text/event-stream"));
  if (!res.body) throw new ApiError({ code: "bad_response", message: "The API returned an empty stream." });

  for await (const frame of readSseFrames(res.body)) {
    const event = toStreamEvent<R>(frame);
    if (!event) continue;
    yield event;
    if (event.type === "done" || event.type === "error") return;
  }
  throw new ApiError({
    code: "bad_response",
    message: "The stream closed before it finished.",
    retryable: true,
  });
}

export function streamSummary(body: SummaryBody, options?: StreamOptions) {
  return streamEvents<SummaryResponse>("/api/summary/stream", body, options);
}

export function streamAnalyze(body: AnalyzeBody, options?: StreamOptions) {
  return streamEvents<AnalyzeResponse>("/api/analyze/stream", body, options);
}
