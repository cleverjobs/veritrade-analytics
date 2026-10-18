import { HttpResponse } from "msw";

import type { StreamEventType } from "@/lib/sse";

export function sseFrame(event: StreamEventType, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export const SSE_PING = ": ping\n\n";

/** Pull-based so a test can hold the stream open between chunks (e.g. awaiting a gate). */
export function sseStream(chunks: AsyncIterable<string> | Iterable<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const iterator =
    Symbol.asyncIterator in chunks ? chunks[Symbol.asyncIterator]() : toAsync(chunks[Symbol.iterator]());
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) controller.close();
      else controller.enqueue(encoder.encode(value));
    },
    async cancel() {
      await iterator.return?.();
    },
  });
}

function toAsync(iterator: Iterator<string>): AsyncIterator<string> {
  return { next: async () => iterator.next() };
}

export function sseResponse(chunks: AsyncIterable<string> | Iterable<string>): HttpResponse<ReadableStream> {
  return new HttpResponse(sseStream(chunks), {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}

export interface Gate {
  promise: Promise<void>;
  open: () => void;
}

export function gate(): Gate {
  let open: () => void = () => undefined;
  const promise = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { promise, open };
}
