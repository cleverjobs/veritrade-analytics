import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import { SseParser, readSseFrames, streamEvents, toStreamEvent, type SseFrame } from "@/lib/sse";

function parseAll(chunks: string[]): SseFrame[] {
  const parser = new SseParser();
  return chunks.flatMap((chunk) => parser.push(chunk));
}

function byteStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function collect<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of iterable) out.push(item);
  return out;
}

const STREAM =
  ": ping\n\n" +
  'event: meta\ndata: {"meta":{"title":"Acme"}}\n\n' +
  'event: sentiment\ndata: {"signal":{"sentiment":"bullish"},\ndata: "elapsed_ms":12}\n\n' +
  ": keep-alive\r\n" +
  'event: summary_delta\r\ndata: {"text":"Shares rose"}\r\n\r\n' +
  'event: summary_delta\rdata: {"text":" 8%."}\r\r' +
  'event: done\ndata: {"result":{}}\n\n';

const EXPECTED: Pick<SseFrame, "event" | "data">[] = [
  { event: "meta", data: '{"meta":{"title":"Acme"}}' },
  { event: "sentiment", data: '{"signal":{"sentiment":"bullish"},\n"elapsed_ms":12}' },
  { event: "summary_delta", data: '{"text":"Shares rose"}' },
  { event: "summary_delta", data: '{"text":" 8%."}' },
  { event: "done", data: '{"result":{}}' },
];

const strip = (frames: SseFrame[]) => frames.map(({ event, data }) => ({ event, data }));

describe("SseParser", () => {
  it("parses events, multi-line data and every line terminator, skipping comments", () => {
    expect(strip(parseAll([STREAM]))).toEqual(EXPECTED);
  });

  it("yields identical frames for a split at every character offset", () => {
    for (let i = 1; i < STREAM.length; i++) {
      const frames = parseAll([STREAM.slice(0, i), STREAM.slice(i)]);
      expect(strip(frames), `split at ${i}`).toEqual(EXPECTED);
    }
  });

  it("yields identical frames when fed one character at a time", () => {
    expect(strip(parseAll([...STREAM]))).toEqual(EXPECTED);
  });

  it("treats a CR ending one chunk and LF starting the next as one line break", () => {
    const frames = parseAll(["event: done\r", "\ndata: {}\r", "\n\r", "\n"]);
    expect(strip(frames)).toEqual([{ event: "done", data: "{}" }]);
  });

  it("defaults the event type and keeps the last id", () => {
    const frames = parseAll(["id: 7\ndata:x\n\n", "data\n\n"]);
    expect(frames).toEqual([
      { event: "message", data: "x", id: "7" },
      { event: "message", data: "", id: "7" },
    ]);
  });

  it("does not dispatch a block without data lines", () => {
    expect(parseAll(["event: meta\n\n", ": ping\n\n", "retry: 1000\n\n"])).toEqual([]);
  });

  it("holds back a frame until its blank line arrives", () => {
    const parser = new SseParser();
    expect(parser.push('event: meta\ndata: {"a":1}\n')).toEqual([]);
    expect(strip(parser.push("\n"))).toEqual([{ event: "meta", data: '{"a":1}' }]);
  });
});

describe("toStreamEvent", () => {
  it("parses JSON payloads of known events", () => {
    const event = toStreamEvent({ event: "summary_delta", data: '{"text":"hi"}', id: null });
    expect(event).toEqual({ type: "summary_delta", data: { text: "hi" } });
  });

  it("ignores unknown event names", () => {
    expect(toStreamEvent({ event: "message", data: "{}", id: null })).toBeNull();
  });

  it("throws a typed error on malformed JSON", () => {
    expect(() => toStreamEvent({ event: "done", data: "{oops", id: null })).toThrow(ApiError);
  });
});

describe("readSseFrames", () => {
  it("decodes UTF-8 characters split across byte chunks", async () => {
    const bytes = new TextEncoder().encode('event: summary_delta\ndata: {"text":"EUR € up"}\n\n');
    const euro = bytes.indexOf(0xe2);
    const chunks = [bytes.slice(0, euro + 1), bytes.slice(euro + 1, euro + 2), bytes.slice(euro + 2)];
    const frames = await collect(readSseFrames(byteStream(chunks)));
    expect(frames.map((f) => JSON.parse(f.data))).toEqual([{ text: "EUR € up" }]);
  });

  it("drops an unterminated trailing frame", async () => {
    const bytes = new TextEncoder().encode("event: meta\ndata: {}\n\nevent: done\ndata: {}\n");
    const frames = await collect(readSseFrames(byteStream([bytes])));
    expect(frames.map((f) => f.event)).toEqual(["meta"]);
  });
});

describe("streamEvents", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubFetch(response: Response) {
    const fetchMock = vi.fn<typeof fetch>(async () => response);
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  const sse = (text: string) =>
    new Response(byteStream([new TextEncoder().encode(text)]), {
      headers: { "Content-Type": "text/event-stream" },
    });

  it("POSTs JSON and yields typed events until done", async () => {
    const fetchMock = stubFetch(
      sse(
        'event: meta\ndata: {"meta":{}}\n\n' +
          'event: summary_delta\ndata: {"text":"a"}\n\n' +
          'event: done\ndata: {"result":{"summary":"a"}}\n\n' +
          'event: summary_delta\ndata: {"text":"after done"}\n\n',
      ),
    );
    const events = await collect(streamEvents("/api/summary/stream", { url: "https://x.test/a" }));
    expect(events.map((e) => e.type)).toEqual(["meta", "summary_delta", "done"]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/api\/summary\/stream$/);
    expect(init?.method).toBe("POST");
    expect(new Headers(init?.headers).get("Accept")).toBe("text/event-stream");
    expect(JSON.parse(String(init?.body))).toEqual({ url: "https://x.test/a" });
  });

  it("stops at an error event", async () => {
    stubFetch(sse('event: error\ndata: {"error":{"code":"inference_timeout","message":"slow"}}\n\n'));
    const events = await collect(streamEvents("/api/analyze/stream", {}));
    expect(events).toEqual([
      { type: "error", data: { error: { code: "inference_timeout", message: "slow" } } },
    ]);
  });

  it("throws when the stream closes without a terminal event", async () => {
    stubFetch(sse('event: meta\ndata: {"meta":{}}\n\n'));
    await expect(collect(streamEvents("/api/summary/stream", {}))).rejects.toMatchObject({
      code: "bad_response",
    });
  });

  it("throws the error envelope sent before the stream opens", async () => {
    stubFetch(
      Response.json(
        { error: { code: "paywalled", message: "Paywall", hint: "Try a free source", retryable: false } },
        { status: 422 },
      ),
    );
    await expect(collect(streamEvents("/api/summary/stream", {}))).rejects.toMatchObject({
      code: "paywalled",
      status: 422,
      hint: "Try a free source",
    });
  });
});
