import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  ERROR_CODES,
  apiFetch,
  apiUrl,
  errorFromPayload,
  getModels,
  isAbortError,
  postSignal,
  readError,
} from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("errorFromPayload", () => {
  it("maps the API error envelope", () => {
    const err = errorFromPayload(502, {
      error: {
        code: "source_blocked",
        message: "Blocked by Cloudflare",
        hint: "Try another source",
        retryable: false,
        upstream_status: 403,
      },
    });
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({
      code: "source_blocked",
      message: "Blocked by Cloudflare",
      hint: "Try another source",
      status: 502,
      retryable: false,
      upstreamStatus: 403,
    });
  });

  it.each(ERROR_CODES)("keeps the %s code", (code) => {
    expect(errorFromPayload(400, { error: { code, message: "m" } }).code).toBe(code);
  });

  it("maps FastAPI validation detail to invalid_request", () => {
    const err = errorFromPayload(422, {
      detail: [{ loc: ["body", "url"], msg: "Input should be a valid URL", type: "url_parsing" }],
    });
    expect(err).toMatchObject({ code: "invalid_request", message: "url: Input should be a valid URL" });
  });

  it("falls back to http_error for unknown codes and junk", () => {
    expect(errorFromPayload(500, { error: { code: "nope", message: "x" } }).code).toBe("http_error");
    expect(errorFromPayload(503, null)).toMatchObject({ code: "http_error", retryable: true });
  });
});

describe("readError", () => {
  it("handles a non-JSON body", async () => {
    const err = await readError(new Response("<html>Bad Gateway</html>", { status: 502 }));
    expect(err).toMatchObject({ code: "http_error", status: 502 });
  });
});

describe("apiUrl", () => {
  it("uses NEXT_PUBLIC_API_BASE when set", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "http://localhost:8000/");
    expect(apiUrl("/api/models")).toBe("http://localhost:8000/api/models");
  });

  it("falls back to the page origin", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "");
    expect(apiUrl("/api/models")).toBe(`${window.location.origin}/api/models`);
  });
});

describe("apiFetch", () => {
  it("turns a network failure into network_error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await expect(apiFetch("/api/health")).rejects.toMatchObject({ code: "network_error", retryable: true });
  });

  it("rethrows aborts untouched", async () => {
    const abort = new DOMException("aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abort));
    const err: unknown = await apiFetch("/api/health").catch((e: unknown) => e);
    expect(err).toBe(abort);
    expect(isAbortError(err)).toBe(true);
  });

  it("throws the parsed envelope on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          Response.json({ error: { code: "model_not_allowed", message: "no" } }, { status: 400 }),
        ),
    );
    await expect(getModels()).rejects.toMatchObject({ code: "model_not_allowed", status: 400 });
  });
});

describe("postSignal", () => {
  it("sends JSON with the abort signal and returns the body", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => Response.json({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();
    await expect(postSignal({ url: "https://x.test/a", scope: "lede" }, controller.signal)).resolves.toEqual({
      ok: true,
    });
    const [, init] = fetchMock.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(init?.signal).toBe(controller.signal);
    expect(JSON.parse(String(init?.body))).toEqual({ url: "https://x.test/a", scope: "lede" });
  });

  it("reports invalid JSON as bad_response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not json", { status: 200 })));
    await expect(postSignal({ url: "https://x.test/a" })).rejects.toMatchObject({ code: "bad_response" });
  });
});
