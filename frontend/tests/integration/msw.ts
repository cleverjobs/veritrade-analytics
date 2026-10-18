import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "@/mocks/node";

/** Registers the shared MSW server for the calling test file. */
export function setupMswServer() {
  beforeAll(() => server.listen({ onUnhandledFrame: "error" }));
  afterEach(() => server.resetHandlers());
  afterAll(() => server.close());
  return server;
}
