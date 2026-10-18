import { defineConfig, devices } from "@playwright/test";

const CI = Boolean(process.env.CI);

// Runs against an already-running stack (docker compose serves the export and the API on one origin).
export default defineConfig({
  testDir: "./e2e",
  // One Ollama instance: parallel inference would only measure contention.
  fullyParallel: false,
  workers: 1,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  // CPU-only inference on CI runners is slow; assertions wait on results, not on fixed sleeps.
  timeout: 180_000,
  expect: { timeout: 90_000 },
  reporter: CI ? [["github"], ["html", { open: "never" }]] : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:8000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
