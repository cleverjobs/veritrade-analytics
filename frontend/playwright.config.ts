import { defineConfig, devices } from "@playwright/test";

const CI = Boolean(process.env.CI);

// Runs against an already-running stack: `docker compose -f compose.yaml -f compose.e2e.yaml up -d --wait`.
export default defineConfig({
  testDir: "./e2e",
  // One Ollama instance: parallel inference would only measure contention.
  fullyParallel: false,
  workers: 1,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  // CPU-only inference on a 4-vCPU runner is slow; assertions wait on results, not on fixed sleeps.
  timeout: 300_000,
  expect: { timeout: 150_000 },
  reporter: CI
    ? [["github"], ["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:8000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
