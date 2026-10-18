import { expect, test, type Page } from "@playwright/test";

// Defaults match compose.e2e.yaml: the fixture site answers for publisher hostnames inside the stack.
const ARTICLE_URL = process.env.E2E_ARTICLE_URL ?? "http://www.cnn.com/vt-skeleton/article-1";
const DECISION_MODEL = process.env.E2E_DECISION_MODEL ?? "tev1:0.8b-q8_0";
const EXTRACTOR = process.env.E2E_EXTRACTOR ?? "trafilatura";

// Model-agnostic: any valid label passes, the point is the plumbing.
const SENTIMENT = /^(Bullish|Bearish|Neutral)$/;
const LABELS = ["bullish", "bearish", "neutral"];

async function open(page: Page, mode: "Split" | "Combined", streaming: boolean) {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // Pickers fill from /api/models; waiting here keeps selectOption from racing the fetch.
  await expect(page.getByLabel("Signal model")).toContainText("(default)");
  await page.getByRole("radio", { name: new RegExp(mode) }).check();
  const toggle = page.getByRole("checkbox", { name: /Streaming/ });
  if (streaming) await toggle.check();
  else await toggle.uncheck();
}

async function analyze(page: Page) {
  await page.getByLabel("Article URL").fill(ARTICLE_URL);
  await page.getByRole("button", { name: "Analyze" }).click();
}

async function expectNoError(page: Page) {
  // Not getByRole("alert"): Next's route announcer also carries that role.
  await expect(page.getByTestId("error-banner")).toHaveCount(0);
}

test.describe("analyze page", () => {
  test("split mode streams a summary next to the signal, with timings", async ({ page }) => {
    await open(page, "Split", true);
    await analyze(page);

    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("summary-text")).not.toBeEmpty();
    await expectNoError(page);

    const latency = page.getByTestId("latency");
    await expect(latency.getByTestId("lane-signal")).toBeVisible();
    await expect(latency.getByTestId("lane-summary")).toBeVisible();
    await expect(latency.getByTestId("mark-signal-signal")).toBeAttached();
    await expect(latency.getByTestId("mark-summary-token")).toBeAttached();
    await expect(latency.getByText("server total")).toHaveCount(2);
  });

  test("combined streaming delivers sentiment, summary and timings in one call", async ({ page }) => {
    await open(page, "Combined", true);
    await analyze(page);

    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("summary-text")).not.toBeEmpty();
    await expectNoError(page);

    const latency = page.getByTestId("latency");
    await expect(latency.getByTestId("lane-analyze")).toBeVisible();
    await expect(latency.getByTestId("mark-analyze-signal")).toBeAttached();
    await expect(latency.getByText("server total")).toHaveCount(1);
  });

  test("decision model and extractor switch flow through the JSON endpoints", async ({ page }) => {
    await open(page, "Split", false);
    await page.getByLabel("Signal model").selectOption(DECISION_MODEL);
    await page.getByLabel("Extractor").selectOption(EXTRACTOR);
    await expect(page.getByLabel("Parser")).toBeDisabled();
    await analyze(page);

    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
    const signal = page.getByRole("region", { name: "Signal", exact: true });
    await expect(signal).toContainText(DECISION_MODEL);
    await expect(signal).toContainText("probability");
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("article-extraction")).toContainText(EXTRACTOR);
    await expectNoError(page);
  });
});

test("signal-only API call with the decision model returns a label and stage timings", async ({
  request,
}) => {
  const resp = await request.post("/api/signal", { data: { url: ARTICLE_URL, model: DECISION_MODEL } });
  expect(resp.status()).toBe(200);
  const body = await resp.json();

  expect(LABELS).toContain(body.signal.sentiment);
  expect(body.signal.policy).toBe("probability");
  expect(Object.keys(body.signal.probabilities).sort()).toEqual([...LABELS].sort());
  expect(body.meta).toMatchObject({ model: DECISION_MODEL, model_kind: "decision", endpoint: "signal" });
  // The fixture keeps the publisher hostname, so the source profile still resolves.
  expect(body.meta.source).toBeTruthy();
  expect(body.timings.inference.wall_ms).toBeGreaterThan(0);
  expect(body.timings.extract_ms).toBeGreaterThanOrEqual(0);
});
