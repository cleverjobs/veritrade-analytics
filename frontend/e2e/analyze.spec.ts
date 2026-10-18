import { expect, test, type Page } from "@playwright/test";

// The fixture site serves skeleton articles; models stay at server defaults unless overridden.
const ARTICLE_URL = process.env.E2E_ARTICLE_URL ?? "";
const SIGNAL_MODEL = process.env.E2E_SIGNAL_MODEL ?? "";
const SUMMARY_MODEL = process.env.E2E_SUMMARY_MODEL ?? "";
const ANALYZE_MODEL = process.env.E2E_ANALYZE_MODEL ?? "";
const EXTRACTOR = process.env.E2E_EXTRACTOR ?? "trafilatura";

// Model-agnostic: any valid label passes, the point is the plumbing.
const SENTIMENT = /^(Bullish|Bearish|Neutral)$/;

async function open(page: Page, mode: "Split" | "Combined", streaming = true) {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // Pickers fill from /api/models; waiting here keeps selectOption from racing the fetch.
  await expect(page.getByLabel("Signal model")).toContainText("(default)");
  await page.getByRole("radio", { name: new RegExp(mode) }).check();
  const toggle = page.getByRole("checkbox", { name: /Streaming/ });
  if (streaming) await toggle.check();
  else await toggle.uncheck();
}

async function pick(page: Page, label: string, value: string) {
  if (value) await page.getByLabel(label).selectOption(value);
}

async function analyze(page: Page) {
  await page.getByLabel("Article URL").fill(ARTICLE_URL);
  await page.getByRole("button", { name: "Analyze" }).click();
}

test.describe("analyze page", () => {
  test.skip(!ARTICLE_URL, "set E2E_ARTICLE_URL to an article the stack can fetch");

  test("split mode shows the signal, a streamed summary and timings", async ({ page }) => {
    await open(page, "Split");
    await pick(page, "Signal model", SIGNAL_MODEL);
    await pick(page, "Summary model", SUMMARY_MODEL);
    await analyze(page);

    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("summary-text")).not.toBeEmpty();
    // Not getByRole("alert"): Next's route announcer also carries that role.
    await expect(page.getByTestId("error-banner")).toHaveCount(0);

    const latency = page.getByTestId("latency");
    await expect(latency.getByTestId("lane-signal")).toBeVisible();
    await expect(latency.getByTestId("lane-summary")).toBeVisible();
    await expect(latency.getByTestId("mark-signal-signal")).toBeAttached();
    await expect(latency.getByTestId("mark-summary-token")).toBeAttached();
    await expect(latency.getByText("server total")).toHaveCount(2);
  });

  test("combined streaming delivers sentiment, summary and timings in one call", async ({ page }) => {
    await open(page, "Combined");
    await pick(page, "Analyze model", ANALYZE_MODEL);
    await analyze(page);

    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("summary-text")).not.toBeEmpty();
    const latency = page.getByTestId("latency");
    await expect(latency.getByTestId("lane-analyze")).toBeVisible();
    await expect(latency.getByTestId("mark-analyze-signal")).toBeAttached();
    await expect(latency.getByText("server total")).toHaveCount(1);
  });

  test("switching the extractor is reflected in the article metadata", async ({ page }) => {
    await open(page, "Split", false);
    await pick(page, "Signal model", SIGNAL_MODEL);
    await pick(page, "Summary model", SUMMARY_MODEL);
    await page.getByLabel("Extractor").selectOption(EXTRACTOR);
    await expect(page.getByLabel("Parser")).toBeDisabled();
    await analyze(page);

    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("article-extraction")).toContainText(EXTRACTOR);
    await expect(page.getByTestId("signal-badge")).toHaveText(SENTIMENT);
  });
});
