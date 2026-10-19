import { expect, test, type Page } from "@playwright/test";

// Source key -> hostname the fixture site answers for inside the stack (compose.e2e.yaml).
const FIXTURE_HOSTS: Record<string, string> = {
  yahoo: "finance.yahoo.com",
  cnbc: "www.cnbc.com",
  fool: "www.fool.com",
  prnewswire: "www.prnewswire.com",
  globenewswire: "www.globenewswire.com",
  sec: "www.sec.gov",
  cnn: "www.cnn.com",
  fortune: "fortune.com",
  guardian: "www.theguardian.com",
  bbc: "www.bbc.com",
};
const fixtureUrl = (host: string) => `http://${host}/vt-skeleton/article-1`;

const ARTICLE_URL = process.env.E2E_ARTICLE_URL ?? fixtureUrl(FIXTURE_HOSTS.cnbc);
const DECISION_MODEL = process.env.E2E_DECISION_MODEL ?? "tev1:0.8b-q8_0";
const EXTRACTOR = process.env.E2E_EXTRACTOR ?? "trafilatura";

// Model-agnostic: any valid label passes, the point is the plumbing. A call under the model's
// frozen threshold reads "No signal".
const SENTIMENT = /^(Bullish|Bearish|Neutral|No signal)$/;
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
    await expect(signal.getByTestId("signal-threshold")).toBeVisible();
    await expect(page.getByTestId("summary-status")).toHaveText("Complete");
    await expect(page.getByTestId("article-extraction")).toContainText(EXTRACTOR);
    await expectNoError(page);
  });
});

test.describe("signal-only API with the decision model", () => {
  for (const [source, host] of Object.entries(FIXTURE_HOSTS)) {
    test(`${source}: label, source profile and stage timings`, async ({ request }) => {
      // no_cache: earlier tests and the CI warm-up may have cached this exact request.
      const resp = await request.post("/api/signal", {
        data: { url: fixtureUrl(host), model: DECISION_MODEL, no_cache: true },
      });
      expect(resp.status()).toBe(200);
      const body = await resp.json();

      expect(LABELS).toContain(body.signal.sentiment);
      expect(body.signal.policy).toBe("probability");
      expect(body.signal.threshold).toEqual(expect.any(Number));
      expect(Object.keys(body.signal.probabilities).sort()).toEqual([...LABELS].sort());
      expect(body.meta).toMatchObject({
        source,
        model: DECISION_MODEL,
        model_kind: "decision",
        endpoint: "signal",
        cached: false,
      });
      expect(body.timings.inference.wall_ms).toBeGreaterThan(0);
      // null when the residency probe times out on a slow runner.
      expect([true, false, null]).toContain(body.timings.inference.cold_start);
    });
  }
});
