import { expect, test } from "@playwright/test";

const SECTIONS = [
  "Selected models",
  "Signal accuracy",
  "Latency",
  "Summary quality",
  "Extraction matrix",
  "Article set",
];

test("benchmarks page renders every results table", async ({ page }) => {
  await page.goto("/benchmarks/");
  await expect(page.getByRole("heading", { level: 1, name: "Benchmarks" })).toBeVisible();
  for (const name of SECTIONS) {
    await expect(page.getByRole("heading", { level: 2, name })).toBeVisible();
  }
  await expect(page.getByRole("table")).toHaveCount(5);
});

test("primary navigation links both pages", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Primary" });
  await nav.getByRole("link", { name: "Benchmarks" }).click();
  await expect(page).toHaveURL(/\/benchmarks\/$/);
  await expect(nav.getByRole("link", { name: "Benchmarks" })).toHaveAttribute("aria-current", "page");
  await nav.getByRole("link", { name: "Analyze" }).click();
  await expect(page.getByLabel("Article URL")).toBeVisible();
});
