import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: { tsconfigPaths: true },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    // Playwright specs run against the compose stack, never under Vitest.
    exclude: [...configDefaults.exclude, "e2e/**"],
    setupFiles: ["tests/setup.ts"],
  },
});
