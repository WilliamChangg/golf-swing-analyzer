import { defineConfig } from "vitest/config";

/**
 * Root Vitest config. Each workspace package that has tests is listed as a
 * project so `npm test` at the repo root runs everything, while each package
 * keeps its own environment settings (the desktop app needs jsdom; pure
 * TypeScript packages do not).
 */
export default defineConfig({
  test: {
    projects: ["apps/desktop"],
    coverage: {
      provider: "v8",
      include: ["apps/desktop/src/**/*.{ts,tsx}"],
      exclude: [
        "**/*.{test,spec}.{ts,tsx}",
        "**/*.fixture.ts",
        "**/test-setup.ts",
        "**/main.tsx",
      ],
      reporter: ["text", "json-summary", "lcov", "html"],
      reportsDirectory: "coverage/typescript",
      reportOnFailure: true,
      thresholds: { statements: 80, branches: 66, functions: 74, lines: 83 },
    },
  },
});
