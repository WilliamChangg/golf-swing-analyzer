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
  },
});
