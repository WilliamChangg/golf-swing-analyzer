import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests run against the Vite dev server rather than the packaged
 * Tauri binary. Driving the real WebView needs tauri-driver plus a platform
 * WebDriver, which is not yet wired up; that is recorded as a limitation rather
 * than pretended away.
 *
 * The Tauri IPC bridge is stubbed per-test (see e2e/fixtures.ts), so these
 * tests cover layout and rendering of engine data. The Rust transport itself is
 * covered by `cargo test`, and the engine by pytest.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",

  use: {
    baseURL: "http://localhost:1420",
    trace: "on-first-retry",
  },

  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

  webServer: {
    command: "npm run dev:web -w @gsa/desktop",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
