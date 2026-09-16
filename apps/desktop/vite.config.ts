/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tauri drives this dev server, so the port is fixed and failures must be loud
// rather than silently falling back to another port the Rust side is not
// pointed at.
const DEV_PORT = 1420;

export default defineConfig({
  plugins: [react(), tailwindcss()],

  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },

  // Keep Vite's output readable when it is running inside `tauri dev`, which
  // interleaves Rust compiler output on the same terminal.
  clearScreen: false,

  server: {
    port: DEV_PORT,
    strictPort: true,
    watch: {
      // Rust build artifacts churn constantly; watching them causes reload storms.
      ignored: ["**/src-tauri/**"],
    },
  },

  envPrefix: ["VITE_", "TAURI_"],

  build: {
    // Safari 13 is the floor for the macOS WebView Tauri uses.
    target: "safari14",
    sourcemap: true,
  },

  test: {
    name: "desktop",
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    css: true,
  },
});
