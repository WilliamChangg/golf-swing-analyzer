import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Vitest's globals are enabled, but React Testing Library does not auto-clean
// unless the global afterEach is registered explicitly.
afterEach(() => {
  cleanup();
});
