import { expect, test } from "@playwright/test";

import { HEALTHY_REPORT, stubEngine } from "./fixtures";

test.describe("health screen", () => {
  test("renders the header, compute panel and every component", async ({
    page,
  }) => {
    await stubEngine(page);
    await page.goto("/");

    // Header must be visible without scrolling: a report that opens mid-list
    // hides the overall status, which is the first thing a user looks for.
    const heading = page.getByRole("heading", { name: "Golf Swing Analyzer" });
    await expect(heading).toBeInViewport();

    await expect(page.getByText("Compute")).toBeVisible();
    await expect(page.getByText("MediaPipe delegate")).toBeVisible();

    for (const component of HEALTHY_REPORT.components) {
      await expect(
        page.getByText(component.name, { exact: true }),
      ).toBeVisible();
    }
  });

  test("does not clip content above the fold", async ({ page }) => {
    await stubEngine(page);
    await page.goto("/");
    await expect(page.getByText("python", { exact: true })).toBeVisible();

    // Regression guard for a layout bug where `height: 100%` on #root made the
    // document open scrolled past the header instead of at the top.
    const scrollY = await page.evaluate(() => window.scrollY);
    expect(scrollY).toBe(0);

    const headingBox = await page
      .getByRole("heading", { name: "Golf Swing Analyzer" })
      .boundingBox();
    expect(headingBox).not.toBeNull();
    expect(headingBox!.y).toBeGreaterThanOrEqual(0);
  });

  test("shows the CPU-delegate caveat returned by the engine", async ({
    page,
  }) => {
    await stubEngine(page);
    await page.goto("/");

    await expect(page.getByText(/Pose inference runs on CPU/)).toBeVisible();
  });

  test("renders a failure rather than an empty healthy list", async ({
    page,
  }) => {
    await stubEngine(page, {
      doctor: { error: { kind: "spawn", message: "Could not find `uv`." } },
    });
    await page.goto("/");

    await expect(
      page.getByText("The analysis engine could not be started"),
    ).toBeVisible();
    await expect(page.getByText(/Could not find `uv`\./)).toBeVisible();
    await expect(page.getByText("OK", { exact: true })).toHaveCount(0);
  });
});
