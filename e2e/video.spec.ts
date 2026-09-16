import { expect, test } from "@playwright/test";

import { stubEngine, VFR_ROTATED_METADATA } from "./fixtures";

/** The file picker and the engine both travel over the Tauri bridge. */
function importFlow(metadata: unknown) {
  return {
    "plugin:dialog|open": { result: VFR_ROTATED_METADATA.path },
    probe_video: { result: metadata },
  };
}

test.describe("video import", () => {
  test("starts empty rather than implying a clip is loaded", async ({
    page,
  }) => {
    await stubEngine(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();

    await expect(page.getByText(/No clip loaded/)).toBeVisible();
  });

  test("imports a clip and reports what the engine measured", async ({
    page,
  }) => {
    await stubEngine(page, importFlow(VFR_ROTATED_METADATA));
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();
    await page.getByRole("button", { name: /Choose video/ }).click();

    await expect(page.getByText("Frame timing")).toBeVisible();
    await expect(page.getByText("45", { exact: true })).toBeVisible();
    await expect(page.getByText("22.759 fps")).toBeVisible();
  });

  test("shows the variable-rate verdict and the numbers behind it", async ({
    page,
  }) => {
    await stubEngine(page, importFlow(VFR_ROTATED_METADATA));
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();
    await page.getByRole("button", { name: /Choose video/ }).click();

    await expect(page.getByText("Variable", { exact: true })).toBeVisible();
    // Exact, because the same count also appears inside the caveat text below.
    await expect(page.getByText("14 of 44", { exact: true })).toBeVisible();
    await expect(
      page.getByText(/frame_index \/ fps is not valid/),
    ).toBeVisible();
  });

  test("distinguishes stored from display geometry on a rotated clip", async ({
    page,
  }) => {
    await stubEngine(page, importFlow(VFR_ROTATED_METADATA));
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();
    await page.getByRole("button", { name: /Choose video/ }).click();

    await expect(page.getByText("1080×1920")).toBeVisible();
    await expect(page.getByText("1920×1080")).toBeVisible();
    await expect(page.getByText("90° ccw (display_matrix)")).toBeVisible();
  });

  test("renders a rejected file with the engine's remedy", async ({ page }) => {
    await stubEngine(page, {
      "plugin:dialog|open": { result: "/Users/example/music.m4a" },
      probe_video: {
        error: {
          kind: "method",
          message: "music.m4a contains no video stream.",
          code: -31001,
          data: {
            remediation:
              "Select a video recording. Audio-only files and still images cannot be analysed.",
          },
        },
      },
    });
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();
    await page.getByRole("button", { name: /Choose video/ }).click();

    await expect(page.getByText(/contains no video stream/)).toBeVisible();
    await expect(page.getByText(/Audio-only files/)).toBeVisible();
    // The failure must not leave metadata panels from a previous clip on screen.
    await expect(page.getByText("Frame timing")).toHaveCount(0);
  });

  test("switches between screens without losing the loaded clip", async ({
    page,
  }) => {
    await stubEngine(page, importFlow(VFR_ROTATED_METADATA));
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();
    await page.getByRole("button", { name: /Choose video/ }).click();
    await expect(page.getByText("Frame timing")).toBeVisible();

    await page.getByRole("button", { name: "Environment" }).click();
    await expect(page.getByText("Compute")).toBeVisible();

    await page.getByRole("button", { name: "Video" }).click();
    await expect(page.getByText(/No clip loaded/)).toBeVisible();
  });
});
