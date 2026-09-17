import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";

import {
  NO_SWING,
  POSE_RESULT,
  SWING_PHASES,
  VFR_ROTATED_METADATA,
  stubEngine,
} from "./fixtures";

/** The file picker and the engine both travel over the Tauri bridge. */
function importFlow(metadata: unknown) {
  return {
    "plugin:dialog|open": { result: VFR_ROTATED_METADATA.path },
    probe_video: { result: metadata },
  };
}

/** Load a clip, then hand back a page sitting on its metadata. */
async function loadClip(page: Page, extra = {}) {
  await stubEngine(page, { ...importFlow(VFR_ROTATED_METADATA), ...extra });
  await page.goto("/");
  await page.getByRole("button", { name: "Video" }).click();
  await page.getByRole("button", { name: /Choose video/ }).click();
  await expect(page.getByText("Frame timing")).toBeVisible();
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

test.describe("pose extraction", () => {
  test("offers extraction only once a clip is loaded", async ({ page }) => {
    await stubEngine(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Video" }).click();

    await expect(
      page.getByRole("button", { name: /Extract pose/ }),
    ).toHaveCount(0);
  });

  test("starts with nothing extracted", async ({ page }) => {
    await loadClip(page);
    await expect(page.getByText(/No landmarks extracted/)).toBeVisible();
  });

  test("reports what the engine measured", async ({ page }) => {
    await loadClip(page, { extract_poses: { result: POSE_RESULT } });
    await page.getByRole("button", { name: /Extract pose/ }).click();

    await expect(page.getByText("44 (97.8%)")).toBeVisible();
    await expect(page.getByText("15.3 ms")).toBeVisible();
    await expect(
      page.getByText("pose_landmarker_full (float16)"),
    ).toBeVisible();
  });

  test("renders a failed extraction with the engine's remedy", async ({
    page,
  }) => {
    await loadClip(page, {
      extract_poses: {
        error: {
          kind: "method",
          message: "Model 'pose_landmarker_heavy' is not downloaded.",
          code: -31001,
          data: { remediation: "Run `python scripts/download_models.py`." },
        },
      },
    });
    await page.getByRole("button", { name: /Extract pose/ }).click();

    await expect(page.getByText(/is not downloaded/)).toBeVisible();
    await expect(page.getByText(/download_models\.py/)).toBeVisible();
    // A failure must not leave a previous run's numbers on screen.
    await expect(page.getByText("Frames processed")).toHaveCount(0);
  });

  test("surfaces a warning about an unusable extraction", async ({ page }) => {
    await loadClip(page, {
      extract_poses: {
        result: {
          ...POSE_RESULT,
          stats: {
            ...POSE_RESULT.stats,
            frames_detected: 0,
            detection_rate: 0,
            mean_visibility: null,
          },
          warnings: [
            "No pose was detected in any frame. Nothing downstream can be computed from this extraction.",
          ],
        },
      },
    });
    await page.getByRole("button", { name: /Extract pose/ }).click();

    await expect(page.getByText(/No pose was detected/)).toBeVisible();
    await expect(page.getByText("not measured")).toBeVisible();
  });
});

test.describe("swing phase detection", () => {
  test("detects nothing until asked", async ({ page }) => {
    await loadClip(page);
    await expect(page.getByText("Nothing detected yet.")).toBeVisible();
  });

  test("shows the located events and their confidence factors", async ({
    page,
  }) => {
    await loadClip(page, { detect_phases: { result: SWING_PHASES } });
    await page.getByRole("button", { name: /Detect swing/ }).click();

    await expect(page.getByText("Swing detected.")).toBeVisible();
    const events = page.getByRole("table");
    await expect(events.getByText("Impact")).toBeVisible();
    await expect(events.getByText("1.600s")).toBeVisible();
    // The truncated finish keeps its frame and loses its confidence.
    await expect(events.getByText("Finish")).toBeVisible();
  });

  test("steps frame by frame across a phase boundary", async ({ page }) => {
    // The whole point of the inspector: a frame number for an event is only
    // checkable by watching where the phase actually changes.
    await loadClip(page, { detect_phases: { result: SWING_PHASES } });
    await page.getByRole("button", { name: /Detect swing/ }).click();

    await page.getByRole("button", { name: "Top", exact: true }).click();
    const status = page.getByRole("status");
    await expect(status).toContainText("Downswing");

    await page.getByRole("button", { name: "Previous frame" }).click();
    await expect(status).toContainText("Backswing");
    await expect(status).toContainText("frame 37");
  });

  test("opens the inspector on impact", async ({ page }) => {
    await loadClip(page, { detect_phases: { result: SWING_PHASES } });
    await page.getByRole("button", { name: /Detect swing/ }).click();

    await expect(page.getByLabel("Frame", { exact: true })).toHaveValue("48");
  });

  test("reports a clip with no swing as a finding rather than a failure", async ({
    page,
  }) => {
    await loadClip(page, { detect_phases: { result: NO_SWING } });
    await page.getByRole("button", { name: /Detect swing/ }).click();

    await expect(
      page.getByText("No swing detected in this clip."),
    ).toBeVisible();
    // The measurement that decided it, and the reason, are both on screen.
    // Scoped to the field, because the same figure appears in the explanation.
    await expect(page.locator('dd[title="0.04 torso lengths"]')).toBeVisible();
    await expect(
      page.getByText(/below the 0.5 a swing requires/),
    ).toBeVisible();
    // No inspector, because there is nothing to inspect.
    await expect(page.getByLabel("Frame", { exact: true })).toHaveCount(0);
  });

  test("renders a missing extraction with the engine's remedy", async ({
    page,
  }) => {
    await loadClip(page, {
      detect_phases: {
        error: {
          kind: "method",
          message:
            "No extracted poses for faceon.mov with model 'pose_landmarker_full'.",
          code: -31001,
          data: {
            remediation:
              "Run: analyzer extract faceon.mov --model pose_landmarker_full",
          },
        },
      },
    });
    await page.getByRole("button", { name: /Detect swing/ }).click();

    await expect(page.getByText(/No extracted poses/)).toBeVisible();
    await expect(page.getByText(/analyzer extract/)).toBeVisible();
  });
});
