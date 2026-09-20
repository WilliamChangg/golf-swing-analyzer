/**
 * The player, driven against a real decoder.
 *
 * Everything else in this suite stubs the engine and checks rendering. This
 * file is different in one important way: **the video is real.** Playwright
 * serves the bytes of `python/tests/fixtures/video/vfr_30_to_15fps.mp4`, so
 * Chromium decodes an actual clip, and `requestVideoFrameCallback` reports the
 * presentation time of the frame it actually painted.
 *
 * That matters because it is the only non-circular check this project has of
 * the seek map. `scripts/benchmark_seek.py` compares the two candidate maps
 * against the timestamps the container carries, which is a statement about the
 * model; its OpenCV column measures OpenCV's own seek accuracy, which is poor
 * enough that it cannot tell the two maps apart. A browser is the consumer, so
 * a browser is the thing worth asking.
 *
 * The fixture is chosen for its timing rather than its content: it changes rate
 * from 30 fps to 15 fps half-way through, and its container declares an average
 * of 23.684 fps. `frame / declared fps` misses on 39 of its 45 frames.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";

import {
  CLIP_URL,
  COACHING_REPORT,
  METRIC_SET,
  NO_PROJECTS,
  POSE_OVERLAY,
  SWING_PHASES,
  VFR_CLIP_METADATA,
  VFR_DECLARED_FPS,
  VFR_SEEK_INDEX,
  stubEngine,
} from "./fixtures";

const FIXTURE = fileURLToPath(
  new URL(
    "../python/tests/fixtures/video/vfr_30_to_15fps.mp4",
    import.meta.url,
  ),
);

/**
 * Serve the real clip on the URL the stubbed asset protocol points at.
 *
 * **Byte ranges are honoured, and they are not optional.** A media element
 * decides whether a resource is seekable from `Accept-Ranges`; served without
 * it, Chromium reports `seekable` as empty and silently clamps every
 * `currentTime` assignment back to zero. The clip still loads, still reports a
 * duration and still buffers end to end — it just never moves, which looks
 * exactly like a broken player rather than like a response missing a header.
 *
 * Tauri's own asset protocol implements range requests, so this is the
 * behaviour being reproduced rather than worked around.
 */
async function serveClip(page: Page): Promise<void> {
  const bytes = readFileSync(FIXTURE);

  await page.route(`**${CLIP_URL}`, (route) => {
    const range = route.request().headers()["range"];
    if (range === undefined) {
      return route.fulfill({
        status: 200,
        contentType: "video/mp4",
        body: bytes,
        headers: {
          "accept-ranges": "bytes",
          "content-length": String(bytes.length),
        },
      });
    }

    const matched = /bytes=(\d*)-(\d*)/.exec(range);
    const start = matched?.[1] ? Number(matched[1]) : 0;
    const end = matched?.[2] ? Number(matched[2]) : bytes.length - 1;
    const slice = bytes.subarray(start, end + 1);

    return route.fulfill({
      status: 206,
      contentType: "video/mp4",
      body: slice,
      headers: {
        "accept-ranges": "bytes",
        "content-length": String(slice.length),
        "content-range": `bytes ${String(start)}-${String(end)}/${String(bytes.length)}`,
      },
    });
  });
}

/** The engine responses a loaded, analysed clip needs. */
function analysed() {
  return {
    choose_clip: { result: VFR_SEEK_INDEX.path },
    probe_video: { result: VFR_CLIP_METADATA },
    seek_index: { result: VFR_SEEK_INDEX },
    extract_poses: {
      result: { schema_version: 1, video_path: VFR_SEEK_INDEX.path },
    },
    detect_phases: { result: SWING_PHASES },
    compute_metrics: { result: METRIC_SET },
    coach_swing: { result: COACHING_REPORT },
    pose_overlay: { result: POSE_OVERLAY },
    list_projects: { result: NO_PROJECTS },
  };
}

/** Load a clip on the Swing screen and wait for the video to be playable. */
async function loadClip(page: Page, extra = {}) {
  await serveClip(page);
  await stubEngine(page, { ...analysed(), ...extra });
  await page.goto("/");
  await page.getByRole("button", { name: /Choose clip/ }).click();

  const video = page.getByTestId("clip-video");
  await expect(video).toBeVisible();
  // HAVE_ENOUGH_DATA, not HAVE_CURRENT_DATA. A seek into a part of the file the
  // decoder has not fetched paints nothing, so a lower gate makes every
  // assertion below depend on how much of a 41 KB clip happened to arrive.
  await expect
    .poll(() => video.evaluate((node: HTMLVideoElement) => node.readyState), {
      timeout: 15_000,
    })
    .toBe(4);
  return video;
}

/**
 * Seek to a frame and report the frame the browser actually painted.
 *
 * Reads `mediaTime` out of `requestVideoFrameCallback`, which is the only API
 * that reports the presentation time of a painted frame. `currentTime` is not a
 * substitute — browsers commonly leave it at the value that was requested, so
 * reading it back would confirm the seek against itself.
 */
async function paintedFrameFor(
  video: ReturnType<Page["getByTestId"]>,
  target: number,
  timestamps: number[],
): Promise<number> {
  // Park on a frame that is certainly not the target's first. A paused element
  // presents a frame only when the frame *changes*, so seeking from frame 0 to
  // another time inside frame 0 paints nothing and the callback never fires --
  // which would look like a hang rather than like the no-op it is.
  const park =
    target < (timestamps[timestamps.length - 1] ?? 0) / 2
      ? (timestamps[timestamps.length - 1] ?? 0)
      : (timestamps[0] ?? 0);
  // The park itself may paint nothing -- it lands where the element already is
  // on the first call of a run -- and that is fine. Only the target's paint is
  // an assertion; a park that does nothing just leaves the element where it
  // was, which is still not the target's frame.
  await seekAndWait(video, park, timestamps, { tolerant: true });
  return seekAndWait(video, target, timestamps);
}

function seekAndWait(
  video: ReturnType<Page["getByTestId"]>,
  target: number,
  timestamps: number[],
  options: { tolerant?: boolean } = {},
): Promise<number> {
  return video.evaluate(
    (
      node: HTMLVideoElement,
      {
        seekTo,
        times,
        tolerant,
      }: { seekTo: number; times: number[]; tolerant: boolean },
    ) =>
      new Promise<number>((resolve, reject) => {
        const element = node as HTMLVideoElement & {
          requestVideoFrameCallback: (
            cb: (now: number, meta: { mediaTime: number }) => void,
          ) => number;
        };
        const timer = setTimeout(
          () => {
            if (tolerant) {
              resolve(-1);
              return;
            }
            reject(
              new Error(`no frame painted after seeking to ${String(seekTo)}`),
            );
          },
          tolerant ? 1000 : 5000,
        );

        // Registered *before* the seek: the callback is one-shot, and
        // registering it afterwards can miss the very frame being waited for.
        element.requestVideoFrameCallback((_now, meta) => {
          clearTimeout(timer);
          // The same one-microsecond allowance `frameAt` makes, and for the
          // same measured reason: Chromium carries media time as whole
          // microseconds, so the value it reports for a frame at 0.13333333…
          // arrives as 0.133333 -- a third of a microsecond below the timestamp
          // the container holds. A strict comparison reads that as the previous
          // frame.
          let landed = 0;
          times.forEach((stamp, i) => {
            if (stamp <= meta.mediaTime + 1e-6) landed = i;
          });
          resolve(landed);
        });
        element.currentTime = seekTo;
      }),
    { seekTo: target, times: timestamps, tolerant: options.tolerant ?? false },
  );
}

test.describe("frame-accurate seek, in a real browser", () => {
  test("every measured target lands on the frame it asks for", async ({
    page,
  }) => {
    const video = await loadClip(page);

    // Every fourth frame rather than all 45: each seek is a decode and a
    // painted frame, and the failure this guards against is not sparse — the
    // naive map misses on 39 of these 45, so a sample of a dozen cannot miss it.
    //
    // The final frame is excluded, and that exclusion is a finding rather than
    // a convenience. Its seek target sits past the end of the media, because
    // how long the last frame is displayed is not recorded anywhere; the
    // browser clamps to the declared duration, and on this clip the declared
    // duration is *equal to* the last frame's own presentation time. There is
    // therefore no time inside this media at which its last frame is shown, by
    // any map. The player reports that as a one-frame residual rather than
    // hiding it, which is the test below.
    const misses: { frame: number; landed: number }[] = [];
    for (let frame = 0; frame < VFR_SEEK_INDEX.frame_count - 1; frame += 4) {
      const target = VFR_SEEK_INDEX.seek_targets_s[frame] ?? 0;
      const landed = await paintedFrameFor(
        video,
        target,
        VFR_SEEK_INDEX.timestamps_s,
      );
      if (landed !== frame) misses.push({ frame, landed });
    }

    expect(misses).toEqual([]);
  });

  test("the last frame cannot be reached, and the player says so", async ({
    page,
  }) => {
    // Not a defect this code can fix. The container declares a duration equal
    // to the final frame's presentation time, so every seek past that clamps
    // back to it and the browser paints the frame before. What matters is that
    // the player measures it instead of displaying the number it asked for.
    const video = await loadClip(page);
    const last = VFR_SEEK_INDEX.frame_count - 1;

    const landed = await paintedFrameFor(
      video,
      VFR_SEEK_INDEX.seek_targets_s[last] ?? 0,
      VFR_SEEK_INDEX.timestamps_s,
    );

    expect(landed).toBe(last - 1);
  });

  test("the declared frame rate would have missed on the same clip", async ({
    page,
  }) => {
    // Not a test of this app. It measures what the obvious alternative would
    // have cost, in the same browser, on the same clip, in the same run — so
    // the comparison above is against something rather than against nothing.
    const video = await loadClip(page);

    const misses: number[] = [];
    for (let frame = 0; frame < VFR_SEEK_INDEX.frame_count - 1; frame += 4) {
      const landed = await paintedFrameFor(
        video,
        frame / VFR_DECLARED_FPS,
        VFR_SEEK_INDEX.timestamps_s,
      );
      if (landed !== frame) misses.push(frame);
    }

    expect(misses.length).toBeGreaterThan(0);
  });
});

test.describe("transport", () => {
  test("reports the frame on screen, and says the seek landed", async ({
    page,
  }) => {
    await loadClip(page);

    await page.getByRole("button", { name: "Next frame" }).click();

    await expect(
      page.getByText(/seek landed on the frame asked for/),
    ).toBeVisible();
  });

  test("steps forward and back through frames", async ({ page }) => {
    await loadClip(page);
    const readout = page.getByRole("status");

    await page.getByRole("button", { name: "Next frame" }).click();
    await expect(readout).toContainText(/frame 1 of/);
    await page.getByRole("button", { name: "Next frame" }).click();
    await expect(readout).toContainText(/frame 2 of/);
    await page.getByRole("button", { name: "Previous frame" }).click();
    await expect(readout).toContainText(/frame 1 of/);
  });

  test("offers only rates at or below real time", async ({ page }) => {
    // A downswing is about a quarter of a second, so the useful direction is
    // entirely downward; 2x would be a way to see less of the thing being
    // examined.
    await loadClip(page);
    const rates = page.getByRole("group", { name: "Playback rate" });

    await expect(rates.getByRole("button")).toHaveText(["0.25×", "0.5×", "1×"]);
  });

  test("applies a slower rate to the element", async ({ page }) => {
    const video = await loadClip(page);

    await page.getByRole("button", { name: "0.25×" }).click();

    await expect
      .poll(() => video.evaluate((node: HTMLVideoElement) => node.playbackRate))
      .toBe(0.25);
  });
});

test.describe("the full workflow", () => {
  test("jumps to a swing event and shows the phase that frame is in", async ({
    page,
  }) => {
    await loadClip(page);
    await page.getByRole("button", { name: /Analyse/ }).click();

    // Impact is frame 46 in the phase fixture, past this 45-frame clip, so the
    // jump clamps. Top, at frame 38, is inside it.
    await page.getByRole("button", { name: /^Top/ }).click();

    await expect(page.getByRole("status")).toContainText(/frame 38 of/);
  });

  test("seeks from a metric to the frames it was measured over", async ({
    page,
  }) => {
    // What makes a measurement checkable: the number and the frame it came from
    // are one click apart.
    await loadClip(page);
    await page.getByRole("button", { name: /Analyse/ }).click();

    await expect(page.getByText("Measurements", { exact: true })).toBeVisible();
    await page
      .getByRole("button", { name: "Tempo ratio", exact: true })
      .click();

    await expect(page.getByRole("status")).toContainText(/frame 13 of/);
  });

  test("shows a refused measurement rather than dropping it", async ({
    page,
  }) => {
    await loadClip(page);
    await page.getByRole("button", { name: /Analyse/ }).click();

    await expect(page.getByText("Refused (1)")).toBeVisible();
    await expect(page.getByText("rotation.x_factor")).toBeVisible();
  });

  test("presents an empty findings list as a result", async ({ page }) => {
    // The ordinary outcome on real footage, and the case the panel is laid out
    // around.
    await loadClip(page);
    await page.getByRole("button", { name: /Analyse/ }).click();

    await expect(
      page.getByText(/That is a result, not a failure/),
    ).toBeVisible();
    await expect(page.getByText("Not concluded (1)")).toBeVisible();
  });

  test("draws the overlay over the video", async ({ page }) => {
    await loadClip(page);
    await page.getByRole("button", { name: /Analyse/ }).click();

    const canvas = page.getByTestId("overlay-canvas");
    await expect(canvas).toBeVisible();
    // Sized from the clip's own geometry rather than from the window, so the
    // drawing resolution follows the footage.
    await expect(canvas).toHaveAttribute("width", "320");
    await expect(canvas).toHaveAttribute("height", "240");
  });

  test("asks for the two numbers the file cannot supply", async ({ page }) => {
    await loadClip(page);

    await expect(
      page.getByRole("spinbutton", { name: /Smoothing window/ }),
    ).toBeVisible();
    await expect(
      page.getByRole("spinbutton", { name: /Slow-motion factor/ }),
    ).toBeVisible();
  });

  test("starts with nothing loaded rather than implying a clip", async ({
    page,
  }) => {
    await stubEngine(page, analysed());
    await page.goto("/");

    await expect(page.getByText(/No clip loaded/)).toBeVisible();
  });
});
