/**
 * The 3D viewport, against a real decoder.
 *
 * Phase 15's exit criterion is "scrub stays in sync with video", and this file is
 * where that sentence is turned into a measurement rather than a claim.
 *
 * **What is being checked is not that two timelines stay together.** There is
 * only one frame index on the screen: `useFramePlayer` reports the frame the
 * browser says it *painted*, and both the video and the viewport read that. Two
 * clocks can drift; one number cannot. What can still go wrong is everything
 * around it — the viewport indexing its own frames positionally, drawing the
 * nearest frame it has rather than none, or pairing the geometry with a
 * recording it never came from — and each of those produces a picture that looks
 * entirely correct.
 *
 * So the assertions below read the painted frame out of
 * `requestVideoFrameCallback`, independently of the app, and compare it with
 * what the viewport says it drew and with where it actually drew a joint. The
 * clip is the real 45-frame VFR fixture, served with byte ranges, decoded by
 * Chromium — the same arrangement `player.spec.ts` established and for the same
 * reason: the browser is the consumer, so the browser is the thing worth asking.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import type { Locator, Page } from "@playwright/test";
import { expect, test } from "@playwright/test";

import {
  CLIP_URL,
  ONE_PROJECT,
  RECONSTRUCTION_SCENE,
  SCENE_EMPTY_FRAME,
  VFR_CLIP_METADATA,
  VFR_SEEK_INDEX,
  stubEngine,
} from "./fixtures";

const FIXTURE = fileURLToPath(
  new URL(
    "../python/tests/fixtures/video/vfr_30_to_15fps.mp4",
    import.meta.url,
  ),
);

/** Serve the real clip with byte ranges, which a media element needs to seek. */
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

function engine(
  extra: Record<string, { result?: unknown; error?: unknown }> = {},
) {
  return {
    list_projects: { result: ONE_PROJECT },
    reconstruct_scene: { result: RECONSTRUCTION_SCENE },
    choose_clip: { result: VFR_SEEK_INDEX.path },
    probe_video: { result: VFR_CLIP_METADATA },
    seek_index: { result: VFR_SEEK_INDEX },
    ...extra,
  };
}

/** Open the Three-D screen and reconstruct. */
async function reconstruct(
  page: Page,
  extra: Record<string, { result?: unknown; error?: unknown }> = {},
): Promise<Locator> {
  await serveClip(page);
  await stubEngine(page, engine(extra));
  await page.goto("/");
  await page.getByRole("button", { name: "Three-D" }).click();
  await page.getByRole("button", { name: /Reconstruct/ }).click();

  const viewport = page.getByTestId("scene-viewport");
  await expect(viewport).toBeVisible();
  return viewport;
}

/** Reconstruct, then load the reference clip and wait for it to be playable. */
async function withClip(
  page: Page,
): Promise<{ viewport: Locator; video: Locator }> {
  const viewport = await reconstruct(page);
  await page.getByRole("button", { name: /Choose the clip/ }).click();

  const video = page.getByTestId("scene-video");
  await expect(video).toBeVisible();
  // HAVE_ENOUGH_DATA. A seek into an unfetched part of the file paints nothing,
  // so a lower gate would make every assertion depend on how much of a 41 KB
  // clip happened to have arrived.
  await expect
    .poll(() => video.evaluate((node: HTMLVideoElement) => node.readyState), {
      timeout: 15_000,
    })
    .toBe(4);
  await recordPaints(video);
  return { viewport, video };
}

/** Where the standing observer records what the browser has painted. */
declare global {
  interface Window {
    __paints?: number[];
  }
}

/**
 * Record every frame the browser paints, independently of the app.
 *
 * A standing, self-renewing `requestVideoFrameCallback` rather than one promise
 * per seek. The promise form deadlocks here: `evaluate` returning a pending
 * promise occupies the page, and the scrubber click that would cause the paint
 * cannot run until it resolves. A recorder has no such ordering problem — it is
 * installed once, and each seek is measured by the entry that appears after it.
 *
 * `mediaTime` is the only value that reports the presentation time of a frame
 * that was actually painted. `currentTime` is not a substitute: browsers
 * commonly leave it at the value that was requested, so reading it back would
 * confirm the seek against itself.
 */
async function recordPaints(video: Locator): Promise<void> {
  await video.evaluate((node: HTMLVideoElement) => {
    const element = node as HTMLVideoElement & {
      requestVideoFrameCallback: (
        cb: (now: number, meta: { mediaTime: number }) => void,
      ) => number;
    };
    window.__paints = [];
    const tick = (_now: number, meta: { mediaTime: number }) => {
      window.__paints?.push(meta.mediaTime);
      element.requestVideoFrameCallback(tick);
    };
    element.requestVideoFrameCallback(tick);
  });
}

/** Which frame a painted media time falls in. */
function frameOf(mediaTime: number): number {
  // The same one-microsecond allowance `frameAt` makes: Chromium carries media
  // time as whole microseconds, so a frame at 0.13333333… comes back as
  // 0.133333 and a strict comparison reads it as the previous frame.
  let landed = 0;
  VFR_SEEK_INDEX.timestamps_s.forEach((stamp, index) => {
    if (stamp <= mediaTime + 1e-6) landed = index;
  });
  return landed;
}

/**
 * Drive the app's own scrubber to a frame and report what the browser painted.
 *
 * The seek goes through the app's control rather than by setting `currentTime`
 * here, because the path a person uses is the path worth testing — it includes
 * the engine's measured seek targets, which is the thing Phase 14 established
 * a naive map gets wrong on 39 of this clip's 45 frames.
 */
async function scrubTo(
  page: Page,
  video: Locator,
  frame: number,
): Promise<number> {
  const before = await video.evaluate(() => window.__paints?.length ?? 0);
  await page.getByRole("slider", { name: "Frame" }).fill(String(frame));

  await expect
    .poll(() => video.evaluate(() => window.__paints?.length ?? 0), {
      timeout: 5000,
    })
    .toBeGreaterThan(before);

  const painted = await video.evaluate(
    () => window.__paints?.[window.__paints.length - 1] ?? -1,
  );
  return frameOf(painted);
}

test.describe("the viewport and the clip, on one frame", () => {
  test("**draws the frame the browser painted, not the one that was asked for**", async ({
    page,
  }) => {
    const { viewport, video } = await withClip(page);
    await expect(page.getByTestId("scene-clip-matched")).toBeVisible();

    // Every fifth frame, avoiding the last (whose seek target lies past the end
    // of the media — see `player.spec.ts`) and the deliberately empty one.
    for (const target of [4, 9, 14, 24, 29, 34, 39]) {
      const painted = await scrubTo(page, video, target);
      // The player reports what was painted, and the viewport draws that. On
      // this clip they are the same frame, which is the point of the measured
      // seek map; if a seek ever landed elsewhere, the viewport would follow the
      // pixels rather than the request.
      await expect
        .poll(() => viewport.getAttribute("data-frame"))
        .toBe(String(painted));
    }
  });

  test("moves the skeleton by a frame's worth when the frame moves by one", async ({
    page,
  }) => {
    // `data-frame` agreeing is necessary and not sufficient: an attribute can
    // update while the geometry does not. The fixture's body travels one
    // centimetre per frame, which at fx = 1400 over 2.5 m is 5.6 px on screen,
    // so a joint that has not moved has not been redrawn.
    const { viewport, video } = await withClip(page);
    const joint = viewport.locator("[data-landmark='11']").first();

    await scrubTo(page, video, 10);
    await expect.poll(() => viewport.getAttribute("data-frame")).toBe("10");
    const before = Number(await joint.getAttribute("cx"));

    await scrubTo(page, video, 30);
    await expect.poll(() => viewport.getAttribute("data-frame")).toBe("30");
    const after = Number(await joint.getAttribute("cx"));

    // Twenty frames at 1 cm each: 0.2 m at fx = 1400 over 2.5 m is 112 px.
    expect(after - before).toBeGreaterThan(100);
    expect(after - before).toBeLessThan(125);
  });

  test("draws nothing on the frame the reconstruction refused", async ({
    page,
  }) => {
    // The failure this guards against is invisible: a viewport that held the
    // previous pose here would show a body in a position it was never measured
    // in, and it would look like the smoothest part of the swing.
    const { viewport, video } = await withClip(page);
    await scrubTo(page, video, SCENE_EMPTY_FRAME);
    await expect
      .poll(() => viewport.getAttribute("data-frame"))
      .toBe(String(SCENE_EMPTY_FRAME));
    await expect(viewport).toHaveAttribute("data-reconstructed", "0");
    await expect(page.getByTestId("scene-empty")).toBeVisible();
    await expect(viewport.locator("[data-testid='scene-joint']")).toHaveCount(
      0,
    );
  });
});

test.describe("pairing the geometry with the right recording", () => {
  test("refuses a clip the reconstruction was not built from", async ({
    page,
  }) => {
    // The video would play, the skeleton would move and the frame numbers would
    // agree. Nothing about it would look wrong, and the body on screen would
    // have nothing to do with the footage behind it.
    await reconstruct(page, {
      seek_index: {
        result: {
          ...VFR_SEEK_INDEX,
          content_key: {
            ...VFR_SEEK_INDEX.content_key,
            digest: "f".repeat(64),
          },
        },
      },
    });
    await page.getByRole("button", { name: /Choose the clip/ }).click();

    await expect(page.getByTestId("scene-clip-mismatch")).toBeVisible();
    await expect(page.getByTestId("scene-clip-matched")).toHaveCount(0);
  });
});

test.describe("what the viewport says about itself", () => {
  test("opens at the reference camera and says how much it is hiding", async ({
    page,
  }) => {
    const viewport = await reconstruct(page);
    await expect(viewport).toBeVisible();

    // The fixture's uncertainty is a cigar down the reference camera's axis, so
    // the default view hides most of it — and the panel says so instead of
    // letting the picture speak for itself.
    await expect(page.getByTestId("viewpoint-warning")).toBeVisible();
    await expect(page.getByTestId("viewpoint-panel")).toContainText(
      "calibrated as a pair",
    );
  });

  test("stops hiding it once the reader orbits across the error", async ({
    page,
  }) => {
    const viewport = await reconstruct(page);
    await expect(page.getByTestId("viewpoint-warning")).toBeVisible();

    const box = await viewport.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;

    // A quarter turn: 225 px at 0.4 degrees per pixel.
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 225, box.y + box.height / 2, {
      steps: 10,
    });
    await page.mouse.up();

    await expect(page.getByTestId("viewpoint-warning")).toHaveCount(0);
    // And the error itself has not changed — only whether it can be seen.
    await expect(page.getByTestId("viewpoint-panel")).toContainText("20.0 mm");
  });

  test("reconstructs without a clip, and scrubs on its own", async ({
    page,
  }) => {
    // The 3D view is worth looking at before the reference clip has been picked.
    // Requiring the video first would make it conditional on a permission grant
    // it does not use.
    const viewport = await reconstruct(page);
    await expect(viewport).toHaveAttribute("data-frame", "0");
    await page.getByRole("slider", { name: "Frame" }).fill("12");
    await expect(viewport).toHaveAttribute("data-frame", "12");
  });
});
