import { describe, expect, it } from "vitest";

import { orbitCamera, referenceCamera, vec } from "@/features/scene/projection";
import {
  EMPTY_FRAME,
  FIRST_FRAME,
  FRAME_COUNT,
  PARTIAL_FRAME,
  matchingSeekIndex,
  sceneFixture,
} from "@/features/scene/scene.fixture";
import {
  defaultDistance,
  frameAt,
  refusalsIn,
  sceneMatchesClip,
  viewpointHonesty,
} from "@/features/scene/scene";

/**
 * Reading a scene: which frame, whose clip, and how honest the view of it is.
 *
 * Three subjects, and two of them are refusals. That is not defensiveness — each
 * is a failure that produces a *plausible picture* rather than an error, which
 * is the class of bug this whole project is built to make visible.
 */

describe("frameAt", () => {
  const scene = sceneFixture();

  it("finds a frame by the reference clip's own index", () => {
    // The fixture starts at frame 10, so a viewport indexing positionally would
    // return frame 10's body for a request of frame 0 and be four frames out on
    // every windowed scene.
    expect(frameAt(scene, FIRST_FRAME)?.frame_index).toBe(FIRST_FRAME);
    expect(frameAt(scene, FIRST_FRAME + 3)?.frame_index).toBe(FIRST_FRAME + 3);
  });

  it("returns null outside the range rather than the nearest frame", () => {
    expect(frameAt(scene, 0)).toBeNull();
    expect(frameAt(scene, FIRST_FRAME - 1)).toBeNull();
    expect(frameAt(scene, FIRST_FRAME + FRAME_COUNT)).toBeNull();
  });

  it("does not trust the offset when the frame disagrees with it", () => {
    // A scene with a gap in it would otherwise draw one frame's body under
    // another frame's number, silently. Falling back to a search is cheap; being
    // silently wrong is not.
    const gapped = sceneFixture({
      frames: scene.frames.map((frame, index) =>
        index === 2
          ? { ...frame, frame_index: frame.frame_index + 100 }
          : frame,
      ),
    });
    expect(frameAt(gapped, FIRST_FRAME + 2)).toBeNull();
    expect(frameAt(gapped, FIRST_FRAME + 102)?.frame_index).toBe(
      FIRST_FRAME + 102,
    );
  });
});

describe("sceneMatchesClip", () => {
  const scene = sceneFixture();

  it("matches the clip the reconstruction was built from", () => {
    expect(sceneMatchesClip(scene, matchingSeekIndex())).toBe(true);
  });

  it("matches by content rather than by path", () => {
    // A clip that has been moved or renamed is still the same recording. That
    // is the rule the project store has used for clip identity since Phase 7.
    expect(
      sceneMatchesClip(
        scene,
        matchingSeekIndex({ path: "/somewhere/else.mp4" }),
      ),
    ).toBe(true);
  });

  it("refuses a different recording at the same path", () => {
    const other = matchingSeekIndex();
    expect(
      sceneMatchesClip(scene, {
        ...other,
        content_key: { ...other.content_key, digest: "c".repeat(64) },
      }),
    ).toBe(false);
  });

  it("refuses a digest taken under a different algorithm", () => {
    // Two digests of the same bytes under different schemes are different
    // strings, and comparing them without the algorithm would be comparing
    // whether two hash functions happened to collide.
    const other = matchingSeekIndex();
    expect(
      sceneMatchesClip(scene, {
        ...other,
        content_key: { ...other.content_key, algorithm: "sha256" },
      }),
    ).toBe(false);
  });

  it("refuses when no clip is loaded at all", () => {
    expect(sceneMatchesClip(scene, null)).toBe(false);
  });
});

describe("viewpointHonesty", () => {
  const scene = sceneFixture();
  const frame = frameAt(scene, FIRST_FRAME);

  it("reports what the measurement is worth, whatever the view", () => {
    const reference = referenceCamera(scene);
    expect(frame).not.toBeNull();
    if (!frame) return;
    const honesty = viewpointHonesty(frame, reference);
    expect(honesty.drawn).toBe(4);
    // The fixture's points all carry a 20 mm worst-direction sigma.
    expect(honesty.sigmaM).toBeCloseTo(0.02, 6);
  });

  it("**shows almost none of it from the reference camera**", () => {
    // Phase 15's finding, in miniature. The fixture's uncertainty is a cigar
    // along z, and the reference camera looks down z — so the long axis points
    // at the reader and the picture is at its most confident exactly where the
    // measurement is at its worst.
    //
    // Straight down the axis the fraction would be sqrt(1e-5 / 4e-4) = 0.158;
    // these joints sit a little off it, which tilts a fraction of the long axis
    // into view and is why the answer is 0.19 rather than 0.158. That is the
    // real behaviour — a viewer at the edge of the frame sees slightly more —
    // and the test pins it rather than the idealised number.
    expect(frame).not.toBeNull();
    if (!frame) return;
    const honesty = viewpointHonesty(frame, referenceCamera(scene));
    expect(honesty.visible).toBeCloseTo(0.19, 2);
    expect(honesty.apparentSigmaM).toBeCloseTo(0.02 * 0.19, 3);
  });

  it("shows all of it from a viewpoint across the long axis", () => {
    expect(frame).not.toBeNull();
    if (!frame) return;
    const sideOn = orbitCamera(
      scene,
      { azimuthDeg: 90, elevationDeg: 0, distance: 2.5 },
      45,
    );
    const honesty = viewpointHonesty(frame, sideOn);
    expect(honesty.visible).toBeGreaterThan(0.99);
    // The uncertainty has not changed — only whether it can be seen.
    expect(honesty.sigmaM).toBeCloseTo(0.02, 6);
    expect(honesty.apparentSigmaM).toBeCloseTo(0.02, 4);
  });

  it("reports nothing visible when there is no camera", () => {
    expect(frame).not.toBeNull();
    if (!frame) return;
    const honesty = viewpointHonesty(frame, null);
    expect(honesty.visible).toBeNull();
    expect(honesty.sigmaM).toBeCloseTo(0.02, 6);
  });

  it("describes a frame with nothing in it as having nothing in it", () => {
    const empty = frameAt(scene, EMPTY_FRAME);
    expect(empty).not.toBeNull();
    if (!empty) return;
    const honesty = viewpointHonesty(empty, referenceCamera(scene));
    expect(honesty.drawn).toBe(0);
    expect(honesty.sigmaM).toBeNull();
    expect(honesty.visible).toBeNull();
  });
});

describe("refusalsIn", () => {
  const scene = sceneFixture();

  it("counts nothing on a complete frame", () => {
    expect(refusalsIn(frameAt(scene, FIRST_FRAME)!).size).toBe(0);
  });

  it("names the reason a single landmark is missing", () => {
    const counts = refusalsIn(frameAt(scene, PARTIAL_FRAME)!);
    // The reason, not a count of failures: the fix for a shallow intersection is
    // to move a camera, and nothing else in the five has that fix.
    expect(counts.get("ill_conditioned")).toBe(1);
    expect(counts.size).toBe(1);
  });

  it("counts every landmark on a frame that produced none", () => {
    expect(refusalsIn(frameAt(scene, EMPTY_FRAME)!).get("not_seen")).toBe(4);
  });
});

describe("defaultDistance", () => {
  it("opens where the camera that filmed it stood", () => {
    const scene = sceneFixture();
    // A viewport opening at a chosen distance would frame every capture the same
    // way and tell a reader nothing about how far off the subject was.
    expect(defaultDistance(scene)).toBeCloseTo(2.502, 2);
  });

  it("never goes inside the body it is framing", () => {
    const near = sceneFixture({
      centroid: vec(0, 0, 0.05),
      radius_m: 3,
    });
    expect(defaultDistance(near)).toBeGreaterThanOrEqual(3 * 1.2);
  });
});
