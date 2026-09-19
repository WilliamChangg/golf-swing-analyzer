/**
 * A small hand-built scene, for the tests that are about presentation.
 *
 * Deliberately **not** a real reconstruction. `python/tests/test_scene.py`
 * checks the engine's output against a body whose 3D positions are inputs, and
 * `projection-truth.json` pins this app's projection to the engine's own pixels.
 * What is left for these tests is what the app does with a scene — which frame
 * it draws, what it does with a hole in one, and whether it can tell that the
 * video beside it is the right recording — and for that a scene with four
 * landmarks and a known shape is far easier to assert against than a megabyte of
 * real geometry.
 *
 * Three things about it are load-bearing:
 *
 * * **It does not start at frame zero.** A viewport that indexed positionally
 *   would pass every test on a scene that did, and draw the wrong body on every
 *   windowed one.
 * * **One frame refuses a single landmark**, so a partial figure has to keep
 *   drawing the rest and drop only the bones that lost an end.
 * * **One frame refuses everything**, which is a real frame with no body in it —
 *   the case a viewport must render as empty rather than by holding the previous
 *   pose.
 *
 * It lives beside the tests rather than inside one so the viewport and the
 * screen can assert against the same scene. Nothing in the app imports it.
 */

import type {
  Landmark,
  ReconstructionScene,
  SceneCamera,
  SceneFrame,
  ScenePoint,
  SeekIndex,
} from "@gsa/types";

/** Shoulders and elbows: enough for two bones and a left/right asymmetry. */
export const FIXTURE_LANDMARKS: Landmark[] = [11, 12, 13, 14];

/** The scene's own frame range, chosen not to start at zero. */
export const FIRST_FRAME = 10;
export const FRAME_COUNT = 6;

/** The frame where one landmark is refused. */
export const PARTIAL_FRAME = 12;
/**
 * The frame where everything is refused.
 *
 * Placed with frames on both sides of it, so the trajectory through it breaks
 * into two runs of more than one point each. A break at the end would leave a
 * single orphaned point, which is not a polyline and would make the "does it
 * draw across the gap" assertion pass for the wrong reason.
 */
export const EMPTY_FRAME = 13;

export const FIXTURE_DIGEST = "b".repeat(64);

function point(landmark: Landmark, frame: number): ScenePoint {
  const refusedEntirely = frame === EMPTY_FRAME;
  const refusedOne = frame === PARTIAL_FRAME && landmark === 14;
  if (refusedEntirely || refusedOne) {
    return {
      landmark,
      position: null,
      refused: refusedEntirely ? "not_seen" : "ill_conditioned",
      uncertainty: null,
      convergence_deg: null,
      reprojection_px: null,
      visibility: 0.2,
    };
  }

  // A body two and a half metres down the axis, drifting sideways with the
  // frame so a viewport drawing the wrong frame lands somewhere visibly wrong.
  const side = landmark % 2 === 1 ? -0.2 : 0.2;
  const height = landmark < 13 ? -0.25 : 0.05;
  return {
    landmark,
    position: {
      x: side + (frame - FIRST_FRAME) * 0.05,
      y: height,
      z: 2.5,
    },
    refused: null,
    uncertainty: {
      // A cigar along z: small across the reference camera's view, long down
      // it. sqrt(4e-4) = 20 mm, sqrt(1e-5) ~ 3.2 mm.
      sigma_m: 0.02,
      xx: 1e-5,
      yy: 1e-5,
      zz: 4e-4,
      xy: 0,
      xz: 0,
      yz: 0,
    },
    convergence_deg: 88,
    reprojection_px: 0.9,
    visibility: 0.95,
  };
}

function frame(index: number): SceneFrame {
  const points = FIXTURE_LANDMARKS.map((landmark) => point(landmark, index));
  return {
    frame_index: index,
    timestamp_s: index / 120,
    points,
    reconstructed: points.filter((entry) => entry.position !== null).length,
  };
}

const REFERENCE: SceneCamera = {
  kind: "reference",
  role: "face_on",
  name: "face_on.mp4",
  position: { x: 0, y: 0, z: 0 },
  forward: { x: 0, y: 0, z: 1 },
  up: { x: 0, y: -1, z: 0 },
  right: { x: 1, y: 0, z: 0 },
  fx: 1400,
  fy: 1400,
  cx: 960,
  cy: 540,
  image_width: 1920,
  image_height: 1080,
  horizontal_fov_deg: 69,
};

const TARGET: SceneCamera = {
  kind: "target",
  role: "down_the_line",
  name: "dtl.mp4",
  // A right angle round from the reference, which is what the capture protocol
  // asks for and what the fixture's near-spherical conditioning implies.
  position: { x: -2.5, y: 0, z: 2.5 },
  forward: { x: 1, y: 0, z: 0 },
  up: { x: 0, y: -1, z: 0 },
  right: { x: 0, y: 0, z: -1 },
  fx: 1500,
  fy: 1500,
  cx: 960,
  cy: 540,
  image_width: 1920,
  image_height: 1080,
  horizontal_fov_deg: 65,
};

export function sceneFixture(
  overrides: Partial<ReconstructionScene> = {},
): ReconstructionScene {
  const frames = Array.from({ length: FRAME_COUNT }, (_, offset) =>
    frame(FIRST_FRAME + offset),
  );

  return {
    schema_version: 1,
    space: "camera",
    reference_content_key: {
      algorithm: "sha256-sampled-v1",
      digest: FIXTURE_DIGEST,
      size_bytes: 4096,
    },
    reference_name: "face_on.mp4",
    target_name: "dtl.mp4",
    reference_role: "face_on",
    target_role: "down_the_line",
    start_frame: FIRST_FRAME,
    end_frame: FIRST_FRAME + FRAME_COUNT,
    frames,
    landmarks: FIXTURE_LANDMARKS,
    connections: [
      [11, 12],
      [11, 13],
      [12, 14],
    ],
    trajectories: [
      {
        landmark: 13,
        name: "left_elbow",
        // A break in the middle, which a polyline must not draw across.
        points: frames.map((entry) =>
          entry.frame_index === EMPTY_FRAME
            ? null
            : (entry.points.find((p) => p.landmark === 13)?.position ?? null),
        ),
        reconstructed: FRAME_COUNT - 1,
      },
    ],
    cameras: [REFERENCE, TARGET],
    calibration: "stereo",
    centroid: { x: 0, y: -0.1, z: 2.5 },
    radius_m: 0.4,
    slow_motion_factor: 1,
    pixel_sigma_px: 2.29,
    report: {
      schema_version: 1,
      reconstructed: true,
      space: "camera",
      reference_role: "face_on",
      target_role: "down_the_line",
      reference_name: "face_on.mp4",
      target_name: "dtl.mp4",
      frames: FRAME_COUNT,
      reconstructed_frames: FRAME_COUNT - 1,
      slow_motion_factor: 1,
      calibration: "stereo",
      baseline_m: 3.54,
      convergence_deg: 90,
      quality: {
        points_attempted: FRAME_COUNT * FIXTURE_LANDMARKS.length,
        points_reconstructed: FRAME_COUNT * FIXTURE_LANDMARKS.length - 5,
        coverage: 0.79,
        median_reprojection_px: 0.9,
        max_reprojection_px: 2.4,
        median_convergence_deg: 88,
        min_convergence_deg: 71,
        median_uncertainty_m: 0.02,
        p95_uncertainty_m: 0.031,
        bones: [],
        symmetry: [],
        worst_bone_variation: 0.04,
        pixel_sigma_px: 2.29,
        methodology: "DLT, refined on reprojection error in both views",
      },
      pairing: null,
      landmarks: [],
      reconstructed_at: null,
      config: {},
      refusal: null,
      warnings: [],
    },
    warnings: [],
    ...overrides,
  };
}

/** A seek index whose content key matches the fixture scene's. */
export function matchingSeekIndex(
  overrides: Partial<SeekIndex> = {},
): SeekIndex {
  const timestamps = Array.from(
    { length: FIRST_FRAME + FRAME_COUNT },
    (_, index) => index / 120,
  );
  return {
    schema_version: 1,
    path: "/data/face_on.mp4",
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest: FIXTURE_DIGEST,
      size_bytes: 4096,
    },
    source: "decoded_frames",
    frame_count: timestamps.length,
    timestamps_s: timestamps,
    seek_targets_s: timestamps.map((value) => value + 1 / 240),
    last_interval_s: 1 / 120,
    warnings: [],
    ...overrides,
  };
}
