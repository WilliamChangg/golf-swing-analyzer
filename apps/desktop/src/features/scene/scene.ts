/**
 * Reading a scene: which frame, which points, and whether it belongs to this clip.
 *
 * Small, and two of the three functions here exist to refuse something.
 *
 * A reconstruction is indexed by the **reference clip's** frames, and a video
 * element plays whatever file it was pointed at. Nothing about a 3D scene makes
 * it obvious that those are the same recording — so `sceneMatchesClip` compares
 * the identity the engine put in the scene against the identity it put in the
 * seek index, and a viewport that cannot prove they match draws nothing rather
 * than pairing one swing's geometry with another's pixels.
 *
 * And a frame the reconstruction refused is a frame with no body in it.
 * `frameAt` returns null there, which a viewport must render as an empty scene:
 * holding the previous pose would put a body on screen in a position it was
 * never measured in, which is Phase 14's rule about interpolating across a
 * blocked landmark, one dimension up.
 */

import type {
  ReconstructionScene,
  SceneFrame,
  ScenePoint,
  SeekIndex,
} from "@gsa/types";

import {
  subtract,
  visibleFraction,
  type ViewCamera,
  type Vec3,
} from "@/features/scene/projection";

/**
 * The scene's frame for a reference-clip frame index, or null.
 *
 * Frames are contiguous from `start_frame`, so this is an offset rather than a
 * search — and the offset is **checked** against the frame's own index before
 * being returned. A scene that ever ships a gap would otherwise draw frame 40's
 * body while the panel above it said 43, silently, which is the exact failure
 * Phase 14 built a residual display to make visible one layer down.
 */
export function frameAt(
  scene: ReconstructionScene,
  frameIndex: number,
): SceneFrame | null {
  const offset = frameIndex - scene.start_frame;
  const frame = offset >= 0 ? scene.frames[offset] : undefined;
  if (frame?.frame_index === frameIndex) return frame;
  // The arithmetic did not agree with the frame it landed on, so it is not
  // trusted. A search costs a pass over at most `MAX_SCENE_FRAMES` entries and
  // is the difference between returning nothing and returning the wrong body.
  return scene.frames.find((entry) => entry.frame_index === frameIndex) ?? null;
}

/** The points of one frame, by landmark, for drawing edges without a scan each. */
export function pointsByLandmark(frame: SceneFrame): Map<number, ScenePoint> {
  return new Map(frame.points.map((point) => [point.landmark, point]));
}

/**
 * Whether a scene and a loaded clip are the same recording.
 *
 * Content, not path: a clip that has been moved or renamed still matches, and a
 * different recording that happens to sit at the same path does not. That is the
 * rule `ProjectStore` has used for clip identity since Phase 7, applied to the
 * one place where getting it wrong produces a plausible picture instead of an
 * error — a skeleton scrubbing in step with pixels it has nothing to do with.
 */
export function sceneMatchesClip(
  scene: ReconstructionScene,
  index: SeekIndex | null,
): boolean {
  if (!index) return false;
  return (
    scene.reference_content_key.digest === index.content_key.digest &&
    scene.reference_content_key.algorithm === index.content_key.algorithm
  );
}

/** What the current viewpoint shows of what the reconstruction does not know. */
export interface ViewpointHonesty {
  /** Points the figure was drawn from. */
  drawn: number;
  /** Median worst-direction uncertainty, in metres. What the measurement is worth. */
  sigmaM: number | null;
  /** Median fraction of it this viewpoint puts across the picture, 0 to 1. */
  visible: number | null;
  /** Median uncertainty a reader of this picture would infer, in metres. */
  apparentSigmaM: number | null;
}

/**
 * How honest the current view is about the frame on screen.
 *
 * **The number Phase 15 exists to report.** `sigmaM` is what the reconstruction
 * is worth and `apparentSigmaM` is what the picture looks like it is worth, and
 * `scripts/benchmark_viewport.py --sweep convergence` measures the second
 * staying flat at 5.3 mm while the first grows from 5.5 mm to 25.6 mm as the two
 * cameras are brought together. A viewport that showed only the picture would be
 * showing the flat one.
 *
 * Medians rather than means, because a handful of badly conditioned points --
 * usually a hand at the edge of the overlap -- would otherwise describe the
 * whole frame.
 */
export function viewpointHonesty(
  frame: SceneFrame,
  camera: ViewCamera | null,
): ViewpointHonesty {
  const sigmas: number[] = [];
  const fractions: number[] = [];
  const apparent: number[] = [];

  for (const point of frame.points) {
    const { position, uncertainty } = point;
    if (!position || !uncertainty) continue;
    sigmas.push(uncertainty.sigma_m);
    if (!camera) continue;
    const fraction = visibleFraction(
      uncertainty,
      uncertainty.sigma_m,
      camera.position,
      position,
    );
    if (fraction === null) continue;
    fractions.push(fraction);
    apparent.push(uncertainty.sigma_m * fraction);
  }

  return {
    drawn: sigmas.length,
    sigmaM: median(sigmas),
    visible: median(fractions),
    apparentSigmaM: median(apparent),
  };
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[middle] ?? null;
  const lower = sorted[middle - 1];
  const upper = sorted[middle];
  if (lower === undefined || upper === undefined) return null;
  return (lower + upper) / 2;
}

/** How far the viewer is from the body, so a reset can put them back. */
export function defaultDistance(scene: ReconstructionScene): number {
  const reference = scene.cameras.find((entry) => entry.kind === "reference");
  if (!reference) return Math.max(scene.radius_m * 3, 1);
  // Where the camera that took the footage actually stood. A viewport that
  // opened at a chosen distance would frame every capture the same way and tell
  // a reader nothing about how far off the subject was.
  return Math.max(
    Math.hypot(
      scene.centroid.x - reference.position.x,
      scene.centroid.y - reference.position.y,
      scene.centroid.z - reference.position.z,
    ),
    scene.radius_m * 1.2,
  );
}

/** Refusal reasons present in a frame, counted, so a hole can explain itself. */
export function refusalsIn(frame: SceneFrame): Map<string, number> {
  const counts = new Map<string, number>();
  for (const point of frame.points) {
    if (!point.refused) continue;
    counts.set(point.refused, (counts.get(point.refused) ?? 0) + 1);
  }
  return counts;
}

/** Distance from the camera to a point, for depth-sorting what is drawn. */
export function depthFrom(camera: ViewCamera, point: Vec3): number {
  const relative = subtract(point, camera.position);
  return (
    relative.x * camera.forward.x +
    relative.y * camera.forward.y +
    relative.z * camera.forward.z
  );
}
