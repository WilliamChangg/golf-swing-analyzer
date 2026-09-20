/**
 * Metres to pixels, for a camera the reader is holding.
 *
 * Everything else this app draws arrives ready to draw. `PoseOverlay` carries
 * coordinates the engine already converted, because a frame has exactly one
 * camera and the engine may as well own the arithmetic — and owning it is what
 * stops the app forming a second opinion about the aspect ratio, the lens or the
 * filter's validity mask.
 *
 * A reconstruction has no such camera. There is no canonical view of a swing in
 * three dimensions, so the viewpoint is a choice made with a mouse sixty times a
 * second, and a round trip to the engine per frame is not a design. The
 * projection therefore lives here, which makes it a **second implementation of
 * arithmetic the engine already has** — and a second implementation is a second
 * opinion unless something forces the two to agree.
 *
 * Two things do. `projection-truth.json` holds real points from the synthetic
 * reconstruction and the pixels the engine's own `StereoGeometry.project` puts
 * them at, and `projection.test.ts` asserts this module reproduces them to
 * floating-point precision. And `referenceCamera` reconstructs the camera that
 * took the reference clip, so the viewport at its default position is drawing
 * the same picture the video shows — a disagreement between them is then a real
 * disagreement rather than a rendering artefact.
 *
 * ## Conventions, stated once
 *
 * Scene coordinates are `LandmarkSpace.CAMERA` metres: the reference camera at
 * the origin looking down +z, with **y increasing downward** because that is
 * where image y goes. A camera's `up` is therefore its image's -y. There is no
 * gravity here and no ground plane — `ReconstructionScene` refuses to supply
 * either, because nothing in a stereo pair measures which way is up.
 */

import type { ReconstructionScene } from "@gsa/types";

/** A point or direction in scene metres. */
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** The six unique elements of a symmetric 3x3 covariance, in metres squared. */
export interface Covariance {
  xx: number;
  yy: number;
  zz: number;
  xy: number;
  xz: number;
  yz: number;
}

/**
 * A camera to draw from: where it is, which way it faces, and its intrinsics.
 *
 * One type for both modes. The reference mode fills it from the real camera the
 * scene carries, so its picture is that camera's picture; the orbit mode
 * synthesises intrinsics from a field of view. Keeping them one type is what
 * makes "look from the reference camera" a position rather than a special case
 * with its own projection.
 */
export interface ViewCamera {
  position: Vec3;
  /** Screen +x. */
  right: Vec3;
  /** Screen up, which is the image's -y. */
  up: Vec3;
  /** The optical axis, pointing the way the camera looks. */
  forward: Vec3;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

/** Where a point landed, and how far away it was. */
export interface Projected {
  x: number;
  y: number;
  /** Depth along the optical axis, in metres. Positive, or the point is not drawn. */
  depth: number;
}

/** A 1-sigma uncertainty ellipse in the same pixels `Projected` uses. */
export interface Ellipse {
  rx: number;
  ry: number;
  /** Degrees, clockwise from screen +x, because SVG's y runs downward too. */
  angleDeg: number;
}

/** Where the viewer stands, relative to the reference camera's own direction. */
export interface Orbit {
  /** Degrees around the scene's vertical. Zero is the reference camera. */
  azimuthDeg: number;
  /** Degrees above the reference camera's own height. */
  elevationDeg: number;
  /** Distance from the orbit centre, in metres. */
  distance: number;
}

/** How many standard deviations the ellipse around each joint represents. */
export const UNCERTAINTY_SIGMAS = 1;

/**
 * Magnifications offered for the uncertainty ellipses, and why there are any.
 *
 * **At true scale the ellipse is invisible, and that is a measurement rather
 * than a rendering problem.** A good capture determines a joint to 5.4 mm at
 * 3.4 m; through a 1400 px focal length that is 2.2 px of a 1920-wide frame,
 * which on screen is about two thirds of one pixel. Drawn honestly at 1x it is
 * nothing — so the one control that exists to show a reader where a measurement
 * is weak would show them a clean skeleton in every case, including the cases it
 * was built for.
 *
 * So it is magnified, and the factor is **in the picture, always** — never a
 * silent constant. The millimetres in `ViewpointPanel` remain the truth; what the
 * ellipse carries is the part a number cannot, which is the *shape* and how it
 * changes as the reader moves. 1x is offered so the true scale can be seen for
 * what it is.
 */
export const MAGNIFICATIONS = [1, 10, 20, 50] as const;

export type Magnification = (typeof MAGNIFICATIONS)[number];

/**
 * Points nearer than this to the camera are treated as behind it.
 *
 * Not zero, for `triangulate._MIN_DEPTH_M`'s reason one layer up: a point
 * exactly at the optical centre divides by zero, and one a millimetre in front
 * of a lens is not a thing anybody photographed.
 */
const MIN_DEPTH_M = 1e-3;

export function vec(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function scale(a: Vec3, factor: number): Vec3 {
  return { x: a.x * factor, y: a.y * factor, z: a.z * factor };
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function length(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}

/** A unit vector, or null where there is no direction to take. */
export function normalize(a: Vec3): Vec3 | null {
  const magnitude = length(a);
  if (!Number.isFinite(magnitude) || magnitude === 0) return null;
  return scale(a, 1 / magnitude);
}

/**
 * The camera that took the reference clip, ready to draw from.
 *
 * Its intrinsics are the ones the calibration measured, so the viewport's
 * default view is not "a view that resembles the video" — it is the projection
 * that produced the video, run again on the reconstruction. That is what makes
 * the 3D skeleton and the 2D overlay checkable against each other.
 */
export function referenceCamera(scene: ReconstructionScene): ViewCamera | null {
  const camera = scene.cameras.find((entry) => entry.kind === "reference");
  if (!camera) return null;
  return {
    position: camera.position,
    right: camera.right,
    up: camera.up,
    forward: camera.forward,
    fx: camera.fx,
    fy: camera.fy,
    cx: camera.cx,
    cy: camera.cy,
    width: camera.image_width,
    height: camera.image_height,
  };
}

/**
 * A camera orbited away from the reference camera, keeping the scene framed.
 *
 * Zero azimuth and zero elevation put the viewer **exactly** where the reference
 * camera stands, looking at `centre` — so the orbit's origin is the view whose
 * errors point away from the reader, and every degree of it is a degree of
 * bringing them into view. `visibleFraction` is what reports how far that has
 * got.
 *
 * The vertical the orbit turns about is the reference camera's own up. Not
 * gravity: nothing in this pipeline measures gravity, and pretending otherwise
 * would put a horizon in a picture that has no horizon in it.
 */
export function orbitCamera(
  scene: ReconstructionScene,
  orbit: Orbit,
  fieldOfViewDeg: number,
): ViewCamera | null {
  const reference = referenceCamera(scene);
  if (!reference) return null;

  const centre = scene.centroid;
  const outward =
    normalize(subtract(reference.position, centre)) ?? vec(0, 0, -1);
  const vertical = reference.up;

  const azimuth = (orbit.azimuthDeg * Math.PI) / 180;
  const elevation = (orbit.elevationDeg * Math.PI) / 180;

  // Turn about the vertical first, then lift towards it. Doing it in this order
  // means elevation is always measured from the horizontal circle the azimuth
  // traced, so dragging up never also drags sideways.
  const turned = rotateAbout(outward, vertical, azimuth);
  const sideways = normalize(cross(vertical, turned));
  const lifted = sideways ? rotateAbout(turned, sideways, -elevation) : turned;

  const position = add(centre, scale(lifted, orbit.distance));
  const forward = normalize(subtract(centre, position));
  if (!forward) return null;

  const right = normalize(cross(forward, vertical));
  if (!right) return null;
  const up = normalize(cross(right, forward));
  if (!up) return null;

  // Intrinsics from a field of view, over the reference camera's frame shape, so
  // the viewport box is the same shape in both modes and switching between them
  // does not restretch the picture.
  const focal =
    reference.width / 2 / Math.tan((fieldOfViewDeg * Math.PI) / 360);
  return {
    position,
    right,
    up,
    forward,
    fx: focal,
    fy: focal,
    cx: reference.width / 2,
    cy: reference.height / 2,
    width: reference.width,
    height: reference.height,
  };
}

/** Rodrigues' rotation of `v` about a unit `axis` by `angle` radians. */
export function rotateAbout(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const unit = normalize(axis);
  if (!unit) return v;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return add(
    add(scale(v, cosine), scale(cross(unit, v), sine)),
    scale(unit, dot(unit, v) * (1 - cosine)),
  );
}

/** The point in the camera's own frame: x right, y **down**, z along the axis. */
function toCameraFrame(camera: ViewCamera, point: Vec3): Vec3 {
  const relative = subtract(point, camera.position);
  return {
    x: dot(relative, camera.right),
    // The camera's up is the image's -y, so image y is the negative of it.
    y: -dot(relative, camera.up),
    z: dot(relative, camera.forward),
  };
}

/**
 * Where a scene point lands, in the camera's image pixels.
 *
 * Null for a point at or behind the optical centre, which is drawn as absent.
 * Clamping it to the frame edge instead would put a joint on screen at a
 * position nothing computed.
 */
export function projectPoint(
  camera: ViewCamera,
  point: Vec3,
): Projected | null {
  const local = toCameraFrame(camera, point);
  if (!Number.isFinite(local.z) || local.z <= MIN_DEPTH_M) return null;
  const x = (camera.fx * local.x) / local.z + camera.cx;
  const y = (camera.fy * local.y) / local.z + camera.cy;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y, depth: local.z };
}

/**
 * The 1-sigma ellipse a point's positional covariance projects to.
 *
 * `A S A'` for the 2x3 Jacobian `A` of the projection at that point — the same
 * first-order propagation the engine used to produce `S` in the first place,
 * applied once more to get from metres to pixels.
 *
 * **This is the whole reason the covariance crosses the wire instead of a
 * radius.** A radius draws as a circle, a circle says the point is equally well
 * known in every direction, and `scripts/benchmark_viewport.py` measures that
 * being false by a factor of five on a capture whose cameras are close together.
 * The ellipse is what puts the anisotropy on screen: from the reference camera
 * it is small and round because the long axis points at the reader, and it opens
 * out as they orbit.
 */
export function projectCovariance(
  camera: ViewCamera,
  point: Vec3,
  covariance: Covariance,
): Ellipse | null {
  const local = toCameraFrame(camera, point);
  if (!Number.isFinite(local.z) || local.z <= MIN_DEPTH_M) return null;

  // d(pixel)/d(local), chained through the camera's own basis. Written as two
  // explicit rows rather than as nested arrays: this runs per joint per frame,
  // and a 2x3 with names is easier to check against the engine's `_jacobian`
  // than the same numbers behind three index lookups.
  const depth = local.z;
  const a0 = camera.fx / depth;
  const a2 = (-camera.fx * local.x) / (depth * depth);
  const b1 = camera.fy / depth;
  const b2 = (-camera.fy * local.y) / (depth * depth);

  // Rows of the world-to-camera rotation: screen x, image y (down), the axis.
  const right = camera.right;
  const down = { x: -camera.up.x, y: -camera.up.y, z: -camera.up.z };
  const forward = camera.forward;

  const rowU: Vec3 = {
    x: a0 * right.x + a2 * forward.x,
    y: a0 * right.y + a2 * forward.y,
    z: a0 * right.z + a2 * forward.z,
  };
  const rowV: Vec3 = {
    x: b1 * down.x + b2 * forward.x,
    y: b1 * down.y + b2 * forward.y,
    z: b1 * down.z + b2 * forward.z,
  };

  // `A S A'`, the 2x2 screen covariance.
  const uu = quadratic(covariance, rowU, rowU);
  const uv = quadratic(covariance, rowU, rowV);
  const vv = quadratic(covariance, rowV, rowV);

  const [major, minor] = eigenvalues2(uu, uv, vv);
  if (!Number.isFinite(major) || major < 0) return null;
  return {
    rx: Math.sqrt(Math.max(major, 0)),
    ry: Math.sqrt(Math.max(minor, 0)),
    angleDeg: (eigenAngle(uu, uv, major) * 180) / Math.PI,
  };
}

/**
 * Eigenvalues of the symmetric `[[a, b], [b, c]]`, largest first.
 *
 * Closed form rather than an iteration: a 2x2 symmetric eigenproblem has one,
 * it is exact, and an ellipse redrawn sixty times a second should not be the
 * output of a loop whose termination depends on the data.
 */
export function eigenvalues2(
  a: number,
  b: number,
  c: number,
): [number, number] {
  const mean = (a + c) / 2;
  const half = (a - c) / 2;
  const spread = Math.sqrt(Math.max(half * half + b * b, 0));
  return [mean + spread, mean - spread];
}

/** Angle of the eigenvector for `eigenvalue`, in radians. */
function eigenAngle(a: number, b: number, eigenvalue: number): number {
  if (b === 0) return a >= eigenvalue ? 0 : Math.PI / 2;
  return Math.atan2(eigenvalue - a, b);
}

/**
 * How much of a point's uncertainty a viewer at `eye` can see. Zero to one.
 *
 * **Phase 15's number.** The largest standard deviation across the line of
 * sight, over the largest there is. One means this viewpoint shows the worst of
 * what is unknown about the joint; a fifth means four fifths of it is pointing
 * at the reader and the picture looks correspondingly more confident than the
 * measurement is.
 *
 * It is a ratio, so it says nothing about whether the reconstruction is good — a
 * precise point and a hopeless one both score 1.0 from a viewpoint that shows
 * their uncertainty honestly. It is a property of the *view*, which is exactly
 * what a viewport control needs to report and what no number in a
 * `ReconstructionReport` could.
 *
 * `sigmaM` is the scene's own worst-direction figure, so the denominator is the
 * engine's rather than a second eigen-decomposition here that could disagree
 * with it.
 */
export function visibleFraction(
  covariance: Covariance,
  sigmaM: number,
  eye: Vec3,
  point: Vec3,
): number | null {
  const direction = normalize(subtract(point, eye));
  if (!direction || !(sigmaM > 0)) return null;

  const [first, second] = perpendicularBasis(direction);
  const across: [number, number, number] = [
    quadratic(covariance, first, first),
    quadratic(covariance, first, second),
    quadratic(covariance, second, second),
  ];
  const [largest] = eigenvalues2(across[0], across[1], across[2]);
  if (!Number.isFinite(largest) || largest < 0) return null;
  return Math.min(Math.sqrt(largest) / sigmaM, 1);
}

/** `a' S b` for the covariance `S`. */
function quadratic(covariance: Covariance, a: Vec3, b: Vec3): number {
  const sa = {
    x: covariance.xx * a.x + covariance.xy * a.y + covariance.xz * a.z,
    y: covariance.xy * a.x + covariance.yy * a.y + covariance.yz * a.z,
    z: covariance.xz * a.x + covariance.yz * a.y + covariance.zz * a.z,
  };
  return dot(sa, b);
}

/** Any orthonormal pair spanning the plane across `direction`. */
export function perpendicularBasis(direction: Vec3): [Vec3, Vec3] {
  // Cross with whichever axis the direction is least aligned to, so the result
  // never degenerates for a direction that happens to be an axis.
  const seed =
    Math.abs(direction.x) < Math.abs(direction.y) &&
    Math.abs(direction.x) < Math.abs(direction.z)
      ? vec(1, 0, 0)
      : Math.abs(direction.y) < Math.abs(direction.z)
        ? vec(0, 1, 0)
        : vec(0, 0, 1);
  const first = normalize(cross(direction, seed)) ?? vec(1, 0, 0);
  const second = normalize(cross(direction, first)) ?? vec(0, 1, 0);
  return [first, second];
}
