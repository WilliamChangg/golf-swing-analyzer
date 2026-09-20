import type { ReconstructionScene } from "@gsa/types";
import { describe, expect, it } from "vitest";

import truth from "./projection-truth.json";
import {
  cross,
  dot,
  eigenvalues2,
  length,
  normalize,
  orbitCamera,
  perpendicularBasis,
  projectCovariance,
  projectPoint,
  referenceCamera,
  subtract,
  vec,
  visibleFraction,
  type Covariance,
  type ViewCamera,
} from "./projection";

/**
 * The viewport's projection, held to the engine's.
 *
 * This module is the one place in the app that re-implements arithmetic the
 * Python engine already has, and it is here for a reason the rest of the app
 * does not have: a viewport's camera moves with the mouse, so the projection
 * cannot live behind an IPC call. What makes that safe is not care — it is
 * `projection-truth.json`, which holds real points from the synthetic stereo
 * reconstruction and the pixels `StereoGeometry.project` puts them at. A sign
 * flipped in either language fails here rather than producing a convincing
 * picture of a body that is inside out.
 *
 * Regenerate it with:
 *   uv run --project python python scripts/benchmark_viewport.py \
 *     --write-fixture apps/desktop/src/features/scene/projection-truth.json
 */

const CAMERA: ViewCamera = {
  position: truth.camera.position,
  right: truth.camera.right,
  up: truth.camera.up,
  forward: truth.camera.forward,
  fx: truth.camera.fx,
  fy: truth.camera.fy,
  cx: truth.camera.cx,
  cy: truth.camera.cy,
  width: truth.camera.image_width,
  height: truth.camera.image_height,
};

/** A minimal scene carrying only what the camera helpers read. */
function sceneWith(
  overrides: Partial<ReconstructionScene> = {},
): ReconstructionScene {
  return {
    schema_version: 1,
    space: "camera",
    reference_content_key: {
      algorithm: "sha256-sampled-v1",
      digest: "0".repeat(64),
      size_bytes: 0,
    },
    reference_name: "face_on.mp4",
    target_name: "dtl.mp4",
    reference_role: "face_on",
    target_role: "down_the_line",
    start_frame: 0,
    end_frame: 0,
    frames: [],
    landmarks: [],
    connections: [],
    trajectories: [],
    cameras: [
      {
        kind: "reference",
        role: "face_on",
        name: "face_on.mp4",
        position: vec(0, 0, 0),
        forward: vec(0, 0, 1),
        up: vec(0, -1, 0),
        right: vec(1, 0, 0),
        fx: CAMERA.fx,
        fy: CAMERA.fy,
        cx: CAMERA.cx,
        cy: CAMERA.cy,
        image_width: CAMERA.width,
        image_height: CAMERA.height,
        horizontal_fov_deg: 69,
      },
    ],
    calibration: "stereo",
    centroid: vec(0, -0.37, 2.92),
    radius_m: 1.17,
    slow_motion_factor: 1,
    pixel_sigma_px: 2.29,
    report: {} as ReconstructionScene["report"],
    warnings: [],
    ...overrides,
  };
}

describe("the engine's pixels", () => {
  it("has a fixture with points spread across the frame", () => {
    // A fixture clustered in the middle would pass on a projection with the
    // principal point wrong, which is the error this is most able to catch.
    expect(truth.samples.length).toBeGreaterThan(8);
    const xs = truth.samples.map((sample) => sample.pixel.u);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(100);
  });

  it("is reproduced by projectPoint to floating-point precision", () => {
    for (const sample of truth.samples) {
      const projected = projectPoint(CAMERA, sample.point);
      expect(projected).not.toBeNull();
      expect(projected?.x).toBeCloseTo(sample.pixel.u, 9);
      expect(projected?.y).toBeCloseTo(sample.pixel.v, 9);
      expect(projected?.depth).toBeCloseTo(sample.point.z, 9);
    }
  });

  it("is reproduced through a camera whose basis is not the identity", () => {
    // The fixture's reference camera is at the origin with an axis-aligned
    // basis, so a projection that ignored `position`, `right`, `up` and
    // `forward` entirely would pass the test above. This one moves the camera
    // and puts the point back where it was relative to it.
    const shifted: ViewCamera = {
      ...CAMERA,
      position: vec(1, 2, -3),
    };
    for (const sample of truth.samples.slice(0, 6)) {
      const moved = {
        x: sample.point.x + 1,
        y: sample.point.y + 2,
        z: sample.point.z - 3,
      };
      const projected = projectPoint(shifted, moved);
      expect(projected?.x).toBeCloseTo(sample.pixel.u, 9);
      expect(projected?.y).toBeCloseTo(sample.pixel.v, 9);
    }
  });
});

describe("points that cannot be drawn", () => {
  it("returns null behind the camera rather than a mirrored ghost", () => {
    // A projection without the depth check puts a point behind the lens on
    // screen, mirrored through the principal point, which looks like a tracking
    // failure rather than like geometry.
    expect(projectPoint(CAMERA, vec(0.1, 0.1, -2))).toBeNull();
    expect(projectPoint(CAMERA, vec(0.1, 0.1, 0))).toBeNull();
  });

  it("returns null for a point that is not a number", () => {
    expect(projectPoint(CAMERA, vec(Number.NaN, 0, 3))).toBeNull();
  });
});

describe("the orbit camera", () => {
  const scene = sceneWith();

  it("starts exactly at the reference camera", () => {
    const reference = referenceCamera(scene);
    const centre = scene.centroid;
    const distance = length(
      subtract(reference?.position ?? vec(0, 0, 0), centre),
    );
    const orbited = orbitCamera(
      scene,
      { azimuthDeg: 0, elevationDeg: 0, distance },
      60,
    );

    expect(orbited).not.toBeNull();
    expect(orbited?.position.x).toBeCloseTo(0, 9);
    expect(orbited?.position.y).toBeCloseTo(0, 9);
    expect(orbited?.position.z).toBeCloseTo(0, 9);
    // And it is looking at the body, not merely standing where the camera did.
    expect(orbited?.forward.z).toBeGreaterThan(0.9);
  });

  it("keeps an orthonormal, image-handed basis at every angle", () => {
    for (const azimuthDeg of [0, 37, 90, 180, -125]) {
      for (const elevationDeg of [0, 25, -40]) {
        const camera = orbitCamera(
          scene,
          { azimuthDeg, elevationDeg, distance: 3 },
          60,
        );
        expect(camera).not.toBeNull();
        if (!camera) continue;

        for (const axis of [camera.right, camera.up, camera.forward]) {
          expect(length(axis)).toBeCloseTo(1, 9);
        }
        expect(dot(camera.right, camera.up)).toBeCloseTo(0, 9);
        expect(dot(camera.right, camera.forward)).toBeCloseTo(0, 9);
        expect(dot(camera.up, camera.forward)).toBeCloseTo(0, 9);
        // Screen right, screen up and the way it looks: left-handed, because
        // the image's y runs downward. The same assertion `test_scene.py` makes
        // of the cameras the engine ships.
        expect(dot(cross(camera.right, camera.up), camera.forward)).toBeCloseTo(
          -1,
          9,
        );
      }
    }
  });

  it("stays the declared distance from the body it orbits", () => {
    for (const azimuthDeg of [0, 45, 90, 200]) {
      const camera = orbitCamera(
        scene,
        { azimuthDeg, elevationDeg: 20, distance: 2.5 },
        60,
      );
      expect(
        length(subtract(camera?.position ?? vec(0, 0, 0), scene.centroid)),
      ).toBeCloseTo(2.5, 9);
    }
  });

  it("keeps the reference camera's frame shape, so switching modes does not restretch", () => {
    const camera = orbitCamera(
      scene,
      { azimuthDeg: 40, elevationDeg: 0, distance: 3 },
      60,
    );
    expect(camera?.width).toBe(CAMERA.width);
    expect(camera?.height).toBe(CAMERA.height);
    expect(camera?.fx).toBeCloseTo(camera?.fy ?? 0, 12);
  });
});

describe("the uncertainty ellipse", () => {
  /** A covariance that is a long cigar along the z axis. */
  const cigar: Covariance = {
    xx: 1e-4,
    yy: 1e-4,
    zz: 1e-2,
    xy: 0,
    xz: 0,
    yz: 0,
  };

  it("is small and round seen down its own long axis", () => {
    // The reference camera looks along +z, which is where this ellipsoid is
    // longest. That is the whole finding: the picture is at its most flattering
    // exactly where the measurement is at its worst.
    const ellipse = projectCovariance(CAMERA, vec(0, 0, 3), cigar);
    expect(ellipse).not.toBeNull();
    if (!ellipse) return;
    expect(ellipse.rx / ellipse.ry).toBeLessThan(1.05);
  });

  it("opens out seen from the side", () => {
    const sideOn: ViewCamera = {
      ...CAMERA,
      position: vec(3, 0, 0),
      right: vec(0, 0, 1),
      up: vec(0, -1, 0),
      forward: vec(-1, 0, 0),
    };
    const ellipse = projectCovariance(sideOn, vec(0, 0, 0), cigar);
    expect(ellipse).not.toBeNull();
    if (!ellipse) return;
    // sqrt(1e-2 / 1e-4) = 10, the true ratio of the ellipsoid's axes.
    expect(ellipse.rx / ellipse.ry).toBeCloseTo(10, 1);
  });

  it("grows with the covariance and shrinks with distance", () => {
    const near = projectCovariance(CAMERA, vec(0, 0, 2), cigar);
    const far = projectCovariance(CAMERA, vec(0, 0, 8), cigar);
    expect((near?.rx ?? 0) / (far?.rx ?? 1)).toBeGreaterThan(3);

    const wider = projectCovariance(CAMERA, vec(0, 0, 2), {
      ...cigar,
      xx: cigar.xx * 4,
      yy: cigar.yy * 4,
    });
    expect(wider?.rx ?? 0).toBeGreaterThan(near?.rx ?? 0);
  });

  it("returns null where the point cannot be drawn", () => {
    expect(projectCovariance(CAMERA, vec(0, 0, -1), cigar)).toBeNull();
  });
});

describe("eigenvalues2", () => {
  it("returns the eigenvalues of a diagonal matrix, largest first", () => {
    expect(eigenvalues2(9, 0, 4)).toEqual([9, 4]);
    expect(eigenvalues2(4, 0, 9)).toEqual([9, 4]);
  });

  it("agrees with the trace and determinant", () => {
    for (const [a, b, c] of [
      [3, 1, 2],
      [0.5, -0.25, 1.5],
      [7, 7, 7],
    ] as const) {
      const [major, minor] = eigenvalues2(a, b, c);
      expect(major + minor).toBeCloseTo(a + c, 9);
      expect(major * minor).toBeCloseTo(a * c - b * b, 9);
      expect(major).toBeGreaterThanOrEqual(minor);
    }
  });
});

describe("visibleFraction", () => {
  /** Ten times longer along z than across it. */
  const cigar: Covariance = {
    xx: 1e-4,
    yy: 1e-4,
    zz: 1e-2,
    xy: 0,
    xz: 0,
    yz: 0,
  };
  const sigma = Math.sqrt(1e-2);

  it("shows a tenth of it looking down the long axis", () => {
    // sqrt(1e-4 / 1e-2) = 0.1. A viewer here sees a small round dot around a
    // joint that could be ten centimetres out towards or away from them.
    expect(
      visibleFraction(cigar, sigma, vec(0, 0, -3), vec(0, 0, 0)),
    ).toBeCloseTo(0.1, 6);
  });

  it("shows all of it looking across the long axis", () => {
    expect(
      visibleFraction(cigar, sigma, vec(3, 0, 0), vec(0, 0, 0)),
    ).toBeCloseTo(1, 6);
    expect(
      visibleFraction(cigar, sigma, vec(0, 3, 0), vec(0, 0, 0)),
    ).toBeCloseTo(1, 6);
  });

  it("is one from every direction when the uncertainty is a sphere", () => {
    // A well-conditioned capture has no bad direction to hide, which is why the
    // benchmark's 90-degree row barely moves as the reader orbits.
    const sphere: Covariance = {
      xx: 1e-4,
      yy: 1e-4,
      zz: 1e-4,
      xy: 0,
      xz: 0,
      yz: 0,
    };
    for (const eye of [vec(0, 0, -3), vec(3, 0, 0), vec(1, 2, -2)]) {
      expect(visibleFraction(sphere, 1e-2, eye, vec(0, 0, 0))).toBeCloseTo(
        1,
        6,
      );
    }
  });

  it("never exceeds one, whatever the sigma it is given", () => {
    // The denominator is the engine's own worst-direction figure, so this is a
    // guard against a scene whose two numbers were produced inconsistently: a
    // fraction above one would be an impossible claim rather than a rounding.
    const fraction = visibleFraction(
      cigar,
      sigma * 0.5,
      vec(3, 0, 0),
      vec(0, 0, 0),
    );
    expect(fraction).toBe(1);
  });

  it("returns null where there is no direction or no scale", () => {
    expect(
      visibleFraction(cigar, sigma, vec(0, 0, 0), vec(0, 0, 0)),
    ).toBeNull();
    expect(visibleFraction(cigar, 0, vec(3, 0, 0), vec(0, 0, 0))).toBeNull();
  });
});

describe("perpendicularBasis", () => {
  it("spans the plane across the direction, for every axis", () => {
    for (const direction of [
      vec(1, 0, 0),
      vec(0, 1, 0),
      vec(0, 0, 1),
      vec(0.3, -0.5, 0.8),
    ]) {
      const unit = normalize(direction);
      expect(unit).not.toBeNull();
      if (!unit) continue;
      const [first, second] = perpendicularBasis(unit);
      expect(length(first)).toBeCloseTo(1, 9);
      expect(length(second)).toBeCloseTo(1, 9);
      expect(dot(first, unit)).toBeCloseTo(0, 9);
      expect(dot(second, unit)).toBeCloseTo(0, 9);
      expect(dot(first, second)).toBeCloseTo(0, 9);
    }
  });
});
