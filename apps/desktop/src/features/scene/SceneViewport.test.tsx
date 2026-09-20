/**
 * What the viewport draws, and — more often — what it refuses to.
 *
 * These assertions are possible because the viewport is SVG. A WebGL canvas
 * renders the same picture and exposes nothing about it, so "the wrist is where
 * frame 13's wrist should be" would have to be a screenshot comparison, which
 * cannot distinguish a right answer from a plausible one. Every joint below is a
 * DOM node carrying its landmark, so the question can simply be asked.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  orbitCamera,
  projectPoint,
  referenceCamera,
} from "@/features/scene/projection";
import {
  EMPTY_FRAME,
  FIRST_FRAME,
  PARTIAL_FRAME,
  sceneFixture,
} from "@/features/scene/scene.fixture";
import { frameAt } from "@/features/scene/scene";
import { SceneViewport } from "@/features/scene/SceneViewport";

const scene = sceneFixture();

function view(
  frameIndex: number | null,
  overrides: Partial<Parameters<typeof SceneViewport>[0]> = {},
) {
  return render(
    <SceneViewport
      scene={scene}
      frame={frameIndex === null ? null : frameAt(scene, frameIndex)}
      camera={referenceCamera(scene)}
      showUncertainty
      magnification={20}
      showTrajectories
      showCameras
      onDrag={vi.fn()}
      onZoom={vi.fn()}
      {...overrides}
    />,
  );
}

function joints(): SVGElement[] {
  return [
    ...document.querySelectorAll<SVGElement>("[data-testid='scene-joint']"),
  ];
}

function bones(): SVGElement[] {
  return [
    ...document.querySelectorAll<SVGElement>("[data-testid='scene-bone']"),
  ];
}

describe("a complete frame", () => {
  it("draws every landmark and every bone", () => {
    view(FIRST_FRAME);
    expect(joints()).toHaveLength(4);
    expect(bones()).toHaveLength(scene.connections.length);
  });

  it("puts each joint where the engine's own projection puts it", () => {
    // Not "somewhere plausible": the reference camera's intrinsics are the ones
    // the calibration measured, so this is the projection that produced the
    // video, run again on the reconstruction.
    view(FIRST_FRAME);
    const camera = referenceCamera(scene);
    const frame = frameAt(scene, FIRST_FRAME);
    expect(camera && frame).toBeTruthy();
    if (!camera || !frame) return;

    for (const element of joints()) {
      const landmark = Number(element.getAttribute("data-landmark"));
      const point = frame.points.find((entry) => entry.landmark === landmark);
      expect(point?.position).toBeTruthy();
      if (!point?.position) continue;
      const expected = projectPoint(camera, point.position);
      expect(Number(element.getAttribute("cx"))).toBeCloseTo(
        expected?.x ?? 0,
        6,
      );
      expect(Number(element.getAttribute("cy"))).toBeCloseTo(
        expected?.y ?? 0,
        6,
      );
    }
  });

  it("announces the frame it is drawing", () => {
    view(FIRST_FRAME + 1);
    const viewport = screen.getByTestId("scene-viewport");
    expect(viewport).toHaveAttribute("data-frame", String(FIRST_FRAME + 1));
    expect(viewport).toHaveAttribute("data-reconstructed", "4");
  });
});

describe("a frame missing one landmark", () => {
  it("drops the bones that lost an end and keeps the rest", () => {
    view(PARTIAL_FRAME);
    expect(joints()).toHaveLength(3);
    // Landmark 14 is refused, so only the 12-14 connection goes. Drawing to its
    // last known position instead is how an overlay grows a limb pointing at
    // nothing, which is `OverlayCanvas`'s rule in three dimensions.
    expect(bones()).toHaveLength(scene.connections.length - 1);
    expect(
      joints().map((element) => element.getAttribute("data-landmark")),
    ).not.toContain("14");
  });

  it("does not draw an uncertainty ellipse for a point with no position", () => {
    view(PARTIAL_FRAME);
    expect(
      document.querySelectorAll("[data-testid='scene-uncertainty']"),
    ).toHaveLength(3);
  });
});

describe("a frame with nothing in it", () => {
  it("draws no body at all", () => {
    // **The rule Phase 14 set one dimension down.** A viewport that held the
    // previous pose here would put a body on screen in a position it was never
    // measured in, and it would look like the smoothest part of the swing.
    view(EMPTY_FRAME);
    expect(joints()).toHaveLength(0);
    expect(screen.getByTestId("scene-empty")).toHaveTextContent(
      /Nothing was reconstructed/,
    );
  });

  it("still reports which frame it is showing", () => {
    view(EMPTY_FRAME);
    expect(screen.getByTestId("scene-viewport")).toHaveAttribute(
      "data-frame",
      String(EMPTY_FRAME),
    );
  });
});

describe("a frame outside the reconstruction", () => {
  it("says so rather than drawing the nearest one it has", () => {
    view(null);
    expect(joints()).toHaveLength(0);
    expect(screen.getByTestId("scene-empty")).toHaveTextContent(
      /outside the reconstruction/,
    );
  });
});

describe("trajectories", () => {
  it("breaks the polyline at a refused instant instead of drawing across it", () => {
    // One trajectory with a hole in the middle must be two runs. A single
    // polyline with the nulls filtered out would draw a straight chord over the
    // missing frames — which is exactly what a missing downswing would look most
    // convincing doing.
    view(FIRST_FRAME);
    expect(
      document.querySelectorAll("[data-testid='scene-trajectory-run']"),
    ).toHaveLength(2);
  });

  it("is not drawn when it is switched off", () => {
    view(FIRST_FRAME, { showTrajectories: false });
    expect(
      document.querySelectorAll("[data-testid='scene-trajectory-run']"),
    ).toHaveLength(0);
  });
});

describe("the cameras", () => {
  it("draws both once the reader has moved off one of them", () => {
    // Where the two cameras stood is the single largest thing deciding what the
    // reconstruction is worth, and the one a panel of numbers makes least
    // legible. Orbited away, both are in the picture.
    render(
      <SceneViewport
        scene={scene}
        frame={frameAt(scene, FIRST_FRAME)}
        camera={orbitCamera(
          scene,
          { azimuthDeg: 55, elevationDeg: 20, distance: 6 },
          45,
        )}
        showUncertainty={false}
        magnification={20}
        showTrajectories={false}
        showCameras
        onDrag={vi.fn()}
        onZoom={vi.fn()}
      />,
    );
    const marks = [
      ...document.querySelectorAll("[data-testid='scene-camera']"),
    ];
    expect(marks.map((element) => element.getAttribute("data-kind"))).toEqual([
      "reference",
      "target",
    ]);
  });

  it("does not draw the camera the reader is standing in", () => {
    // From the reference camera, the reference camera is at zero depth — behind
    // its own lens. Projecting it anyway would put a marker at the principal
    // point of every default view, which reads as a joint.
    view(FIRST_FRAME);
    const marks = [
      ...document.querySelectorAll("[data-testid='scene-camera']"),
    ];
    expect(marks.map((element) => element.getAttribute("data-kind"))).toEqual([
      "target",
    ]);
  });
});

describe("the uncertainty ellipses", () => {
  it("are nearly round from the reference camera and long from the side", () => {
    // The same fact the panel states in words. The fixture's ellipsoid is a
    // cigar down the reference camera's optical axis.
    view(FIRST_FRAME);
    const head = document.querySelector("[data-testid='scene-uncertainty']");
    const flatRatio =
      Number(head?.getAttribute("rx")) / Number(head?.getAttribute("ry"));

    render(
      <SceneViewport
        scene={scene}
        frame={frameAt(scene, FIRST_FRAME)}
        camera={orbitCamera(
          scene,
          { azimuthDeg: 90, elevationDeg: 0, distance: 2.5 },
          45,
        )}
        showUncertainty
        magnification={20}
        showTrajectories={false}
        showCameras={false}
        onDrag={vi.fn()}
        onZoom={vi.fn()}
      />,
    );
    const all = [
      ...document.querySelectorAll("[data-testid='scene-uncertainty']"),
    ];
    const sideOn = all[all.length - 1];
    const sideRatio =
      Number(sideOn?.getAttribute("rx")) / Number(sideOn?.getAttribute("ry"));

    expect(flatRatio).toBeLessThan(2);
    expect(sideRatio).toBeGreaterThan(4);
  });

  it("are not drawn when they are switched off", () => {
    view(FIRST_FRAME, { showUncertainty: false });
    expect(
      document.querySelectorAll("[data-testid='scene-uncertainty']"),
    ).toHaveLength(0);
    expect(screen.queryByTestId("scene-magnification")).toBeNull();
  });

  it("scale linearly with the magnification, and say what it is", () => {
    // **The factor is never a silent constant.** At true scale a well-determined
    // joint's 1-sigma ellipse is a fraction of a screen pixel, so the ellipses
    // are exaggerated to be worth drawing at all — and an exaggerated size with
    // no factor beside it is a size claim that is simply false.
    const radiusOf = (container: HTMLElement) =>
      Number(
        container
          .querySelector(
            "[data-testid='scene-uncertainty'][data-landmark='11']",
          )
          ?.getAttribute("rx"),
      );

    const trueScale = view(FIRST_FRAME, { magnification: 1 });
    expect(
      within(trueScale.container).getByTestId("scene-magnification"),
    ).toHaveTextContent("×1");

    const magnified = view(FIRST_FRAME, { magnification: 50 });
    expect(
      within(magnified.container).getByTestId("scene-magnification"),
    ).toHaveTextContent("×50");
    expect(
      radiusOf(magnified.container) / radiusOf(trueScale.container),
    ).toBeCloseTo(50, 6);
  });
});

describe("with no camera", () => {
  it("draws nothing rather than falling back to some default projection", () => {
    view(FIRST_FRAME, { camera: null });
    expect(joints()).toHaveLength(0);
  });
});
