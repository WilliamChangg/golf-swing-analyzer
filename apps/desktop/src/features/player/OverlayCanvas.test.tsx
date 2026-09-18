/**
 * The overlay canvas.
 *
 * jsdom has no 2D context, so `getContext` is replaced with a recorder and the
 * tests assert on the drawing calls. That is the right level anyway: what
 * matters here is *what gets drawn and what does not* — a landmark nothing
 * supports must not appear, and a club head must not be marked on a frame where
 * the evidence stopped short of one.
 */

import type { OverlayFrame, PoseOverlay } from "@gsa/types";
import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { OverlayCanvas } from "./OverlayCanvas";

interface Recorder {
  moves: [number, number][];
  lines: [number, number][];
  arcs: [number, number, number][];
  fills: number;
  strokes: number;
}

let recorder: Recorder;

beforeEach(() => {
  recorder = { moves: [], lines: [], arcs: [], fills: 0, strokes: 0 };

  const context = {
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: (x: number, y: number) => recorder.moves.push([x, y]),
    lineTo: (x: number, y: number) => recorder.lines.push([x, y]),
    arc: (x: number, y: number, r: number) => recorder.arcs.push([x, y, r]),
    fill: () => {
      recorder.fills += 1;
    },
    stroke: () => {
      recorder.strokes += 1;
    },
    lineWidth: 0,
    lineCap: "butt",
    strokeStyle: "",
    fillStyle: "",
  };

  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    context as unknown as CanvasRenderingContext2D,
  );
});

function frame(overrides: Partial<OverlayFrame> = {}): OverlayFrame {
  return {
    frame_index: 46,
    timestamp_s: 1.533,
    points: [
      { landmark: 11, x: 0.6, y: 0.44, state: "observed", visibility: 0.99 },
      { landmark: 12, x: 0.49, y: 0.42, state: "observed", visibility: 1 },
    ],
    shaft: null,
    ...overrides,
  };
}

function overlay(overrides: Partial<PoseOverlay> = {}): PoseOverlay {
  return {
    schema_version: 1,
    video_path: "/data/swing.mov",
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest: "a".repeat(64),
      size_bytes: 1,
    },
    geometry: { width: 720, height: 1280 },
    start_frame: 46,
    end_frame: 47,
    frames: [frame()],
    landmarks: [11, 12],
    connections: [[11, 12]],
    slow_motion_factor: 1,
    undistorted: false,
    club_tracked: false,
    warnings: [],
    ...overrides,
  };
}

describe("OverlayCanvas", () => {
  it("scales normalised coordinates by the clip's own geometry", () => {
    // The only arithmetic this component does. The engine sends fractions of the
    // displayed frame precisely so that this is a multiply and nothing else.
    render(<OverlayCanvas overlay={overlay()} frame={46} />);

    expect(
      recorder.arcs.map(([x, y]) => [Math.round(x), Math.round(y)]),
    ).toEqual([
      [432, 563],
      [353, 538],
    ]);
  });

  it("does not draw a landmark nothing supports", () => {
    // The case Phase 4 found on the down-the-line clip, where motion blur lost
    // the wrists for 1.92 s exactly when they moved fastest. Interpolating
    // across it would hide the one thing worth seeing.
    render(
      <OverlayCanvas
        overlay={overlay({
          frames: [
            frame({
              points: [
                {
                  landmark: 11,
                  x: 0.6,
                  y: 0.44,
                  state: "observed",
                  visibility: 0.99,
                },
                {
                  landmark: 12,
                  x: null,
                  y: null,
                  state: "blocked",
                  visibility: 0,
                },
              ],
            }),
          ],
        })}
        frame={46}
      />,
    );

    expect(recorder.arcs).toHaveLength(1);
  });

  it("does not draw an edge with a missing endpoint", () => {
    // Drawing to the last known position of an absent endpoint is how an
    // overlay grows a limb that points at nothing.
    render(
      <OverlayCanvas
        overlay={overlay({
          frames: [
            frame({
              points: [
                {
                  landmark: 11,
                  x: 0.6,
                  y: 0.44,
                  state: "observed",
                  visibility: 0.99,
                },
                {
                  landmark: 12,
                  x: null,
                  y: null,
                  state: "blocked",
                  visibility: 0,
                },
              ],
            }),
          ],
        })}
        frame={46}
      />,
    );

    expect(recorder.lines).toHaveLength(0);
  });

  it("strokes a filled point and fills an observed one", () => {
    // A hollow marker means the position is supported but this frame
    // contributed no observation to it, which a reader checking a number
    // against this frame needs to be able to see.
    render(
      <OverlayCanvas
        overlay={overlay({
          frames: [
            frame({
              points: [
                {
                  landmark: 11,
                  x: 0.6,
                  y: 0.44,
                  state: "observed",
                  visibility: 0.9,
                },
                {
                  landmark: 12,
                  x: 0.49,
                  y: 0.42,
                  state: "filled",
                  visibility: 0.2,
                },
              ],
            }),
          ],
        })}
        frame={46}
      />,
    );

    expect(recorder.fills).toBe(1);
    // One for the bone, one for the hollow marker.
    expect(recorder.strokes).toBe(2);
  });

  it("marks the club head only when the evidence reached it", () => {
    // Otherwise a head appears in every frame where the shaft was blurred or
    // pointing at the camera, which is exactly where nothing was found.
    const shaft = {
      grip_x: 0.5,
      grip_y: 0.5,
      tip_x: 0.7,
      tip_y: 0.8,
      reaches_head: false,
      confidence: 0.6,
    };

    render(
      <OverlayCanvas
        overlay={overlay({ frames: [frame({ shaft })], club_tracked: true })}
        frame={46}
      />,
    );
    const withoutHead = recorder.arcs.length;

    recorder.arcs = [];
    render(
      <OverlayCanvas
        overlay={overlay({
          frames: [frame({ shaft: { ...shaft, reaches_head: true } })],
          club_tracked: true,
        })}
        frame={46}
      />,
    );

    expect(recorder.arcs).toHaveLength(withoutHead + 1);
  });

  it("draws nothing for a frame outside the fetched range", () => {
    // A scrubber can outrun the window the overlay was fetched for. Drawing the
    // nearest frame instead would put a skeleton from somewhere else in the
    // swing over the picture.
    render(<OverlayCanvas overlay={overlay()} frame={200} />);

    expect(recorder.arcs).toHaveLength(0);
    expect(recorder.lines).toHaveLength(0);
  });

  it("draws nothing when there is no overlay", () => {
    render(<OverlayCanvas overlay={null} frame={0} />);

    expect(recorder.arcs).toHaveLength(0);
  });
});
