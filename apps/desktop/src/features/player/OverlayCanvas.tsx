/**
 * The skeleton and the club, drawn over the frame they were measured from.
 *
 * Coordinates arrive from the engine as fractions of the displayed frame, so
 * drawing is a multiply by the canvas size and nothing else. That is deliberate:
 * the aspect correction, the y flip and the lens are all applied on the engine
 * side, where the frame geometry and the calibration live. A canvas that did any
 * of it would be a second opinion about what a pixel means, and the first one is
 * in `analyzer/coordinates.py`.
 *
 * **Three states, drawn differently, because only one of them is a measurement
 * of this frame.** An observed landmark is solid; one the filter carried across
 * a gap is hollow; one nothing supports is not drawn at all. The last is the
 * case Phase 4 found on the down-the-line clip, where motion blur lost the
 * wrists for 1.92 s exactly when they were moving fastest — and a canvas that
 * interpolated across it would hide the single most useful thing on screen.
 */

import type { OverlayFrame, PoseOverlay } from "@gsa/types";
import { useEffect, useRef } from "react";

/**
 * Colours, as literals rather than as theme tokens.
 *
 * These are drawn onto a canvas over a photograph, so they have to hold against
 * whatever is behind them rather than against the app's background. A token
 * that changes with the light and dark themes would be legible in one of them.
 */
const INK = {
  bone: "rgba(56, 189, 248, 0.9)",
  observed: "rgb(56, 189, 248)",
  filled: "rgba(56, 189, 248, 0.45)",
  shaft: "rgb(251, 146, 60)",
  head: "rgb(249, 115, 22)",
} as const;

export function OverlayCanvas({
  overlay,
  frame,
  className,
}: {
  overlay: PoseOverlay | null;
  frame: number;
  className?: string;
}) {
  const canvas = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;

    const context = element.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, element.width, element.height);
    if (!overlay) return;

    const current = overlay.frames.find((entry) => entry.frame_index === frame);
    if (!current) return;

    draw(context, overlay, current, element.width, element.height);
  }, [overlay, frame]);

  // Sized in device pixels from the clip's own geometry, so the drawing
  // resolution follows the footage rather than the window. CSS stretches it to
  // the element box, which is set to the same aspect ratio by the player.
  return (
    <canvas
      ref={canvas}
      width={overlay?.geometry.width ?? 0}
      height={overlay?.geometry.height ?? 0}
      className={className}
      aria-hidden="true"
      data-testid="overlay-canvas"
    />
  );
}

function draw(
  context: CanvasRenderingContext2D,
  overlay: PoseOverlay,
  current: OverlayFrame,
  width: number,
  height: number,
): void {
  const positions = new Map<
    number,
    { x: number; y: number; filled: boolean }
  >();
  for (const point of current.points) {
    if (point.x === null || point.y === null) continue;
    positions.set(point.landmark, {
      x: point.x * width,
      y: point.y * height,
      filled: point.state === "filled",
    });
  }

  // Scaled to the frame, so the skeleton reads the same on a 720-wide clip and
  // a 4K one instead of vanishing on the second.
  const unit = Math.max(width, height) / 250;

  context.lineWidth = unit * 1.2;
  context.strokeStyle = INK.bone;
  context.lineCap = "round";
  for (const [from, to] of overlay.connections) {
    const a = positions.get(from);
    const b = positions.get(to);
    // An edge is drawn only when both ends are known. Drawing to the last known
    // position of a missing endpoint is how an overlay grows a limb that points
    // at nothing.
    if (!a || !b) continue;
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }

  for (const point of positions.values()) {
    context.beginPath();
    context.arc(point.x, point.y, unit * 1.6, 0, Math.PI * 2);
    if (point.filled) {
      // Hollow: the position is supported, but not by an observation of this
      // frame. A reader checking a number against this frame should be able to
      // see that this frame contributed nothing to it.
      context.strokeStyle = INK.filled;
      context.lineWidth = unit * 0.9;
      context.stroke();
    } else {
      context.fillStyle = INK.observed;
      context.fill();
    }
  }

  const shaft = current.shaft;
  if (!shaft) return;

  context.strokeStyle = INK.shaft;
  context.lineWidth = unit * 1.4;
  context.beginPath();
  context.moveTo(shaft.grip_x * width, shaft.grip_y * height);
  context.lineTo(shaft.tip_x * width, shaft.tip_y * height);
  context.stroke();

  // The club head is marked only when the evidence ran to the end of the club.
  // Otherwise the tip is where the image stopped drawing a line — the shaft was
  // blurred, or pointing at the camera — and putting a head there would invent
  // one in exactly the frames where nothing was found.
  if (shaft.reaches_head) {
    context.fillStyle = INK.head;
    context.beginPath();
    context.arc(
      shaft.tip_x * width,
      shaft.tip_y * height,
      unit * 2.2,
      0,
      Math.PI * 2,
    );
    context.fill();
  }
}
