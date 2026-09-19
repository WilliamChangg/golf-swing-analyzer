/**
 * The reconstruction, drawn from wherever the reader is standing.
 *
 * **SVG and a projection written here, rather than three.js and WebGL**, and the
 * roadmap asked for the latter. Three measured reasons, in the order they
 * decided it:
 *
 * 1. **The exit criterion is a measurement.** "Scrub stays in sync with video"
 *    has to be checked, and Phase 14 established what checking means here — a
 *    real browser, a real decoder, and the frame it reports painting looked back
 *    up. A WebGL canvas has no readable content: verifying it means a screenshot
 *    diff, which cannot say *which frame* is on screen. Every joint below is a
 *    DOM node carrying its landmark and its state, so `e2e/scene.spec.ts` can
 *    assert that the wrist moved to where frame 31's wrist is when the video
 *    reported painting frame 31.
 * 2. **The uncertainty ellipse needs the projection in hand.** What is drawn per
 *    joint is `A S A'` — the 3D covariance pushed through this camera's own
 *    Jacobian — which is the honest rendering of an anisotropic uncertainty and
 *    is not a scaled sphere. Owning the projection is what makes it available.
 * 3. **The scene is small.** Thirty-three joints, thirty-five bones and two
 *    trajectories is about 150 nodes a frame. `scripts/benchmark_viewport.py`
 *    measures the payload; the render is a React reconciliation over a list that
 *    size, which is not what a scene graph is for.
 *
 * The cost is real and worth stating: no depth buffer, no lighting, and drawing
 * order standing in for occlusion. For a stick figure of a body, in a viewport
 * whose job is to show where a measurement is uncertain rather than to look
 * solid, that is the right trade — but it would not survive a mesh.
 *
 * **There is no ground plane and no horizon.** `ReconstructionScene` carries
 * `CAMERA` metres, not `WORLD`: nothing in a stereo pair measures which way is
 * up or where the target line runs, so a grid on the floor would be a drawing of
 * an assumption. The reference camera's own up is used as a screen direction and
 * labelled as that.
 */

import type { ReconstructionScene, SceneFrame, ScenePoint } from "@gsa/types";
import { useCallback, useRef } from "react";

import {
  projectCovariance,
  projectPoint,
  UNCERTAINTY_SIGMAS,
  type Magnification,
  type Projected,
  type Vec3,
  type ViewCamera,
} from "@/features/scene/projection";
import { depthFrom } from "@/features/scene/scene";

/**
 * Colours as literals, for `OverlayCanvas`'s reason: this is drawn against its
 * own dark ground rather than against the app's surface, so a theme token would
 * be legible in one theme and not the other.
 */
const INK = {
  ground: "rgb(15, 23, 42)",
  bone: "rgba(56, 189, 248, 0.85)",
  joint: "rgb(56, 189, 248)",
  uncertainty: "rgba(56, 189, 248, 0.16)",
  uncertaintyEdge: "rgba(125, 211, 252, 0.5)",
  trajectory: "rgba(251, 146, 60, 0.85)",
  cameraMark: "rgba(148, 163, 184, 0.9)",
  cameraActive: "rgba(250, 204, 21, 0.95)",
} as const;

export interface SceneViewportProps {
  scene: ReconstructionScene;
  frame: SceneFrame | null;
  camera: ViewCamera | null;
  showUncertainty: boolean;
  /** How far the 1-sigma ellipses are exaggerated. Drawn in the corner. */
  magnification: Magnification;
  showTrajectories: boolean;
  showCameras: boolean;
  /** Pointer drag, in viewport pixels. */
  onDrag: (dx: number, dy: number) => void;
  /** Wheel or pinch, as a multiplier on the orbit distance. */
  onZoom: (factor: number) => void;
}

export function SceneViewport({
  scene,
  frame,
  camera,
  showUncertainty,
  magnification,
  showTrajectories,
  showCameras,
  onDrag,
  onZoom,
}: SceneViewportProps) {
  const dragging = useRef<{ x: number; y: number } | null>(null);

  // Pointer capture keeps a drag alive once the pointer leaves the viewport,
  // which is what makes a turn of more than a screen width possible. It is a
  // nicety rather than the mechanism — the drag is tracked by the ref either way
  // — and it is not implemented everywhere (jsdom has none of it), so it is
  // asked for and not depended on.
  const onPointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      dragging.current = { x: event.clientX, y: event.clientY };
      event.currentTarget.setPointerCapture?.(event.pointerId);
    },
    [],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const from = dragging.current;
      if (!from) return;
      onDrag(event.clientX - from.x, event.clientY - from.y);
      dragging.current = { x: event.clientX, y: event.clientY };
    },
    [onDrag],
  );

  const onPointerUp = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      dragging.current = null;
      if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [],
  );

  const width = camera?.width ?? 1920;
  const height = camera?.height ?? 1080;
  // Scaled to the frame so the figure reads the same on a 720-wide reference
  // clip and a 4K one, exactly as `OverlayCanvas` does.
  const unit = Math.max(width, height) / 250;

  return (
    <svg
      viewBox={`0 0 ${String(width)} ${String(height)}`}
      className="size-full touch-none select-none"
      style={{ background: INK.ground, cursor: "grab" }}
      role="img"
      aria-label={
        frame
          ? `Reconstruction at frame ${String(frame.frame_index)}, ${String(frame.reconstructed)} of ${String(frame.points.length)} landmarks`
          : "No reconstruction at this frame"
      }
      data-testid="scene-viewport"
      data-frame={frame ? String(frame.frame_index) : ""}
      data-reconstructed={frame ? String(frame.reconstructed) : "0"}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onWheel={(event) => {
        onZoom(Math.exp(event.deltaY * 0.001));
      }}
    >
      {camera ? (
        <>
          {showCameras ? (
            <Cameras scene={scene} camera={camera} unit={unit} />
          ) : null}
          {showTrajectories ? (
            <Trajectories scene={scene} camera={camera} unit={unit} />
          ) : null}
          {frame ? (
            <Figure
              scene={scene}
              frame={frame}
              camera={camera}
              unit={unit}
              showUncertainty={showUncertainty}
              magnification={magnification}
            />
          ) : null}
        </>
      ) : null}

      {/* The exaggeration, in the picture rather than beside it. An ellipse
          without its factor is a size claim, and this one is not to scale. */}
      {showUncertainty && frame && frame.reconstructed > 0 ? (
        <text
          x={unit * 2}
          y={height - unit * 2}
          fill={INK.uncertaintyEdge}
          fontSize={unit * 3.2}
          data-testid="scene-magnification"
        >
          uncertainty &times;{magnification} &middot; {UNCERTAINTY_SIGMAS}
          &sigma;
        </text>
      ) : null}

      {frame === null || frame.reconstructed === 0 ? (
        <text
          x={width / 2}
          y={height / 2}
          textAnchor="middle"
          fill="rgba(226, 232, 240, 0.75)"
          fontSize={unit * 9}
          data-testid="scene-empty"
        >
          {frame === null
            ? "This frame is outside the reconstruction"
            : "Nothing was reconstructed at this frame"}
        </text>
      ) : null}
    </svg>
  );
}

/**
 * The skeleton at one instant.
 *
 * Bones are drawn only where **both** ends were reconstructed, for
 * `OverlayCanvas`'s reason: an edge drawn to a missing endpoint's last known
 * position is a limb pointing at nothing. Joints are drawn back to front so the
 * nearer ones overlap the further, which is the only occlusion cue an SVG has.
 */
function Figure({
  scene,
  frame,
  camera,
  unit,
  showUncertainty,
  magnification,
}: {
  scene: ReconstructionScene;
  frame: SceneFrame;
  camera: ViewCamera;
  unit: number;
  showUncertainty: boolean;
  magnification: number;
}) {
  const drawn = new Map<number, { at: Projected; point: ScenePoint }>();
  for (const point of frame.points) {
    if (!point.position) continue;
    const at = projectPoint(camera, point.position);
    if (at) drawn.set(point.landmark, { at, point });
  }

  const sorted = [...drawn.values()].sort(
    (a, b) =>
      depthFrom(camera, b.point.position as Vec3) -
      depthFrom(camera, a.point.position as Vec3),
  );

  return (
    <g>
      {showUncertainty
        ? sorted.map(({ at, point }) => {
            const { position, uncertainty } = point;
            if (!position || !uncertainty) return null;
            const ellipse = projectCovariance(camera, position, uncertainty);
            if (!ellipse) return null;
            return (
              <ellipse
                key={`sigma-${String(point.landmark)}`}
                cx={at.x}
                cy={at.y}
                rx={Math.max(
                  ellipse.rx * UNCERTAINTY_SIGMAS * magnification,
                  0.1,
                )}
                ry={Math.max(
                  ellipse.ry * UNCERTAINTY_SIGMAS * magnification,
                  0.1,
                )}
                transform={`rotate(${String(ellipse.angleDeg)} ${String(at.x)} ${String(at.y)})`}
                fill={INK.uncertainty}
                stroke={INK.uncertaintyEdge}
                strokeWidth={unit * 0.25}
                data-testid="scene-uncertainty"
                data-landmark={String(point.landmark)}
              />
            );
          })
        : null}

      {scene.connections.map(([from, to]) => {
        const a = drawn.get(from);
        const b = drawn.get(to);
        if (!a || !b) return null;
        return (
          <line
            key={`bone-${String(from)}-${String(to)}`}
            x1={a.at.x}
            y1={a.at.y}
            x2={b.at.x}
            y2={b.at.y}
            stroke={INK.bone}
            strokeWidth={unit * 1.1}
            strokeLinecap="round"
            data-testid="scene-bone"
          />
        );
      })}

      {sorted.map(({ at, point }) => (
        <circle
          key={`joint-${String(point.landmark)}`}
          cx={at.x}
          cy={at.y}
          r={unit * 1.5}
          fill={INK.joint}
          data-testid="scene-joint"
          data-landmark={String(point.landmark)}
        />
      ))}
    </g>
  );
}

/**
 * The wrists' paths through the swing, broken wherever a point was refused.
 *
 * One polyline per unbroken run rather than one per trajectory. A single
 * polyline with the gaps filtered out would draw a straight line across the
 * refusal — which is exactly what `SceneTrajectory` sends nulls to prevent, and
 * what a chord across a missing downswing would look most convincing doing.
 */
function Trajectories({
  scene,
  camera,
  unit,
}: {
  scene: ReconstructionScene;
  camera: ViewCamera;
  unit: number;
}) {
  return (
    <g data-testid="scene-trajectories">
      {(scene.trajectories ?? []).flatMap((trajectory) => {
        const runs: string[][] = [];
        let run: string[] = [];
        for (const point of trajectory.points) {
          const at = point ? projectPoint(camera, point) : null;
          if (!at) {
            if (run.length > 1) runs.push(run);
            run = [];
            continue;
          }
          run.push(`${String(at.x)},${String(at.y)}`);
        }
        if (run.length > 1) runs.push(run);

        return runs.map((entries, index) => (
          <polyline
            key={`${trajectory.name}-${String(index)}`}
            points={entries.join(" ")}
            fill="none"
            stroke={INK.trajectory}
            strokeWidth={unit * 0.7}
            strokeLinecap="round"
            data-testid="scene-trajectory-run"
            data-landmark={String(trajectory.landmark)}
          />
        ));
      })}
    </g>
  );
}

/**
 * Where the two cameras stood, and which way they looked.
 *
 * Worth the twenty lines because **the angle between them is the single largest
 * thing deciding what the reconstruction is worth** — Phase 9 measured the error
 * growing 4.7x as it closes from 90 degrees to 15 — and it is a fact about the
 * capture that no number on a panel makes as legible as two markers and the
 * angle between them.
 */
function Cameras({
  scene,
  camera,
  unit,
}: {
  scene: ReconstructionScene;
  camera: ViewCamera;
  unit: number;
}) {
  return (
    <g data-testid="scene-cameras">
      {scene.cameras.map((entry) => {
        const at = projectPoint(camera, entry.position);
        if (!at) return null;
        const ahead = projectPoint(camera, {
          x: entry.position.x + entry.forward.x * scene.radius_m * 0.5,
          y: entry.position.y + entry.forward.y * scene.radius_m * 0.5,
          z: entry.position.z + entry.forward.z * scene.radius_m * 0.5,
        });
        const colour =
          entry.kind === "reference" ? INK.cameraActive : INK.cameraMark;
        return (
          <g key={entry.kind} data-testid="scene-camera" data-kind={entry.kind}>
            {ahead ? (
              <line
                x1={at.x}
                y1={at.y}
                x2={ahead.x}
                y2={ahead.y}
                stroke={colour}
                strokeWidth={unit * 0.5}
                strokeDasharray={`${String(unit * 1.5)} ${String(unit)}`}
              />
            ) : null}
            <rect
              x={at.x - unit * 1.8}
              y={at.y - unit * 1.3}
              width={unit * 3.6}
              height={unit * 2.6}
              fill="none"
              stroke={colour}
              strokeWidth={unit * 0.45}
              rx={unit * 0.4}
            />
            <text
              x={at.x}
              y={at.y - unit * 2.2}
              textAnchor="middle"
              fill={colour}
              fontSize={unit * 3}
            >
              {entry.role.replace(/_/g, " ")}
            </text>
          </g>
        );
      })}
    </g>
  );
}
