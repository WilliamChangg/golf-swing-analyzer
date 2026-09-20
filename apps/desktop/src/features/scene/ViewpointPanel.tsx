/**
 * What the picture beside this is worth, and how much of that it is showing.
 *
 * Every panel in this app exists so a number cannot be read without its
 * qualifications. This one has a job the others do not: **the qualification
 * changes when the reader moves the mouse.**
 *
 * A reconstructed joint's uncertainty is an ellipsoid, not a radius. Seen across
 * its long axis it draws as a smear a reader can discount; seen along it, it
 * hides behind the joint and the picture looks exactly as confident as a perfect
 * one would. `scripts/benchmark_viewport.py --sweep convergence` measures what
 * that costs: as the two cameras close from 90 degrees to 15, the median
 * uncertainty grows from 5.5 mm to 25.6 mm while the uncertainty *visible from
 * the reference camera* stays at 5.3 mm at every step. Flat, across a range over
 * which the thing it appears to describe grows nearly five-fold — which is
 * Phase 8's tilt sweep and Phase 9's convergence sweep for a third time.
 *
 * So the two numbers are shown side by side and labelled as what they are: what
 * the measurement is worth, and what this viewpoint lets a reader see of it.
 */

import type { ReconstructionScene, SceneFrame } from "@gsa/types";

import { Separator } from "@/components/ui/separator";
import { StatusBadge, type Status } from "@/components/ui/status-badge";
import type { ViewpointHonesty } from "@/features/scene/scene";
import { refusalsIn } from "@/features/scene/scene";

/**
 * How a refused landmark is explained, exhaustively.
 *
 * A `Record` over the contract's own enum, so a reason added in Python is a type
 * error here until it is given words — and the words are the *fix*, because the
 * five reasons have five unrelated ones and a reader looking at a hole in a
 * skeleton wants to know which.
 */
const REFUSAL_LABEL: Record<string, string> = {
  not_seen: "not seen in both views",
  outside_overlap: "outside the clips' overlap",
  ill_conditioned: "rays too shallow to intersect",
  reprojection: "the two views disagree",
  uncertain: "determined, but not well enough",
};

/** Where the visible fraction stops being reassuring. Stated, not measured. */
const HONEST_VIEW = 0.7;
const FLATTERING_VIEW = 0.4;

function millimetres(value: number | null): string {
  return value === null ? "—" : `${(value * 1000).toFixed(1)} mm`;
}

function Row({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: string;
}) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 py-1">
      <span className="text-muted-foreground text-sm">{label}</span>
      <span className="text-right">
        <span className={`font-mono text-sm ${tone ?? ""}`}>{value}</span>
        {note ? (
          <span className="text-muted-foreground ml-2 text-xs">{note}</span>
        ) : null}
      </span>
    </div>
  );
}

export function ViewpointPanel({
  scene,
  frame,
  honesty,
}: {
  scene: ReconstructionScene;
  frame: SceneFrame | null;
  honesty: ViewpointHonesty;
}) {
  const quality = scene.report.quality ?? null;
  const refusals = frame ? refusalsIn(frame) : new Map<string, number>();

  const visible = honesty.visible;
  const viewTone =
    visible === null
      ? ""
      : visible >= HONEST_VIEW
        ? "text-status-ok"
        : visible >= FLATTERING_VIEW
          ? "text-status-degraded"
          : "text-status-missing";

  // `stereo` is the only status a scene can carry -- an uncalibrated project has
  // no third dimension to draw -- but it is shown rather than assumed, because
  // three dimensions on a screen is itself a claim about calibration and the
  // claim should be visible next to the picture making it.
  const calibration: Status = scene.calibration === "stereo" ? "ok" : "missing";

  return (
    <div className="space-y-2" data-testid="viewpoint-panel">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={calibration} className="capitalize" />
        <span className="text-muted-foreground text-xs">
          {scene.reference_name} &amp; {scene.target_name}, calibrated as a pair
          {scene.report.convergence_deg
            ? `, ${scene.report.convergence_deg.toFixed(0)}° apart`
            : null}
          {scene.report.baseline_m
            ? ` on a ${scene.report.baseline_m.toFixed(2)} m baseline`
            : null}
        </span>
      </div>

      <Separator />

      <Row
        label="Landmarks drawn"
        value={
          frame
            ? `${String(frame.reconstructed)} / ${String(frame.points.length)}`
            : "—"
        }
        note={
          frame ? `frame ${String(frame.frame_index)}` : "outside the range"
        }
      />
      <Row
        label="Rays meet at"
        value={
          quality?.median_convergence_deg == null
            ? "—"
            : `${quality.median_convergence_deg.toFixed(0)}°`
        }
        note="the gate; depth error scales as 1/sin"
      />
      <Row
        label="Uncertainty"
        value={millimetres(honesty.sigmaM)}
        note="worst direction, at this frame's median"
      />

      <Separator />

      <Row
        label="Visible from here"
        value={visible === null ? "—" : `${(visible * 100).toFixed(0)}%`}
        tone={viewTone}
        note="of the uncertainty, across the line of sight"
      />
      <Row
        label="Drawn as"
        value={millimetres(honesty.apparentSigmaM)}
        note="what this viewpoint makes it look like"
      />

      {visible !== null && visible < HONEST_VIEW ? (
        <p
          className="text-status-degraded text-xs"
          data-testid="viewpoint-warning"
        >
          {(100 - visible * 100).toFixed(0)}% of what is unknown about these
          joints is pointing along the line of sight, so it is hidden behind
          them rather than drawn. Orbit the view to bring it into the picture —
          the error does not change, only whether you can see it.
        </p>
      ) : null}

      {refusals.size > 0 ? (
        <>
          <Separator />
          <ul className="space-y-1 text-xs" data-testid="scene-refusals">
            {[...refusals.entries()].map(([reason, count]) => (
              <li key={reason} className="flex justify-between gap-3">
                <span className="text-muted-foreground">
                  {REFUSAL_LABEL[reason] ?? reason}
                </span>
                <span className="font-mono">{count}</span>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
