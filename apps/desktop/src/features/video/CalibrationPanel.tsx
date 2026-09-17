/**
 * Camera calibration: run it, and show why the result should or should not be used.
 *
 * The panel is built around one fact that the numbers alone do not convey. A
 * calibration reports a reprojection error, everybody reads it as the quality
 * of the calibration, and it is not: it says how well the model fits the board
 * views it was given, and a capture that determines nothing fits *better* than
 * a good one. So the residual is shown, labelled with what it actually
 * measures, and the thing that decides whether the calibration may be used —
 * coverage — is shown next to it with equal weight.
 *
 * The map is the reason this is a panel rather than a table. A person who has
 * just filmed a board and been told their capture is too centred has to work
 * out what to do differently; a picture of where their board actually went says
 * it immediately, and no column of numbers does.
 */

import type {
  BoardObservation,
  CameraCalibration,
  CoverageReport,
  EngineError,
  ProgressUpdate,
} from "@gsa/types";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Grid3x3, Loader2, ScanSearch } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import { calibrateCamera, onProgress, progressFraction } from "@/lib/ipc";

/** Engine method name, used to filter progress off the shared channel. */
const TASK = "detect_board";

type State =
  | { phase: "idle" }
  | { phase: "running"; update: ProgressUpdate | null }
  | { phase: "done"; result: CameraCalibration }
  | { phase: "failed"; error: EngineError };

function Field({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "good" | "warn" | "bad";
}) {
  const colour =
    tone === "bad"
      ? "text-destructive"
      : tone === "warn"
        ? "text-amber-600 dark:text-amber-500"
        : tone === "good"
          ? "text-emerald-600 dark:text-emerald-500"
          : "";

  return (
    <div className="space-y-0.5">
      <div className="flex justify-between gap-4 text-sm">
        <dt className="text-muted-foreground shrink-0">{label}</dt>
        <dd className={`truncate font-mono ${colour}`} title={value}>
          {value}
        </dd>
      </div>
      {note ? (
        <p className="text-muted-foreground/80 text-xs leading-snug">{note}</p>
      ) : null}
    </div>
  );
}

/**
 * Where each board view landed in the frame, and how well it reprojected.
 *
 * Drawn at the frame's own aspect ratio, so a capture that never left the
 * middle looks like what it is. Each view is one dot at its corner centroid;
 * dropped views are drawn hollow, because "you shot forty frames and eight were
 * useful" is a different problem from "you shot eight frames".
 */
function CoverageMap({
  observations,
  width,
  height,
}: {
  observations: BoardObservation[];
  width: number;
  height: number;
}) {
  const used = observations.filter((entry) => entry.used);
  const worst = used.reduce(
    (peak, entry) => Math.max(peak, entry.reprojection_rms_px ?? 0),
    0,
  );

  return (
    <figure className="space-y-1">
      <svg
        viewBox={`0 0 ${String(width)} ${String(height)}`}
        className="bg-muted/40 ring-border w-full rounded ring-1"
        role="img"
        aria-label={`Where the board was seen: ${String(used.length)} view(s) used of ${String(observations.length)} detected`}
      >
        {/* The outer fifth of the radius, where lens distortion is measurable
            at all. A capture with nothing out here has fitted its distortion
            coefficients to almost no evidence. */}
        <ellipse
          cx={width / 2}
          cy={height / 2}
          rx={(width / 2) * 0.7}
          ry={(height / 2) * 0.7}
          fill="none"
          stroke="currentColor"
          strokeWidth={Math.max(width, height) / 400}
          strokeDasharray="12 10"
          className="text-muted-foreground/40"
        />
        {observations.map((entry) => {
          const radius = Math.max(width, height) / 110;
          const error = entry.reprojection_rms_px ?? 0;
          const heat = worst > 0 ? error / worst : 0;
          return (
            <circle
              key={entry.frame}
              cx={entry.centroid_x}
              cy={entry.centroid_y}
              r={radius}
              fill={
                entry.used
                  ? `rgb(${String(Math.round(60 + heat * 180))} 110 220)`
                  : "none"
              }
              stroke="currentColor"
              strokeWidth={radius / 3}
              className={
                entry.used ? "text-transparent" : "text-muted-foreground/50"
              }
            >
              <title>
                {`frame ${String(entry.frame)}: ${String(entry.corners)} corners` +
                  (entry.reprojection_rms_px == null
                    ? " (not used)"
                    : `, ${entry.reprojection_rms_px.toFixed(2)} px rms` +
                      (entry.tilt_deg == null
                        ? ""
                        : `, ${entry.tilt_deg.toFixed(0)}° tilt`) +
                      (entry.distance_m == null
                        ? ""
                        : `, ${entry.distance_m.toFixed(2)} m`))}
              </title>
            </circle>
          );
        })}
      </svg>
      <figcaption className="text-muted-foreground text-xs">
        Where the board was seen, in the frame. Hollow dots were detected and
        not used. Inside the dashed ellipse the lens barely bends, so views out
        beyond it are the only evidence about distortion.
      </figcaption>
    </figure>
  );
}

function CoverageFields({ coverage }: { coverage: CoverageReport }) {
  return (
    <dl className="space-y-2">
      <Field
        label="frame area visited"
        value={`${String(Math.round(coverage.image_fraction * 100))}%`}
        tone={coverage.image_fraction >= 0.35 ? "good" : "bad"}
      />
      <Field
        label="tilt spread"
        value={`${coverage.tilt_range_deg.toFixed(0)}°`}
        tone={coverage.tilt_range_deg >= 20 ? "good" : "bad"}
        note="The field that catches the failure a residual cannot: a board held at one orientation leaves the focal length unmeasured while fitting beautifully."
      />
      <Field
        label="corners near the edge"
        value={`${String(Math.round(coverage.edge_fraction * 100))}%`}
        tone={coverage.edge_fraction >= 0.05 ? "good" : "warn"}
      />
      <Field
        label="apparent size spread"
        value={`${coverage.scale_range.toFixed(2)}×`}
        tone={coverage.scale_range >= 1.5 ? "good" : "warn"}
      />
      <Field label="views used" value={String(coverage.views)} />
      <Field label="corners" value={String(coverage.corners)} />
    </dl>
  );
}

function Result({ result }: { result: CameraCalibration }) {
  const intrinsics = result.intrinsics;
  const quality = result.quality;
  const focalRatio =
    intrinsics.fx_uncertainty == null
      ? null
      : intrinsics.fx_uncertainty / intrinsics.fx;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <StatusBadge status={result.usable ? "ok" : "error"} />
        <span className="text-sm">
          {result.usable
            ? "This calibration may be used."
            : "This calibration will not be used."}
        </span>
      </div>

      {result.refusal ? (
        <p className="border-destructive/40 bg-destructive/5 rounded border p-3 text-sm leading-relaxed">
          {result.refusal}
        </p>
      ) : null}

      <div className="grid gap-6 sm:grid-cols-2">
        <section className="space-y-3">
          <h3 className="text-sm font-medium">What was measured</h3>
          <dl className="space-y-2">
            <Field
              label="focal length"
              value={`fx ${intrinsics.fx.toFixed(1)}, fy ${intrinsics.fy.toFixed(1)} px`}
            />
            <Field
              label="optical centre"
              value={`${intrinsics.cx.toFixed(1)}, ${intrinsics.cy.toFixed(1)} px`}
            />
            <Field
              label="field of view"
              value={`${horizontalFov(intrinsics.fx, intrinsics.image_width).toFixed(1)}° across`}
              note="The one number you can check against the lens without any tooling: a phone's main camera sees about 65–70°."
            />
            <Field
              label="distortion"
              value={
                intrinsics.distortion.length === 0
                  ? "none fitted"
                  : intrinsics.distortion
                      .map((value) => value.toFixed(4))
                      .join(", ")
              }
            />
            <Field
              label="frame size"
              value={`${String(intrinsics.image_width)}×${String(intrinsics.image_height)}`}
              note="A calibration describes one camera at one setting. It is refused on footage of any other frame size."
            />
          </dl>
        </section>

        <section className="space-y-3">
          <h3 className="text-sm font-medium">How well it is known</h3>
          <dl className="space-y-2">
            <Field
              label="reprojection error"
              value={`${quality.rms_reprojection_px.toFixed(3)} px rms`}
              note="How well the model fits these views. Necessary, and not evidence that the views determined it — a capture that constrains nothing fits better than one that does."
            />
            <Field
              label="worst corner"
              value={`${quality.max_reprojection_px.toFixed(3)} px`}
            />
            <Field
              label="focal uncertainty"
              value={
                focalRatio === null
                  ? "not reported"
                  : `${(intrinsics.fx_uncertainty ?? 0).toFixed(2)} px (${(focalRatio * 100).toFixed(2)}%)`
              }
              tone={
                focalRatio === null
                  ? undefined
                  : focalRatio <= 0.02
                    ? "good"
                    : "bad"
              }
              note="Whether these views pinned the focal length down, from the fit's own covariance."
            />
            <Field
              label="spare degrees of freedom"
              value={String(quality.degrees_of_freedom)}
            />
          </dl>
        </section>
      </div>

      <section className="space-y-3">
        <h3 className="text-sm font-medium">
          Coverage — what decides whether this is usable
        </h3>
        <div className="grid gap-6 sm:grid-cols-2">
          <CoverageFields coverage={quality.coverage} />
          <CoverageMap
            observations={result.detection.observations}
            width={intrinsics.image_width}
            height={intrinsics.image_height}
          />
        </div>
      </section>

      <p className="text-muted-foreground text-xs">
        Scanned {result.detection.frames_scanned} frame(s); board found in{" "}
        {result.detection.frames_with_board}; {result.detection.views_used}{" "}
        distinct views kept.
      </p>

      {/* Both lists are optional in the schema, because Pydantic defaults them
          to empty. Treating absent as empty here keeps the two cases the same
          thing, which they are. */}
      {[...(result.detection.warnings ?? []), ...(result.warnings ?? [])].map(
        (warning) => (
          <p
            key={warning}
            className="text-muted-foreground border-l-2 border-amber-500/60 pl-3 text-xs leading-relaxed"
          >
            {warning}
          </p>
        ),
      )}
    </div>
  );
}

/** Horizontal field of view in degrees, mirroring the Python property. */
function horizontalFov(fx: number, width: number): number {
  return (2 * Math.atan(width / (2 * fx)) * 180) / Math.PI;
}

function ProgressBar({ update }: { update: ProgressUpdate | null }) {
  const fraction = update ? progressFraction(update) : null;
  const percent = fraction === null ? null : Math.round(fraction * 100);

  return (
    <div className="space-y-2">
      <div className="text-muted-foreground flex justify-between text-sm">
        <span>
          Looking for the board
          {update?.detail ? ` — ${update.detail}` : ""}
        </span>
        <span className="font-mono">
          {percent === null ? "—" : `${String(percent)}%`}
        </span>
      </div>
      <div
        className="bg-muted h-2 w-full overflow-hidden rounded-full"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        {...(percent === null ? {} : { "aria-valuenow": percent })}
        aria-label="Board detection progress"
      >
        <div
          className={`bg-primary h-full transition-[width] duration-200 ${
            percent === null ? "w-1/3 animate-pulse" : ""
          }`}
          style={
            percent === null ? undefined : { width: `${String(percent)}%` }
          }
        />
      </div>
    </div>
  );
}

/** Video containers a board capture is likely to arrive in. */
const BOARD_VIDEO_EXTENSIONS = ["mov", "mp4", "m4v", "avi", "mkv"];

/**
 * @param source Board footage to calibrate from. Omitted, the panel asks for it.
 *
 * The source is **not** the swing clip, and the panel takes its own rather than
 * inheriting the screen's for that reason: a swing clip contains no board, so
 * calibrating from it would find nothing and report a detection failure that
 * looks like a lighting problem. A calibration is a separate recording of a
 * separate thing.
 */
export function CalibrationPanel({
  source: initialSource,
  squareLengthMm = 35,
}: {
  source?: string;
  squareLengthMm?: number;
}) {
  const [source, setSource] = useState<string | null>(initialSource ?? null);
  const [state, setState] = useState<State>({ phase: "idle" });
  const unlisten = useRef<(() => void) | null>(null);

  // Only the unsubscribe is tied to the component's life. The subscription
  // itself happens in `run`, matching PosePanel: a listener registered on mount
  // would be live for every screen that renders this panel, and one registered
  // after the call would miss the first updates -- including the only one a
  // short capture produces.
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  const run = useCallback(async () => {
    if (source === null) return;
    setState({ phase: "running", update: null });

    unlisten.current = await onProgress(
      (update) => {
        setState((current) =>
          current.phase === "running" ? { phase: "running", update } : current,
        );
      },
      { task: TASK },
    );

    try {
      const result = await calibrateCamera(source, { squareLengthMm });
      // A refused calibration is a result, not an error: it carries the
      // numbers that say what to reshoot.
      setState(
        result.ok
          ? { phase: "done", result: result.value }
          : { phase: "failed", error: result.error },
      );
    } finally {
      unlisten.current?.();
      unlisten.current = null;
    }
  }, [source, squareLengthMm]);

  const choose = useCallback(async () => {
    // Either a video of the board or a directory of photographs of it. A
    // directory is offered because a set of stills is what a careful capture
    // usually is, and because the engine treats one as a set someone chose
    // rather than as frames to thin.
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Board footage", extensions: BOARD_VIDEO_EXTENSIONS }],
    });
    if (typeof selected === "string") {
      setSource(selected);
      setState({ phase: "idle" });
    }
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Grid3x3 className="size-4" aria-hidden />
          Camera calibration
        </CardTitle>
        <CardDescription>
          Measure this camera&rsquo;s lens from footage of a Charuco board. A
          calibrated camera knows which direction each pixel came from — not how
          far away anything was, which takes two of them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {state.phase === "idle" ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="secondary" onClick={() => void choose()}>
                <FolderOpen className="size-4" aria-hidden />
                Choose board footage
              </Button>
              <Button onClick={() => void run()} disabled={source === null}>
                <ScanSearch className="size-4" aria-hidden />
                Find the board
              </Button>
            </div>
            <p className="text-muted-foreground truncate font-mono text-xs">
              {source ?? "No board footage chosen yet."}
            </p>
          </div>
        ) : null}

        {state.phase === "running" ? (
          <div className="space-y-4">
            <Button disabled>
              <Loader2 className="size-4 animate-spin" aria-hidden />
              Detecting
            </Button>
            <ProgressBar update={state.update} />
          </div>
        ) : null}

        {state.phase === "failed" ? (
          <div className="space-y-4">
            <EngineErrorPanel error={state.error} onRetry={() => void run()} />
          </div>
        ) : null}

        {state.phase === "done" ? (
          <div className="space-y-6">
            <Result result={state.result} />
            <Button variant="secondary" onClick={() => void run()}>
              <ScanSearch className="size-4" aria-hidden />
              Run again
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
