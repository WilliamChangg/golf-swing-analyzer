/**
 * What differs between two recordings, and the longer list of what cannot be told.
 *
 * Laid out against its own contents, exactly as `FindingsPanel` is and for a
 * sharper version of the same reason: on the reference footage in this
 * repository a comparison produces at most four differences and refuses
 * thirty-odd. The refusals are the panel.
 *
 * Three presentation decisions are load-bearing.
 *
 * **There is no better.** A difference has a direction and a size and nothing
 * else — no colour meaning good, no arrow meaning improvement, no ordering by
 * anything but the engine's own. Which direction of a quantity is desirable is
 * not something this system measures, and a green tick would assert it without
 * ever saying so.
 *
 * **An unresolved difference shows its numbers.** What is refused is the claim
 * that the two swings differ, not the values themselves, so both are printed
 * next to the bracket that swallowed them. A reader who cannot see the numbers
 * cannot see how close the call was, and "0.01 inside a bracket of 0.40" and
 * "identical" are very different recordings.
 *
 * **The bracket is itemised.** A clock term is fixed by a faster camera, a
 * measurement term by a steadier landmark and a camera term by putting the
 * tripod back where it was, so the three are listed rather than summed into one
 * number a reader cannot act on.
 */

import type {
  CameraAgreement,
  MetricDifference,
  RefusedDifference,
  SwingComparison,
} from "@gsa/types";
import { CircleSlash, MoveHorizontal } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Why a quantity was not compared, in the reader's terms.
 *
 * Exhaustive, so a refusal added to the Python contract is a TypeScript error
 * here until it is given words. A system that refuses most of what it is asked
 * looks broken rather than careful if the refusals arrive as raw enum names.
 */
const REFUSAL_NOTE: Record<RefusedDifference["refusal"], string> = {
  no_swing: "one of these clips contains no detected swing",
  missing: "only one of the two clips produced this measurement",
  view_mismatch:
    "the two clips were filmed from different camera positions, so this is not one quantity measured twice",
  camera_moved:
    "the cameras were not in the same place, and nothing here separates that from a player built differently",
  calibration_mismatch:
    "one clip's landmarks had the lens removed and the other's did not",
  supplied_timebase:
    "a slow-motion factor was supplied rather than measured, so a duration in seconds is a guess multiplied by a measurement",
  low_confidence:
    "one of the two measurements is too weakly determined to support a difference",
  no_bracket:
    "nothing has quantified how finely either clip pins this value down, so there is no distance a difference could clear",
  unresolved:
    "the values differ by less than these two recordings can resolve — which is not a finding that the swings agree",
};

function measure(value: number, unit: string): string {
  const digits = unit === "seconds" ? 3 : 2;
  return value.toFixed(digits);
}

function DifferenceRow({ entry }: { entry: MetricDifference }) {
  return (
    <li className="space-y-2 rounded-md border p-4" data-testid="difference">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="font-medium">{entry.label}</h4>
        <span className="text-muted-foreground text-xs">
          {entry.unit.replace(/_/g, " ")}
        </span>
      </div>

      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono text-sm">
        <span>{measure(entry.reference_value, entry.unit)}</span>
        <MoveHorizontal
          className="text-muted-foreground size-3.5"
          aria-hidden="true"
        />
        <span>{measure(entry.target_value, entry.unit)}</span>
        <span className="font-semibold">
          {entry.difference > 0 ? "+" : ""}
          {measure(entry.difference, entry.unit)}
        </span>
        {/* The word, not a colour. "higher" is a direction; a green arrow would
            be a verdict, and this system has no source for one. */}
        <span className="text-muted-foreground font-sans text-xs">
          the target measures {entry.direction}
        </span>
      </div>

      <p className="text-muted-foreground text-xs">{entry.interpretation}</p>

      <dl className="space-y-1 text-xs">
        <div className="flex justify-between gap-2">
          <dt className="text-muted-foreground">Had to clear</dt>
          <dd className="font-mono">{measure(entry.bracket, entry.unit)}</dd>
        </div>
        {(entry.terms ?? []).map((term) => (
          <div key={term.source} className="flex justify-between gap-2 pl-3">
            <dt className="text-muted-foreground">{term.source}</dt>
            <dd className="font-mono">{measure(term.value, entry.unit)}</dd>
          </div>
        ))}
        <div className="flex justify-between gap-2">
          <dt className="text-muted-foreground">Cleared it by</dt>
          <dd className="font-mono">{measure(entry.margin, entry.unit)}</dd>
        </div>
      </dl>

      <p className="text-muted-foreground text-xs">
        Reference frames{" "}
        <span className="font-mono">{frameSpan(entry.reference_frames)}</span>,
        target frames{" "}
        <span className="font-mono">{frameSpan(entry.target_frames)}</span>.
      </p>
    </li>
  );
}

function frameSpan(frames: number[]): string {
  if (frames.length === 0) return "none cited";
  if (frames.length === 1) return String(frames[0]);
  return `${String(frames[0])}–${String(frames[frames.length - 1])}`;
}

function RefusalRow({ entry }: { entry: RefusedDifference }) {
  const both = entry.reference_value != null && entry.target_value != null;
  return (
    <li className="flex flex-col gap-1 text-sm" data-testid="refusal">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="font-medium">{entry.label}</span>
        {both ? (
          <span className="font-mono text-xs">
            {entry.reference_value?.toFixed(2)} vs{" "}
            {entry.target_value?.toFixed(2)}
            {entry.bracket == null
              ? ""
              : ` · bracket ${entry.bracket.toFixed(2)}`}
          </span>
        ) : null}
      </div>
      <span className="text-muted-foreground text-xs">
        {REFUSAL_NOTE[entry.refusal]}
      </span>
    </li>
  );
}

/**
 * Where the two cameras stood, which decides most of the panel below it.
 *
 * At the top rather than in a footnote because it is the single fact that
 * refuses every projected comparison when it goes wrong, and a reader looking at
 * thirty refusals needs to see the one sentence that explains them before
 * reading the thirty.
 */
function CameraNote({ camera }: { camera: CameraAgreement }) {
  const spans =
    camera.reference_span == null || camera.target_span == null
      ? null
      : `${camera.reference_span.toFixed(2)} against ${camera.target_span.toFixed(2)} torso lengths`;

  return (
    <div
      className={`rounded-md border p-3 text-sm ${camera.consistent ? "" : "border-status-degraded/40"}`}
      data-testid="camera-note"
    >
      <p className={camera.consistent ? "" : "text-status-degraded"}>
        {camera.consistent
          ? "The two cameras are close enough together for a projected measurement to mean the same thing in both."
          : "These two recordings do not agree about where the camera stood, so every projected, image-plane and foreshortened comparison below is refused."}
      </p>
      <p className="text-muted-foreground mt-1 text-xs">
        Measured from the shoulder span at address
        {spans ? `: ${spans}` : ", which at least one clip did not supply"}
        {camera.azimuth_separation_deg == null
          ? ""
          : `, putting the cameras up to ${camera.azimuth_separation_deg.toFixed(1)}° apart round the player`}
        . It assumes both clips show the same body — a broader player projects a
        broader line from the same place, and nothing here tells those apart.
      </p>
    </div>
  );
}

export function DifferencesPanel({ result }: { result: SwingComparison }) {
  const differences = result.differences ?? [];
  const refused = result.refused ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Differences</CardTitle>
        <CardDescription>
          {differences.length === 0
            ? `Nothing measured in both clips differs by more than these recordings can resolve. ${String(refused.length)} quantities were examined and refused.`
            : `${String(differences.length)} of ${String(result.metrics_considered)} quantities differ by more than these recordings can resolve.`}{" "}
          There is no score: a single number ranking two swings would need a
          scale relating degrees of turn to seconds of tempo, and nobody has
          measured one.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <CameraNote camera={result.camera} />

        {differences.length > 0 ? (
          <ul className="space-y-3">
            {differences.map((entry) => (
              <DifferenceRow
                key={`${entry.name}-${entry.event ?? "clip"}`}
                entry={entry}
              />
            ))}
          </ul>
        ) : null}

        {refused.length > 0 ? (
          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <CircleSlash
                className="text-muted-foreground size-4"
                aria-hidden="true"
              />
              Not compared ({refused.length})
            </h3>
            <ul className="space-y-3">
              {refused.map((entry) => (
                <RefusalRow
                  key={`${entry.name}-${entry.event ?? "clip"}`}
                  entry={entry}
                />
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
