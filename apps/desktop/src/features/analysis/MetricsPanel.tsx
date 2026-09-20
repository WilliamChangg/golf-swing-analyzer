/**
 * The measurements, with what each one is a measurement *of*.
 *
 * Every row carries four things the number alone does not: the `basis`, which
 * says what kind of claim it is; the confidence, decomposed rather than rolled
 * into one figure; the uncertainty where one was measured; and the methodology
 * in words. None of those is an optional extra here — `Metric.basis` is the
 * field Phase 5 said does the most work, because a projected angle and a
 * three-dimensional one are different quantities that print the same way.
 *
 * **The refusals get equal billing, in their own list.** On a down-the-line clip
 * Phase 5 refuses shoulder turn, pelvis turn and X-factor outright: that camera
 * position does not contain the measurement. A panel that showed the 33 metrics
 * that survived and dropped the 3 that did not would be answering a question
 * nobody asked, and the reader would have no way to discover that the number
 * they came for was deliberately absent rather than merely missing.
 *
 * Clicking a metric seeks the player to the frames it was measured over, which
 * is what makes any of this checkable. The frames are the metric's own
 * `source_frames`; nothing here recomputes them.
 */

import type { Metric, MetricSet, RefusedMetric } from "@gsa/types";
import { Info } from "lucide-react";
import { useState } from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/** How a metric's unit is written. Exhaustive: a new unit is a type error here. */
const UNIT_SUFFIX: Record<Metric["unit"], string> = {
  degrees: "°",
  seconds: " s",
  ratio: " : 1",
  torso_lengths: " torso",
  torso_lengths_per_s: " torso/s",
  metres_per_s: " m/s",
};

/**
 * What a `basis` means, in one line.
 *
 * Shown next to every value rather than in a legend, because the distinction it
 * draws is the one a reader is most likely to assume away: an angle inferred
 * from how much a line shortened in one photograph is not the angle a
 * three-dimensional capture would report, and both render as a number followed
 * by a degree sign.
 */
const BASIS_NOTE: Record<Metric["basis"], string> = {
  temporal: "a duration, measured on the clip's own clock",
  image_plane: "a distance in the picture, in units that survive the framing",
  projected_angle: "an angle in the picture, not an angle of the body",
  foreshortened_angle:
    "a rotation inferred from how much a line shortened — one camera cannot see it directly",
  spatial: "measured in three dimensions from two calibrated, aligned views",
};

function confidenceTone(value: number): string {
  if (value >= 0.6) return "text-status-ok";
  if (value > 0) return "text-status-degraded";
  return "text-destructive";
}

function formatValue(metric: Metric): string {
  const digits = metric.unit === "seconds" || metric.unit === "ratio" ? 2 : 1;
  const base = `${metric.value.toFixed(digits)}${UNIT_SUFFIX[metric.unit]}`;
  // Only where one was measured. Phase 6 quantifies uncertainty for the
  // foreshortened rotations and for nothing else, and printing "± 0" on the
  // rest would claim a precision that was never computed.
  return metric.uncertainty == null
    ? base
    : `${base} ± ${metric.uncertainty.toFixed(1)}${UNIT_SUFFIX[metric.unit]}`;
}

function MetricRow({
  metric,
  onSeek,
}: {
  metric: Metric;
  onSeek: (frame: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const frames = metric.source_frames;

  return (
    <>
      <tr className="border-t">
        <td className="py-1.5 pr-3">
          <button
            type="button"
            className="text-left hover:underline"
            // Seeks to the first frame the metric was measured over. A metric
            // measured across the address phase cites 161 consecutive frames,
            // so the first is the only defensible single target; the count is
            // shown beside it so the reader knows it is a span.
            onClick={() => {
              if (frames.length > 0) onSeek(frames[0] ?? 0);
            }}
            disabled={frames.length === 0}
            title={
              frames.length > 0
                ? `Jump to frame ${String(frames[0])}`
                : "No frames cited"
            }
          >
            {metric.label}
          </button>
        </td>
        <td className="py-1.5 pr-3 text-right font-mono">
          {formatValue(metric)}
        </td>
        <td
          className={`py-1.5 pr-3 text-right font-mono ${confidenceTone(metric.confidence.overall)}`}
        >
          {metric.confidence.overall.toFixed(2)}
        </td>
        <td className="py-1.5 pr-1 text-right">
          <button
            type="button"
            aria-label={`How ${metric.label} was measured`}
            aria-expanded={open}
            className="text-muted-foreground hover:text-foreground"
            onClick={() => {
              setOpen(!open);
            }}
          >
            <Info className="size-4" />
          </button>
        </td>
      </tr>
      {open ? (
        <tr className="bg-muted/30">
          <td colSpan={4} className="space-y-2 px-3 py-3 text-xs">
            <p>
              <span className="font-medium">{metric.basis}</span> —{" "}
              {BASIS_NOTE[metric.basis]}
            </p>
            <p className="text-muted-foreground">{metric.methodology}</p>
            <p className="text-muted-foreground">{metric.interpretation}</p>
            <p className="font-mono">
              observation {metric.confidence.observation.toFixed(2)} &times;
              anchor {metric.confidence.anchor.toFixed(2)} &times; method{" "}
              {metric.confidence.method.toFixed(2)}
            </p>
            <p className="text-muted-foreground font-mono">
              {frames.length === 0
                ? "no frames cited"
                : frames.length === 1
                  ? `frame ${String(frames[0])}`
                  : `frames ${String(frames[0])}–${String(frames[frames.length - 1])} (${String(frames.length)})`}
            </p>
          </td>
        </tr>
      ) : null}
    </>
  );
}

function Refusals({ refused }: { refused: RefusedMetric[] }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">Refused ({refused.length})</h4>
      <p className="text-muted-foreground text-xs">
        Measurements this recording does not contain. Listed by name, because a
        number that is absent on purpose and one that is absent by accident look
        identical from the outside.
      </p>
      <ul className="space-y-1.5 text-xs">
        {refused.map((entry) => (
          <li
            key={`${entry.name}-${entry.event ?? "none"}`}
            className="flex gap-2"
          >
            <span className="font-mono shrink-0">{entry.name}</span>
            <span className="text-muted-foreground">{entry.reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function MetricsPanel({
  result,
  onSeek,
}: {
  result: MetricSet;
  onSeek: (frame: number) => void;
}) {
  // Defaulted once rather than guarded at each use: these carry
  // `default_factory=list` on the Python side and so arrive optional.
  const metrics = result.metrics ?? [];
  const refused = result.refused ?? [];
  const warnings = result.warnings ?? [];

  const groups = new Map<string, Metric[]>();
  for (const metric of metrics) {
    const bucket = groups.get(metric.group) ?? [];
    bucket.push(metric);
    groups.set(metric.group, bucket);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Measurements</span>
          <span className="text-muted-foreground text-sm font-normal">
            {metrics.length} measured &middot; {refused.length} refused
          </span>
        </CardTitle>
        <CardDescription>
          {result.view
            ? `Filmed ${result.view.view.replace(/_/g, " ")} (confidence ${result.view.confidence.toFixed(2)}). `
            : ""}
          Every value is clickable and jumps to the frames it was measured over.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {warnings.length > 0 ? (
          <ul className="text-status-degraded space-y-2 text-sm">
            {warnings.map((warning) => (
              <li key={warning} className="flex gap-2">
                <span aria-hidden="true">&middot;</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        ) : null}

        {[...groups.entries()].map(([group, metrics]) => (
          <div key={group} className="space-y-1">
            <h4 className="text-sm font-medium capitalize">
              {group.replace(/_/g, " ")}
            </h4>
            <table className="w-full text-sm">
              <caption className="sr-only">{group} measurements</caption>
              <thead className="text-muted-foreground text-left text-xs">
                <tr>
                  <th className="py-1 pr-3 font-medium">Metric</th>
                  <th className="py-1 pr-3 text-right font-medium">Value</th>
                  <th className="py-1 pr-3 text-right font-medium">Conf.</th>
                  <th className="py-1 pr-1 text-right font-medium">
                    <span className="sr-only">Methodology</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric) => (
                  <MetricRow
                    key={`${metric.name}-${metric.event ?? metric.phase ?? "clip"}`}
                    metric={metric}
                    onSeek={onSeek}
                  />
                ))}
              </tbody>
            </table>
          </div>
        ))}

        {refused.length > 0 ? <Refusals refused={refused} /> : null}
      </CardContent>
    </Card>
  );
}
