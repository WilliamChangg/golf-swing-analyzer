/**
 * Two swings' curves on one axis, with the part of the gap nobody can attribute.
 *
 * The drawing decision this component exists for is the **band**, not the two
 * lines. Two lines on one axis is what every swing-comparison product draws, and
 * it invites the reader to attribute every gap between them to the two swings.
 * Most of the gap near the ends of a clip is not that: the four events are
 * located to a frame, so the position of any real instant on the axis is known
 * only as well as the events bounding it, and the same curve read a frame early
 * is a different curve.
 *
 * So the reference curve is drawn with that ambiguity around it, as a filled
 * band, and the target line is drawn on top. Where the target stays inside the
 * band, the two recordings cannot tell the swings apart — and the engine has
 * already said so per sample, so the band is a rendering of `sample.bracket`
 * rather than a second opinion about it.
 *
 * **Null is not zero.** A position where either clip supports no value is a break
 * in the path, not a point on the baseline. `SwingComparison` carries null
 * exactly where nothing is supported, and `pathOf` turns each run of non-null
 * samples into its own `M …` subpath so a gap stays a gap.
 *
 * No chart library. The axis has four labelled positions and the series are 121
 * points; what a library would add here is a dependency and a second opinion
 * about scales, and what it would not add is the band, which is the whole point.
 */

import type { TrajectoryComparison } from "@gsa/types";

import { bandOf, pathOf, scaleFor } from "@/features/compare/paths";

const WIDTH = 720;
const HEIGHT = 190;
const BOX = {
  width: WIDTH,
  height: HEIGHT,
  pad: { top: 12, right: 12, bottom: 24, left: 46 },
};

/**
 * The four instants the normalisation pins, which are the axis' only landmarks.
 *
 * `prose` is not the axis label with an article stuck on: "impact" takes none in
 * English and the other three do, and a caption reading "at the impact" is the
 * kind of thing that makes a reader wonder what else was assembled by template.
 */
const KNOTS: { at: number; label: string; prose: string }[] = [
  { at: 0, label: "takeaway", prose: "the takeaway" },
  { at: 1, label: "top", prose: "the top" },
  { at: 2, label: "impact", prose: "impact" },
  { at: 3, label: "finish", prose: "the finish" },
];

/** Enough digits to distinguish two samples, and no more. */
function format(value: number): string {
  return Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2);
}

/**
 * Where on the axis a position sits, in the reader's words.
 *
 * "at the top" rather than "at 1.00", because the axis' only landmarks are the
 * four events and a reader who has to convert a decimal back into one of them is
 * being handed the engine's coordinates rather than the swing's.
 */
function near(position: number): string {
  const nearest = KNOTS.reduce((best, knot) =>
    Math.abs(knot.at - position) < Math.abs(best.at - position) ? knot : best,
  );
  if (Math.abs(nearest.at - position) < 0.12) return `at ${nearest.prose}`;
  return position < nearest.at
    ? `just before ${nearest.prose}`
    : `just after ${nearest.prose}`;
}

export function TrajectoryPlot({
  channel,
  referenceLabel,
  targetLabel,
}: {
  channel: TrajectoryComparison;
  referenceLabel: string;
  targetLabel: string;
}) {
  const samples = channel.samples ?? [];
  const scale = scaleFor(samples, BOX);

  if (scale === null) {
    return (
      <p className="text-muted-foreground text-sm">
        Neither clip supports a value for {channel.label.toLowerCase()} anywhere
        on the swing, so there is nothing to draw.
      </p>
    );
  }

  const resolved = samples.filter((sample) => sample.resolved);
  const [low, high] = scale.domain;

  return (
    <figure className="space-y-2" data-testid={`plot-${channel.channel}`}>
      <svg
        viewBox={`0 0 ${String(WIDTH)} ${String(HEIGHT)}`}
        className="w-full"
        role="img"
        aria-label={`${channel.label}: ${referenceLabel} against ${targetLabel}`}
      >
        {KNOTS.map((knot) => (
          <g key={knot.label}>
            <line
              x1={scale.x(knot.at)}
              x2={scale.x(knot.at)}
              y1={BOX.pad.top}
              y2={HEIGHT - BOX.pad.bottom}
              className="stroke-border"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <text
              x={scale.x(knot.at)}
              y={HEIGHT - 8}
              textAnchor={
                knot.at === 0 ? "start" : knot.at === 3 ? "end" : "middle"
              }
              className="fill-muted-foreground text-[10px]"
            >
              {knot.label}
            </text>
          </g>
        ))}

        {/* The vertical axis, as two labels rather than a ruler. A comparison
            plot without one shows a reader the shape of a difference and not its
            size, which is most of what they came for — and a full set of ticks
            on a 190px strip is more furniture than information. */}
        {[high, low].map((value) => (
          <text
            key={value}
            x={BOX.pad.left - 6}
            y={scale.y(value) + 3}
            textAnchor="end"
            className="fill-muted-foreground font-mono text-[10px]"
          >
            {format(value)}
          </text>
        ))}

        {/* The band first, so both lines sit on top of it. */}
        <path
          d={bandOf(samples, scale)}
          className="fill-sky-500/15"
          data-testid="bracket-band"
        />

        {/* Where the difference clears the band, a tick on the floor of the plot.
            Drawn rather than left to the reader to spot, because a curve leaving
            a band by a hair looks the same as one leaving it by a mile at this
            scale, and only one of them is a difference. */}
        {resolved.map((sample) => (
          <line
            key={sample.position}
            x1={scale.x(sample.position)}
            x2={scale.x(sample.position)}
            y1={HEIGHT - BOX.pad.bottom - 5}
            y2={HEIGHT - BOX.pad.bottom}
            className="stroke-amber-500"
            strokeWidth={2}
            data-testid="resolved-tick"
          />
        ))}

        <path
          d={pathOf(samples, (sample) => sample.reference, scale)}
          className="stroke-sky-600"
          strokeWidth={1.6}
          fill="none"
          data-testid="reference-path"
        />
        <path
          d={pathOf(samples, (sample) => sample.target, scale)}
          className="stroke-orange-500"
          strokeWidth={1.6}
          fill="none"
          strokeDasharray="5 3"
          data-testid="target-path"
        />
      </svg>

      <figcaption className="text-muted-foreground flex flex-wrap items-baseline gap-x-4 gap-y-1 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-sky-600" aria-hidden="true" />
          {referenceLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-0.5 w-4 bg-orange-500"
            style={{
              backgroundImage:
                "repeating-linear-gradient(90deg, currentColor 0 4px, transparent 4px 7px)",
            }}
            aria-hidden="true"
          />
          {targetLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-4 bg-sky-500/25" aria-hidden="true" />
          what one frame of ambiguity at each event can account for
        </span>
        <span>
          {channel.unit.replace(/_/g, " ")} ·{" "}
          <span className="font-mono">
            {(channel.resolved_fraction * 100).toFixed(0)}%
          </span>{" "}
          of the swing differs by more than that
          {channel.largest_difference == null || channel.largest_at == null ? (
            ""
          ) : (
            <>
              , the largest{" "}
              <span className="font-mono">
                {channel.largest_difference > 0 ? "+" : ""}
                {format(channel.largest_difference)}
              </span>{" "}
              {near(channel.largest_at)}
            </>
          )}
        </span>
      </figcaption>
    </figure>
  );
}
