/**
 * Turning two sampled series into SVG, and the two rules that decide how.
 *
 * Kept out of `TrajectoryPlot.tsx` so that file exports only components — the
 * same split `features/player/phases.ts` makes, and for the same build reason.
 * It lands somewhere useful anyway: these are the two functions with an argument
 * behind them, and they are testable without rendering anything.
 *
 * **Null is not zero.** A position where a clip supports no value is a break in
 * the path, never a point on the baseline. `SwingComparison` carries null exactly
 * where nothing is supported — the filter blocked it, or the clip's tracked range
 * did not reach that far — and a line drawn straight across it would put a curve
 * on screen at a position where that clip was never measured, with nothing for a
 * reader to see. So each run of consecutive non-null samples becomes its own
 * `M`-started subpath.
 *
 * **The band is what the two lines cannot say.** `bandOf` draws the reference
 * curve's own bracket as an area, which is how far the curve can move on one
 * frame of ambiguity at each event. Two lines alone invite a reader to attribute
 * every gap to the two swings, and near the ends of a clip most of it is the
 * clock instead.
 */

import type { ChannelSample } from "@gsa/types";

export interface Scale {
  x: (position: number) => number;
  y: (value: number) => number;
  /** The value range the vertical axis covers, so the plot can label it. */
  domain: [low: number, high: number];
}

export interface PlotBox {
  width: number;
  height: number;
  pad: { top: number; right: number; bottom: number; left: number };
}

/**
 * A scale covering both series **and the band**, or null when nothing is drawable.
 *
 * The band's extent is included deliberately: a bracket drawn past the top of
 * the plot is a bracket the reader silently under-reads, which inverts the
 * point of drawing one.
 */
export function scaleFor(samples: ChannelSample[], box: PlotBox): Scale | null {
  const values: number[] = [];
  for (const sample of samples) {
    const spread = sample.bracket ?? 0;
    if (sample.reference != null) {
      values.push(sample.reference - spread, sample.reference + spread);
    }
    if (sample.target != null) values.push(sample.target);
  }
  if (values.length === 0) return null;

  let low = Math.min(...values);
  let high = Math.max(...values);
  if (high - low < 1e-9) {
    // A flat series still deserves an axis. Without this the scale divides by
    // zero and every point lands on one row.
    low -= 0.5;
    high += 0.5;
  }

  const inner = {
    width: box.width - box.pad.left - box.pad.right,
    height: box.height - box.pad.top - box.pad.bottom,
  };
  return {
    // The axis is always the full swing, takeaway to finish, whether or not
    // this channel has values across all of it. A scale fitted to the supported
    // range would put two clips' gaps in different places.
    x: (position) => box.pad.left + (position / 3) * inner.width,
    y: (value) =>
      box.pad.top + (1 - (value - low) / (high - low)) * inner.height,
    domain: [low, high],
  };
}

/** One series as SVG, broken wherever the series is. */
export function pathOf(
  samples: ChannelSample[],
  pick: (sample: ChannelSample) => number | null | undefined,
  scale: Scale,
): string {
  const parts: string[] = [];
  let open = false;
  for (const sample of samples) {
    const value = pick(sample);
    if (value == null) {
      open = false;
      continue;
    }
    const point = `${scale.x(sample.position).toFixed(1)} ${scale.y(value).toFixed(1)}`;
    parts.push(`${open ? "L" : "M"}${point}`);
    open = true;
  }
  return parts.join(" ");
}

/**
 * The reference curve's bracket, as one closed area per unbroken run.
 *
 * Per run for the reason the lines are: an area spanning a gap would shade a
 * region neither clip measured. A run of one sample is dropped rather than drawn
 * — two points cannot enclose anything, and a zero-width sliver reads as a
 * rendering fault.
 */
export function bandOf(samples: ChannelSample[], scale: Scale): string {
  const runs: ChannelSample[][] = [];
  let current: ChannelSample[] = [];
  for (const sample of samples) {
    if (sample.reference == null || sample.bracket == null) {
      if (current.length > 1) runs.push(current);
      current = [];
      continue;
    }
    current.push(sample);
  }
  if (current.length > 1) runs.push(current);

  return runs
    .map((run) => {
      const edge = (sample: ChannelSample, sign: 1 | -1) =>
        `${scale.x(sample.position).toFixed(1)} ${scale
          .y((sample.reference ?? 0) + sign * (sample.bracket ?? 0))
          .toFixed(1)}`;
      const upper = run.map((sample) => edge(sample, 1));
      const lower = [...run].reverse().map((sample) => edge(sample, -1));
      return `M${upper.join(" L")} L${lower.join(" L")} Z`;
    })
    .join(" ");
}
