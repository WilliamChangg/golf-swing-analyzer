/**
 * The swing, drawn as four coloured spans with the events marked.
 *
 * Extracted from `PhasesPanel` when Phase 14 needed the same timeline under the
 * video. Two copies would have been two opinions about which frames the
 * backswing covers, and they would have looked identical right up until one of
 * them was fixed.
 *
 * Widths come from **frame counts**, not durations, because the scrubber beside
 * it is indexed by frame and the two must line up. On a variable-rate clip those
 * differ, and a timeline that disagreed with its own scrubber would be worse
 * than no timeline.
 */

import type { SwingPhases } from "@gsa/types";

import { PHASE_STYLE } from "@/features/player/phases";

export function PhaseTimeline({
  result,
  frame,
  onSeek,
  showLegend = true,
}: {
  result: SwingPhases;
  frame: number;
  onSeek: (frame: number) => void;
  showLegend?: boolean;
}) {
  const total = Math.max(result.frames, 1);
  const phases = result.phases ?? [];
  const events = result.events ?? [];

  return (
    <div className="space-y-2">
      <div className="relative">
        <div className="flex h-7 w-full overflow-hidden rounded">
          {phases.map((interval) => {
            const style = PHASE_STYLE[interval.phase];
            const width =
              ((interval.end_frame - interval.start_frame) / total) * 100;
            return (
              <button
                key={interval.phase}
                type="button"
                title={`${style.label} — ${interval.duration_s.toFixed(3)} s, confidence ${interval.confidence.toFixed(2)}`}
                aria-label={`Jump to ${style.label}`}
                onClick={() => {
                  onSeek(interval.start_frame);
                }}
                className={`${style.bar} h-full cursor-pointer`}
                style={{ width: `${String(width)}%` }}
              />
            );
          })}
        </div>

        {/* Event ticks, drawn over the phases they divide. */}
        {events.map((entry) => (
          <div
            key={entry.event}
            className="bg-foreground pointer-events-none absolute top-0 h-7 w-px"
            style={{ left: `${String((entry.frame_index / total) * 100)}%` }}
            aria-hidden="true"
          />
        ))}

        {/* Where the playhead is. */}
        <div
          className="bg-primary pointer-events-none absolute -top-1 h-9 w-0.5"
          style={{ left: `${String((frame / total) * 100)}%` }}
          aria-hidden="true"
        />
      </div>

      {showLegend ? (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {phases.map((interval) => {
            const style = PHASE_STYLE[interval.phase];
            return (
              <span key={interval.phase} className="flex items-center gap-1.5">
                <span
                  className={`${style.bar} inline-block size-2 rounded-sm`}
                  aria-hidden="true"
                />
                <span className="text-muted-foreground">
                  {style.label} {interval.duration_s.toFixed(2)}s
                </span>
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
