/**
 * How a swing phase is presented, and which one a frame falls in.
 *
 * Kept out of `PhaseTimeline.tsx` so that file exports only components. The
 * split is a build concern rather than a design one — a module mixing
 * components with constants defeats fast refresh — but it lands in a useful
 * place anyway: the colours and the lookup are shared by the timeline, the
 * inspector and the workflow screen, and none of them should own them.
 */

import type { SwingPhases } from "@gsa/types";

/**
 * Phase colours, as an exhaustive record.
 *
 * Exhaustive so that adding a phase to the Python contract is a TypeScript
 * error here until it is given a presentation, rather than rendering as an
 * unlabelled grey block nobody notices.
 */
export const PHASE_STYLE: Record<
  SwingPhases["phases"] extends (infer P)[] | undefined
    ? P extends { phase: infer K }
      ? K
      : never
    : never,
  { bar: string; text: string; label: string }
> = {
  address: { bar: "bg-slate-300", text: "text-slate-600", label: "Address" },
  backswing: { bar: "bg-sky-400", text: "text-sky-700", label: "Backswing" },
  downswing: {
    bar: "bg-orange-400",
    text: "text-orange-700",
    label: "Downswing",
  },
  follow_through: {
    bar: "bg-emerald-400",
    text: "text-emerald-700",
    label: "Follow-through",
  },
};

/**
 * The phase a frame falls in, from the engine's own reported intervals.
 *
 * Looked up rather than recomputed, so what the UI shows and what the engine
 * concluded cannot drift apart. Returns undefined for a frame outside the
 * detected swing, which is a real state: a clip usually starts before the
 * address and ends after the finish.
 */
export function phaseAt(result: SwingPhases, frame: number) {
  return (result.phases ?? []).find(
    (entry) => frame >= entry.start_frame && frame < entry.end_frame,
  );
}
