/**
 * Swing phase detection: run it, then step through the clip frame by frame.
 *
 * Mounted with the video path as its `key` for the same reason `PosePanel` is:
 * a result can then never be shown next to the wrong file name.
 *
 * The inspector is the point of this panel rather than the summary above it. A
 * detector that reports "impact: frame 48" is impossible to agree or disagree
 * with from that sentence alone — the only way to judge it is to put the frame
 * number next to the phase it fell in and step across the boundary to see
 * whether it moved where it should have. So the timeline is scrubbable, the
 * events are jump targets, and the frame's phase is always on screen.
 *
 * **A clip with no swing in it is a result, not a failure.** It renders as a
 * finding with the engine's explanation attached, in the same panel a detected
 * swing would use, because "nobody swung" and "the worker crashed" are
 * different things and only one of them is a problem with the app.
 */

import type { EngineError, ProgressUpdate, SwingPhases } from "@gsa/types";
import {
  ChevronLeft,
  ChevronRight,
  Loader2,
  Play,
  Waypoints,
} from "lucide-react";
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
import { detectPhases, onProgress, progressFraction } from "@/lib/ipc";

/** Engine method name, used to filter progress off the shared channel. */
const TASK = "filter_poses";

type State =
  | { phase: "idle" }
  | { phase: "running"; update: ProgressUpdate | null }
  | { phase: "done"; result: SwingPhases }
  | { phase: "failed"; error: EngineError };

/**
 * Phase colours, as an exhaustive record.
 *
 * Exhaustive so that adding a phase to the Python contract is a TypeScript
 * error here until it is given a presentation, rather than rendering as an
 * unlabelled grey block nobody notices.
 */
const PHASE_STYLE: Record<
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

const EVENT_LABEL: Record<string, string> = {
  takeaway: "Takeaway",
  top: "Top",
  impact: "Impact",
  finish: "Finish",
};

function confidenceTone(value: number): string {
  if (value >= 0.6) return "text-status-ok";
  if (value > 0) return "text-status-degraded";
  return "text-destructive";
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 text-sm">
      <dt className="text-muted-foreground shrink-0">{label}</dt>
      <dd className="truncate font-mono" title={value}>
        {value}
      </dd>
    </div>
  );
}

function ProgressBar({ update }: { update: ProgressUpdate | null }) {
  const fraction = update ? progressFraction(update) : null;
  const percent = fraction === null ? null : Math.round(fraction * 100);

  return (
    <div className="space-y-2">
      <div className="text-muted-foreground flex justify-between text-sm">
        <span>Filtering landmarks</span>
        <span className="font-mono">
          {percent === null ? "—" : `${String(percent)}%`}
        </span>
      </div>
      <div
        className="bg-muted h-2 w-full overflow-hidden rounded-full"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <div
          className="bg-primary h-full transition-[width] duration-200"
          style={{ width: `${String(percent ?? 0)}%` }}
        />
      </div>
    </div>
  );
}

/**
 * The phase timeline, with event ticks.
 *
 * Widths come from frame counts rather than from durations, because the
 * scrubber underneath is indexed by frame and the two must line up. On a
 * variable-rate clip those differ, and a timeline that disagreed with its own
 * scrubber would be worse than no timeline.
 */
function Timeline({
  result,
  frame,
  onSeek,
}: {
  result: SwingPhases;
  frame: number;
  onSeek: (frame: number) => void;
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

        {/* Where the scrubber currently is. */}
        <div
          className="bg-primary pointer-events-none absolute -top-1 h-9 w-0.5"
          style={{ left: `${String((frame / total) * 100)}%` }}
          aria-hidden="true"
        />
      </div>

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
    </div>
  );
}

/**
 * Frame-by-frame inspector.
 *
 * The phase of the current frame comes from the reported intervals rather than
 * being recomputed here, so what the UI shows and what the engine concluded
 * cannot drift apart.
 */
function Inspector({ result }: { result: SwingPhases }) {
  const [frame, setFrame] = useState(() => {
    const impact = (result.events ?? []).find(
      (entry) => entry.event === "impact",
    );
    return impact?.frame_index ?? 0;
  });

  const last = Math.max(result.frames - 1, 0);
  const clamp = useCallback(
    (value: number) => Math.min(Math.max(value, 0), last),
    [last],
  );

  const interval = (result.phases ?? []).find(
    (entry) => frame >= entry.start_frame && frame < entry.end_frame,
  );
  const style = interval ? PHASE_STYLE[interval.phase] : null;
  const atFrame = (result.events ?? []).find(
    (entry) => entry.frame_index === frame,
  );

  return (
    <div className="space-y-4">
      <Timeline result={result} frame={frame} onSeek={setFrame} />

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="icon"
          aria-label="Previous frame"
          disabled={frame === 0}
          onClick={() => {
            setFrame(clamp(frame - 1));
          }}
        >
          <ChevronLeft />
        </Button>
        <input
          type="range"
          min={0}
          max={last}
          value={frame}
          aria-label="Frame"
          className="min-w-40 flex-1"
          onChange={(event) => {
            setFrame(clamp(Number(event.target.value)));
          }}
        />
        <Button
          variant="outline"
          size="icon"
          aria-label="Next frame"
          disabled={frame === last}
          onClick={() => {
            setFrame(clamp(frame + 1));
          }}
        >
          <ChevronRight />
        </Button>
      </div>

      {/* A live region: scrubbing changes it, and a screen reader should say
          so rather than leaving the phase readable only by hunting for it. */}
      <div
        role="status"
        aria-live="polite"
        className="bg-muted/40 flex flex-wrap items-baseline gap-x-6 gap-y-2 rounded-md p-3"
      >
        <span className="font-mono text-sm">
          frame <span className="font-semibold">{frame}</span> of {last}
        </span>
        <span className={`text-sm font-medium ${style?.text ?? ""}`}>
          {style ? style.label : "outside the detected swing"}
        </span>
        {atFrame ? (
          <span className="text-sm">
            {EVENT_LABEL[atFrame.event] ?? atFrame.event} —{" "}
            <span className={confidenceTone(atFrame.confidence.overall)}>
              confidence {atFrame.confidence.overall.toFixed(2)}
            </span>
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap gap-2">
        {(result.events ?? []).map((entry) => (
          <Button
            key={entry.event}
            variant="outline"
            size="sm"
            onClick={() => {
              setFrame(clamp(entry.frame_index));
            }}
          >
            {EVENT_LABEL[entry.event] ?? entry.event}
          </Button>
        ))}
      </div>
    </div>
  );
}

function EventTable({ result }: { result: SwingPhases }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-muted-foreground text-left">
          <tr>
            <th className="py-1 pr-4 font-medium">Event</th>
            <th className="py-1 pr-4 font-medium">Frame</th>
            <th className="py-1 pr-4 font-medium">Time</th>
            <th className="py-1 pr-4 font-medium">Confidence</th>
            <th className="py-1 pr-4 font-medium">Margin</th>
            <th className="py-1 pr-4 font-medium">Visibility</th>
            <th className="py-1 font-medium">Resolution</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {(result.events ?? []).map((entry) => (
            <tr key={entry.event} className="border-t">
              <td className="py-1 pr-4 font-sans">
                {EVENT_LABEL[entry.event] ?? entry.event}
              </td>
              <td className="py-1 pr-4">{entry.frame_index}</td>
              <td className="py-1 pr-4">{entry.timestamp_s.toFixed(3)}s</td>
              <td
                className={`py-1 pr-4 font-semibold ${confidenceTone(entry.confidence.overall)}`}
              >
                {entry.confidence.overall.toFixed(2)}
              </td>
              <td className="py-1 pr-4">
                {entry.confidence.margin.toFixed(2)}
              </td>
              <td className="py-1 pr-4">
                {entry.confidence.visibility.toFixed(2)}
              </td>
              <td className="py-1">{entry.confidence.resolution.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-muted-foreground mt-3 text-xs">
        Confidence is margin × visibility × resolution. Impact is estimated from
        hand motion — nothing here sees the ball or the club.
      </p>
    </div>
  );
}

export function PhasesPanel({ videoPath }: { videoPath: string }) {
  const [state, setState] = useState<State>({ phase: "idle" });

  const unlisten = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  const run = useCallback(async () => {
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
      const result = await detectPhases(videoPath);
      setState(
        result.ok
          ? { phase: "done", result: result.value }
          : { phase: "failed", error: result.error },
      );
    } catch (cause) {
      setState({
        phase: "failed",
        error: {
          kind: "transport",
          message: cause instanceof Error ? cause.message : String(cause),
        },
      });
    } finally {
      unlisten.current?.();
      unlisten.current = null;
    }
  }, [videoPath]);

  const result = state.phase === "done" ? state.result : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Swing phases</span>
          <Button
            size="sm"
            onClick={() => void run()}
            disabled={state.phase === "running"}
          >
            {state.phase === "running" ? (
              <Loader2 className="animate-spin" />
            ) : (
              <Play />
            )}
            Detect swing
          </Button>
        </CardTitle>
        <CardDescription>
          Locates the takeaway, top, impact and finish from the extracted
          landmarks. Extract poses first.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        {state.phase === "idle" ? (
          <p className="text-muted-foreground flex items-center gap-3 py-4 text-sm">
            <Waypoints className="size-5 opacity-40" aria-hidden="true" />
            Nothing detected yet.
          </p>
        ) : null}

        {state.phase === "running" ? (
          <ProgressBar update={state.update} />
        ) : null}

        {state.phase === "failed" ? (
          <EngineErrorPanel error={state.error} onRetry={() => void run()} />
        ) : null}

        {result ? (
          <>
            <div
              className={`text-sm font-medium ${result.detected ? "text-status-ok" : "text-status-degraded"}`}
            >
              {result.detected
                ? "Swing detected."
                : "No swing detected in this clip."}
            </div>

            <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
              <Field label="Frames" value={String(result.frames)} />
              <Field
                label="Hand tracked from"
                value={`${result.hand.source} (${String(result.hand.valid_frames)}/${String(result.hand.total_frames)})`}
              />
              <Field
                label="Hand travel"
                value={`${result.hand.travel_ratio.toFixed(2)} torso lengths`}
              />
              <Field
                label="Peak hand speed"
                value={`${result.hand.peak_speed.toFixed(2)} widths/s`}
              />
            </dl>

            {result.detected ? (
              <>
                <Inspector result={result} />
                <EventTable result={result} />
              </>
            ) : null}

            {result.warnings?.length ? (
              <ul className="text-status-degraded space-y-2 text-sm">
                {result.warnings.map((warning) => (
                  <li key={warning} className="flex gap-2">
                    <span aria-hidden="true">&middot;</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
