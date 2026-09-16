/**
 * Pose extraction: start it, watch it, and show what it produced.
 *
 * Mounted with the video path as its `key`, so loading a different clip
 * remounts this and discards the previous clip's result. That is cheaper and
 * harder to get wrong than resetting state from an effect, and it means a
 * result can never be shown next to the wrong file name.
 *
 * The progress bar exists because extraction runs for minutes on a
 * high-frame-rate clip, and an indeterminate spinner cannot distinguish slow
 * work from a hung worker — which is the one thing a user needs to know when a
 * job is taking longer than they expected. Every figure shown is one the engine
 * measured.
 */

import type {
  EngineError,
  PoseExtractionResult,
  ProgressUpdate,
} from "@gsa/types";
import { Loader2, Play, ScanLine } from "lucide-react";
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
import { extractPoses, onProgress, progressFraction } from "@/lib/ipc";

/** Engine method name, used to filter progress off the shared channel. */
const TASK = "extract_poses";

type State =
  | { phase: "idle" }
  | { phase: "running"; update: ProgressUpdate | null }
  | { phase: "done"; result: PoseExtractionResult }
  | { phase: "failed"; error: EngineError };

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
        <span>
          {update?.stage === "starting"
            ? "Loading the model"
            : "Estimating pose"}
          {update && update.total != null
            ? ` — frame ${String(update.current)} of ${String(update.total)}`
            : ""}
        </span>
        {/* An unknown total renders as no number at all, rather than as a
            percentage derived from a denominator the engine did not supply. */}
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
        aria-label="Pose extraction progress"
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
      {update ? (
        <p className="text-muted-foreground font-mono text-xs">
          {update.elapsed_s.toFixed(1)}s elapsed
        </p>
      ) : null}
    </div>
  );
}

function ResultPanel({ result }: { result: PoseExtractionResult }) {
  const { stats, model } = result;

  return (
    <div className="space-y-4">
      <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
        <Field
          label="Frames processed"
          value={String(stats.frames_processed)}
        />
        <Field
          label="Frames with a pose"
          value={`${String(stats.frames_detected)} (${(stats.detection_rate * 100).toFixed(1)}%)`}
        />
        <Field
          label="Time per frame"
          value={`${stats.ms_per_frame.toFixed(1)} ms`}
        />
        <Field label="Total time" value={`${stats.elapsed_s.toFixed(1)} s`} />
        <Field label="Model" value={`${model.name} (${model.precision})`} />
        <Field label="Inference on" value={model.delegate} />
        <Field
          label="Mean visibility"
          value={
            stats.mean_visibility == null
              ? "not measured"
              : stats.mean_visibility.toFixed(3)
          }
        />
        <Field label="Model digest" value={model.sha256.slice(0, 16)} />
      </dl>
      <p className="text-muted-foreground font-mono text-xs break-all">
        {result.output_path}
      </p>
    </div>
  );
}

export function PosePanel({ videoPath }: { videoPath: string }) {
  const [state, setState] = useState<State>({ phase: "idle" });

  // Held in a ref so the effect that tears the subscription down does not have
  // to re-run every time an update arrives.
  const unlisten = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  const run = useCallback(async () => {
    setState({ phase: "running", update: null });

    // Subscribed before the call, not after: extraction starts reporting
    // immediately, and a listener registered afterwards would miss the first
    // updates — including the only one for a clip short enough to finish fast.
    unlisten.current = await onProgress(
      (update) => {
        setState((current) =>
          current.phase === "running" ? { phase: "running", update } : current,
        );
      },
      { task: TASK },
    );

    try {
      const result = await extractPoses(videoPath);
      setState(
        result.ok
          ? { phase: "done", result: result.value }
          : { phase: "failed", error: result.error },
      );
    } finally {
      unlisten.current?.();
      unlisten.current = null;
    }
  }, [videoPath]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Pose</span>
          <Button
            onClick={() => void run()}
            size="sm"
            variant={state.phase === "done" ? "outline" : "default"}
            disabled={state.phase === "running"}
          >
            {state.phase === "running" ? (
              <Loader2 className="animate-spin" />
            ) : (
              <Play />
            )}
            {state.phase === "done" ? "Extract again" : "Extract pose"}
          </Button>
        </CardTitle>
        <CardDescription>
          Runs the pose model over every frame and stores the landmarks.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {state.phase === "idle" ? (
          <div className="text-muted-foreground flex items-center gap-3 py-4 text-sm">
            <ScanLine className="size-4 opacity-40" aria-hidden="true" />
            No landmarks extracted for this clip yet.
          </div>
        ) : null}

        {state.phase === "running" ? (
          <ProgressBar update={state.update} />
        ) : null}

        {state.phase === "failed" ? (
          <EngineErrorPanel error={state.error} onRetry={() => void run()} />
        ) : null}

        {state.phase === "done" ? (
          <>
            {state.result.warnings?.length ? (
              <ul className="text-status-degraded space-y-2 text-sm">
                {state.result.warnings.map((warning: string) => (
                  <li key={warning} className="flex gap-2">
                    <span aria-hidden="true">&middot;</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            <ResultPanel result={state.result} />
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
