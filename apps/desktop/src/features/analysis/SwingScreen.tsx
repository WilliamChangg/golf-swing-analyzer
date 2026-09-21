/**
 * The workflow, end to end: a clip, a player, and everything measured from it.
 *
 * The arrangement is the argument. The video, the phase timeline, the
 * measurements and the findings are on one screen because each of the last
 * three is only checkable against the first — a metric is checked by looking at
 * the frame it came from, and a finding is checked by following its evidence to
 * the frames it cites. Splitting them across tabs would leave every number on
 * screen unaccompanied by the thing that could contradict it.
 *
 * **Two inputs are asked for rather than measured, and they are next to each
 * other for that reason.** The smoothing window is a choice about how much the
 * trajectory is allowed to bend, and the engine's default cannot be supported
 * by a clip below about 60 fps at all; the slow-motion factor is not recorded
 * anywhere in a conformed file, so it is supplied or it is wrong. Neither has a
 * value this app can discover, and neither is hidden.
 *
 * Analysis is deliberately not automatic on import. Pose extraction is seconds
 * to minutes, and a screen that started it on file-open would spend that time
 * before the user had confirmed they opened the clip they meant to.
 */

import type {
  CoachingReport,
  EngineError,
  MetricSet,
  PoseOverlay,
  ProgressUpdate,
  SeekIndex,
  SwingPhases,
  VideoMetadata,
} from "@gsa/types";
import { FileVideo, FolderOpen, Loader2, Play, Sparkles } from "lucide-react";
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
import { FindingsPanel } from "@/features/analysis/FindingsPanel";
import { MetricsPanel } from "@/features/analysis/MetricsPanel";
import { OverlayCanvas } from "@/features/player/OverlayCanvas";
import { PhaseTimeline } from "@/features/player/PhaseTimeline";
import { PHASE_STYLE, phaseAt } from "@/features/player/phases";
import { Transport } from "@/features/player/Transport";
import { useFramePlayer } from "@/features/player/useFramePlayer";
import {
  chooseClip,
  clipSource,
  coachSwing,
  computeMetrics,
  detectPhases,
  extractPoses,
  onProgress,
  poseOverlay,
  probeVideo,
  progressFraction,
  seekIndex as fetchSeekIndex,
  type AnalysisOptions,
} from "@/lib/ipc";

interface Clip {
  path: string;
  metadata: VideoMetadata;
  index: SeekIndex;
}

interface Analysis {
  phases: SwingPhases | null;
  metrics: MetricSet | null;
  coaching: CoachingReport | null;
  overlay: PoseOverlay | null;
}

const EMPTY: Analysis = {
  phases: null,
  metrics: null,
  coaching: null,
  overlay: null,
};

type Busy = { label: string; update: ProgressUpdate | null } | null;

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function ProgressBar({ busy }: { busy: NonNullable<Busy> }) {
  const fraction = busy.update ? progressFraction(busy.update) : null;
  const percent = fraction === null ? null : Math.round(fraction * 100);

  return (
    <div className="space-y-2">
      <div className="text-muted-foreground flex justify-between text-sm">
        <span>{busy.label}</span>
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
 * The two numbers nothing in the file can supply.
 *
 * Both sit above the analysis rather than behind a settings pane, because
 * getting either wrong does not produce an error — it produces a tempo and a
 * club-head speed that are simply wrong, which is the same failure mode Phase 1
 * built the whole ingestion layer around.
 */
function Inputs({
  options,
  onChange,
  disabled,
}: {
  options: AnalysisOptions;
  onChange: (next: AnalysisOptions) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-wrap items-end gap-4">
      <label className="space-y-1 text-sm">
        <span className="text-muted-foreground block">
          Smoothing window (s)
        </span>
        <input
          type="number"
          step="0.01"
          min="0.01"
          value={options.windowS ?? 0.1}
          disabled={disabled}
          className="border-input w-28 rounded-md border px-2 py-1 font-mono"
          onChange={(event) => {
            onChange({ ...options, windowS: Number(event.target.value) });
          }}
        />
      </label>
      <label className="space-y-1 text-sm">
        <span className="text-muted-foreground block">Slow-motion factor</span>
        <input
          type="number"
          step="0.5"
          min="0.1"
          value={options.slowMotionFactor ?? 1}
          disabled={disabled}
          className="border-input w-28 rounded-md border px-2 py-1 font-mono"
          onChange={(event) => {
            onChange({
              ...options,
              slowMotionFactor: Number(event.target.value),
            });
          }}
        />
      </label>
      <p className="text-muted-foreground max-w-md text-xs">
        A clip below about 60 fps cannot support the 0.10 s default and will
        measure nothing; the engine names the width its rate would support. The
        slow-motion factor is not recorded in a conformed file — it is supplied,
        and every duration and speed below is multiplied by it.
      </p>
    </div>
  );
}

export function SwingScreen() {
  const [clip, setClip] = useState<Clip | null>(null);
  const [analysis, setAnalysis] = useState<Analysis>(EMPTY);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<EngineError | null>(null);
  const [options, setOptions] = useState<AnalysisOptions>({});

  const video = useRef<HTMLVideoElement | null>(null);
  const player = useFramePlayer(video, clip?.index ?? null);
  const unlisten = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  const run = useCallback(
    async <T,>(
      label: string,
      task: () => Promise<
        { ok: true; value: T } | { ok: false; error: EngineError }
      >,
    ): Promise<T | null> => {
      setBusy({ label, update: null });
      setError(null);
      unlisten.current = await onProgress((update) => {
        setBusy((current) => (current ? { ...current, update } : current));
      });
      try {
        const result = await task();
        if (!result.ok) {
          setError(result.error);
          return null;
        }
        return result.value;
      } catch (cause) {
        setError({
          kind: "transport",
          message: cause instanceof Error ? cause.message : String(cause),
        });
        return null;
      } finally {
        unlisten.current?.();
        unlisten.current = null;
        setBusy(null);
      }
    },
    [],
  );

  const choose = useCallback(async () => {
    // The picker runs in Rust and is what admits the file for playback. There
    // is no path-taking counterpart on purpose — see `chooseClip`.
    const chosen = await run("Waiting for a file", () => chooseClip());
    if (chosen == null) return;

    const metadata = await run("Reading the container index", () =>
      probeVideo(chosen),
    );
    if (!metadata) return;

    const index = await run("Reading frame times", () =>
      fetchSeekIndex(chosen),
    );
    if (!index) return;

    setAnalysis(EMPTY);
    setClip({ path: chosen, metadata, index });
  }, [run]);

  const analyse = useCallback(async () => {
    if (!clip) return;
    setAnalysis(EMPTY);

    // Extraction first and separately: it is the only step measured in seconds
    // per clip, and it writes a Parquet the rest read. Running it every time
    // would make changing the smoothing window a minute-long operation, so it
    // is idempotent on the engine side through the content-keyed cache and
    // cheap to repeat here.
    const extracted = await run("Extracting poses", () =>
      extractPoses(clip.path),
    );
    if (!extracted) return;

    const phases = await run("Detecting the swing", () =>
      detectPhases(clip.path, {
        ...(options.model === undefined ? {} : { model: options.model }),
        ...(options.windowS === undefined ? {} : { windowS: options.windowS }),
      }),
    );
    if (!phases) return;
    const metrics = await run("Measuring", () =>
      computeMetrics(clip.path, options),
    );
    if (!metrics) return;
    const coaching = await run("Reaching conclusions", () =>
      coachSwing(clip.path, options),
    );
    if (!coaching) return;

    // The overlay is fetched for the whole clip where that fits inside the
    // engine's range limit, which covers every reference clip here. A longer
    // recording would need windowing; the engine refuses rather than truncating,
    // so that case surfaces as an error naming the limit rather than as a
    // skeleton that silently stops half-way.
    const overlay = await run("Drawing the skeleton", () =>
      poseOverlay(clip.path, { startFrame: 0 }, options),
    );

    setAnalysis({ phases, metrics, coaching, overlay });
    const impact = (phases?.events ?? []).find(
      (entry) => entry.event === "impact",
    );
    if (impact) player.seekTo(impact.frame_index);
  }, [clip, options, player, run]);

  const shown = player.residual?.landed ?? player.frame;
  const interval = analysis.phases
    ? phaseAt(analysis.phases, shown)
    : undefined;
  const style = interval ? PHASE_STYLE[interval.phase] : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">Swing</h2>
          <p className="text-muted-foreground mt-1 truncate text-sm">
            {clip
              ? basename(clip.path)
              : "Choose a clip to play it frame by frame and measure it."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {clip ? (
            <Button
              size="sm"
              onClick={() => void analyse()}
              disabled={busy !== null}
            >
              {busy ? <Loader2 className="animate-spin" /> : <Sparkles />}
              Analyse
            </Button>
          ) : null}
          <Button
            size="sm"
            variant={clip ? "outline" : "default"}
            onClick={() => void choose()}
            disabled={busy !== null}
          >
            <FolderOpen /> Choose clip
          </Button>
        </div>
      </div>

      {clip === null ? (
        <Card>
          <CardContent className="text-muted-foreground flex flex-col items-center gap-3 py-12 text-center text-sm">
            <FileVideo className="size-8 opacity-40" aria-hidden="true" />
            <p className="max-w-md">
              No clip loaded. The file picker runs outside the web view, and the
              clip you choose is the only file this window is allowed to read.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {busy ? (
        <Card>
          <CardContent className="py-6">
            <ProgressBar busy={busy} />
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <EngineErrorPanel error={error} onRetry={() => void analyse()} />
      ) : null}

      {clip ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between gap-4">
              <span>Player</span>
              {style ? (
                <span className={`text-sm font-medium ${style.text}`}>
                  {style.label}
                </span>
              ) : null}
            </CardTitle>
            <CardDescription>
              Seeks go to the midpoint of each frame&rsquo;s display interval,
              from times the engine read out of the container. What the browser
              reports painting is looked back up and shown below.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className="bg-muted relative mx-auto w-full max-w-lg overflow-hidden rounded-md"
              style={{
                aspectRatio: `${String(clip.metadata.stream.display_width)} / ${String(clip.metadata.stream.display_height)}`,
              }}
            >
              <video
                ref={video}
                src={clipSource(clip.path)}
                className="absolute inset-0 size-full"
                playsInline
                muted
                // The whole clip, not just its metadata. This player exists to
                // be scrubbed, and a seek into an unbuffered part of the file
                // displays nothing until the fetch lands -- which on a local
                // file is pure latency for no saving. A swing clip is seconds
                // long and already on this machine.
                preload="auto"
                data-testid="clip-video"
              />
              {/* Absolutely over the video and the same box, so a coordinate
                  that is a fraction of the frame is a fraction of this canvas.
                  Both are laid out by the wrapper's aspect ratio rather than by
                  object-fit, which would letterbox one and not the other. */}
              <OverlayCanvas
                overlay={analysis.overlay}
                frame={shown}
                className="pointer-events-none absolute inset-0 size-full"
              />
            </div>

            {analysis.overlay?.undistorted ? (
              <p className="text-status-degraded text-xs">
                These landmarks have had a lens correction applied, so the
                skeleton will not sit exactly on the pixels underneath it. That
                is the correction, not a rendering fault.
              </p>
            ) : null}

            {analysis.phases?.detected ? (
              <PhaseTimeline
                result={analysis.phases}
                frame={shown}
                onSeek={player.seekTo}
              />
            ) : null}

            <Transport
              player={player}
              phases={analysis.phases}
              lastFrame={Math.max(clip.index.frame_count - 1, 0)}
            />

            {(clip.index.warnings ?? []).length > 0 ? (
              <ul className="text-status-degraded space-y-2 text-sm">
                {(clip.index.warnings ?? []).map((warning) => (
                  <li key={warning} className="flex gap-2">
                    <span aria-hidden="true">&middot;</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {clip ? (
        <Card>
          <CardHeader>
            <CardTitle>Analysis inputs</CardTitle>
            <CardDescription>
              Neither of these can be read from the file. Both change every
              number below.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Inputs
              options={options}
              onChange={setOptions}
              disabled={busy !== null}
            />
          </CardContent>
        </Card>
      ) : null}

      {clip && !analysis.phases && !busy ? (
        <Card>
          <CardContent className="text-muted-foreground flex items-center gap-3 py-8 text-sm">
            <Play className="size-5 opacity-40" aria-hidden="true" />
            Nothing measured yet. Analysing extracts poses, detects the swing,
            measures it and reaches whatever conclusions the measurements
            support.
          </CardContent>
        </Card>
      ) : null}

      {analysis.phases && !analysis.phases.detected ? (
        <Card className="border-status-degraded/40">
          <CardHeader>
            <CardTitle className="text-status-degraded text-base">
              No swing detected in this clip
            </CardTitle>
            <CardDescription>
              A result, not a failure. The player above still works.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {(analysis.phases.warnings ?? []).map((warning) => (
                <li key={warning} className="flex gap-2">
                  <span aria-hidden="true">&middot;</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ) : null}

      {analysis.metrics ? (
        <MetricsPanel result={analysis.metrics} onSeek={player.seekTo} />
      ) : null}

      {analysis.coaching ? (
        <FindingsPanel report={analysis.coaching} onSeek={player.seekTo} />
      ) : null}
    </div>
  );
}
