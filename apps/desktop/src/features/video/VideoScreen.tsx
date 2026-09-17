/**
 * Video import and metadata screen.
 *
 * Shows what the engine read out of a clip's container. The presentation is
 * built around the two facts that decide whether the rest of the pipeline can
 * trust the file -- how it is rotated, and whether its frames are evenly spaced
 * in time -- rather than around the fields that are easiest to display.
 *
 * Nothing here is computed in the browser. Every number shown is a field the
 * engine measured, and a field the engine could not determine renders as
 * "unknown" rather than as a default.
 */

import type { EngineError, EngineResult, VideoMetadata } from "@gsa/types";
import { open } from "@tauri-apps/plugin-dialog";
import { FileVideo, FolderOpen, Loader2, RefreshCw } from "lucide-react";
import { useCallback, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { PhasesPanel } from "@/features/video/PhasesPanel";
import { PosePanel } from "@/features/video/PosePanel";
import { SyncPanel } from "@/features/video/SyncPanel";
import { probeVideo } from "@/lib/ipc";

type ScreenState =
  | { phase: "empty" }
  | { phase: "probing"; path: string }
  | { phase: "loaded"; metadata: VideoMetadata }
  | { phase: "failed"; path: string; error: EngineError };

/** Containers worth offering. The engine accepts whatever ffprobe can read. */
const VIDEO_EXTENSIONS = ["mp4", "mov", "m4v", "avi", "mkv"];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit] ?? "GB"}`;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return minutes > 0
    ? `${String(minutes)}m ${remainder.toFixed(2)}s`
    : `${remainder.toFixed(3)}s`;
}

function basename(path: string): string {
  return path.split("/").pop() ?? path;
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

/**
 * The frame-timing panel.
 *
 * Given its own card because variable frame rate is the single fact that most
 * changes what later phases may do with a clip, and because the verdict is
 * shown next to the measurement it was drawn from. `is_vfr` is nullable on
 * purpose: when the container carried no timestamps the answer is unknown, and
 * "no" would be a guess dressed as a finding.
 */
function TimingPanel({ metadata }: { metadata: VideoMetadata }) {
  const { timing } = metadata;
  const { intervals } = timing;

  const verdict =
    timing.is_vfr == null
      ? { label: "Unknown", tone: "text-status-degraded" }
      : timing.is_vfr
        ? { label: "Variable", tone: "text-status-degraded" }
        : { label: "Constant", tone: "text-status-ok" };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Frame timing</span>
          <span className={`text-sm font-medium ${verdict.tone}`}>
            {verdict.label}
          </span>
        </CardTitle>
        <CardDescription>
          {timing.source === "decoded_frames"
            ? "Frame times read from the container's presentation timestamps."
            : "No timestamps in the container; frame times were synthesised from the declared rate."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          <Field label="Frames" value={String(timing.frame_count)} />
          <Field label="Duration" value={formatDuration(timing.duration_s)} />
          <Field
            label="Measured rate"
            value={
              timing.measured_fps == null
                ? "unknown"
                : `${timing.measured_fps.toFixed(3)} fps`
            }
          />
          <Field
            label="Declared rate"
            value={
              timing.nominal_fps == null
                ? "not stated"
                : `${timing.nominal_fps.toFixed(3)} fps`
            }
          />
          {intervals ? (
            <>
              <Field
                label="Median interval"
                value={`${(intervals.median_s * 1000).toFixed(3)} ms`}
              />
              <Field
                label="Interval range"
                value={`${(intervals.min_s * 1000).toFixed(3)} – ${(
                  intervals.max_s * 1000
                ).toFixed(3)} ms`}
              />
              <Field
                label="Irregular intervals"
                value={`${String(intervals.irregular_count)} of ${String(
                  timing.frame_count - 1,
                )}`}
              />
              <Field
                label="Clock resolution"
                value={`${(intervals.quantum_s * 1e6).toFixed(1)} µs`}
              />
            </>
          ) : null}
        </dl>
      </CardContent>
    </Card>
  );
}

function StreamPanel({ metadata }: { metadata: VideoMetadata }) {
  const { stream } = metadata;
  const rotated = stream.rotation_ccw_degrees !== 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Stream</CardTitle>
        <CardDescription>{metadata.container_format}</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          <Field
            label="Codec"
            value={[stream.codec_name, stream.profile]
              .filter(Boolean)
              .join(" ")}
          />
          <Field label="Pixel format" value={stream.pix_fmt ?? "unknown"} />
          <Field
            label="Display size"
            value={`${String(stream.display_width)}×${String(stream.display_height)}`}
          />
          {/* Only shown when it differs, so its presence is itself the signal
              that the file is stored in a different orientation than shown. */}
          {rotated ? (
            <Field
              label="Stored size"
              value={`${String(stream.coded_width)}×${String(stream.coded_height)}`}
            />
          ) : null}
          <Field
            label="Rotation"
            value={
              rotated
                ? `${String(stream.rotation_ccw_degrees)}° ccw (${stream.rotation_source ?? "unknown source"})`
                : "none"
            }
          />
          <Field label="Size" value={formatBytes(metadata.file_size_bytes)} />
        </dl>
      </CardContent>
    </Card>
  );
}

export function VideoScreen() {
  const [state, setState] = useState<ScreenState>({ phase: "empty" });

  const probe = useCallback(async (path: string, refresh = false) => {
    setState({ phase: "probing", path });
    const result: EngineResult<VideoMetadata> = await probeVideo(path, {
      refresh,
    });
    setState(
      result.ok
        ? { phase: "loaded", metadata: result.value }
        : { phase: "failed", path, error: result.error },
    );
  }, []);

  const choose = useCallback(async () => {
    // The picker is the only filesystem capability the app has; it returns a
    // path, and the WebView never opens the file itself.
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Video", extensions: VIDEO_EXTENSIONS }],
    });
    if (typeof selected === "string") await probe(selected);
  }, [probe]);

  const currentPath =
    state.phase === "loaded"
      ? state.metadata.path
      : state.phase === "probing" || state.phase === "failed"
        ? state.path
        : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">Video</h2>
          <p className="text-muted-foreground mt-1 truncate text-sm">
            {currentPath
              ? basename(currentPath)
              : "Import a clip to read its container metadata."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {state.phase === "loaded" ? (
            <Button
              onClick={() => void probe(state.metadata.path, true)}
              variant="outline"
              size="sm"
            >
              <RefreshCw /> Re-read
            </Button>
          ) : null}
          <Button
            onClick={() => void choose()}
            size="sm"
            disabled={state.phase === "probing"}
          >
            {state.phase === "probing" ? (
              <Loader2 className="animate-spin" />
            ) : (
              <FolderOpen />
            )}
            Choose video
          </Button>
        </div>
      </div>

      {state.phase === "empty" ? (
        <Card>
          <CardContent className="text-muted-foreground flex flex-col items-center gap-3 py-12 text-center text-sm">
            <FileVideo className="size-8 opacity-40" aria-hidden="true" />
            <p className="max-w-md">
              No clip loaded. Choose a video to see what the engine reads from
              its container — resolution, codec, rotation, and whether its
              frames are evenly spaced in time.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {state.phase === "probing" ? (
        <Card>
          <CardContent className="text-muted-foreground flex items-center gap-3 py-8 text-sm">
            <Loader2 className="size-4 animate-spin" />
            Reading the container index.
          </CardContent>
        </Card>
      ) : null}

      {state.phase === "failed" ? (
        <EngineErrorPanel
          error={state.error}
          onRetry={() => void probe(state.path, true)}
          retryLabel="Try again"
        />
      ) : null}

      {state.phase === "loaded" ? (
        <>
          {state.metadata.warnings?.length ? (
            <Card className="border-status-degraded/40">
              <CardHeader>
                <CardTitle className="text-status-degraded text-base">
                  Caveats
                </CardTitle>
                <CardDescription>
                  Things about this clip that affect how it can be analysed.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {state.metadata.warnings.map((warning) => (
                    <li key={warning} className="flex gap-2">
                      <span aria-hidden="true">&middot;</span>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ) : null}

          <TimingPanel metadata={state.metadata} />
          <StreamPanel metadata={state.metadata} />
          <PosePanel
            key={state.metadata.path}
            videoPath={state.metadata.path}
          />
          <PhasesPanel
            key={`phases-${state.metadata.path}`}
            videoPath={state.metadata.path}
          />
          <SyncPanel
            key={`sync-${state.metadata.path}`}
            referencePath={state.metadata.path}
          />
        </>
      ) : null}
    </div>
  );
}
