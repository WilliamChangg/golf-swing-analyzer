/**
 * The swing in three dimensions, beside the footage it came from.
 *
 * ## Why the two are on one screen
 *
 * A reconstruction is the only thing this engine produces that cannot be checked
 * by looking at it. A metric can be drawn on the frame it came from; a phase
 * boundary can be stepped to; a 3D skeleton looks equally plausible whether it is
 * right or fifteen centimetres out in depth, because the error is in the
 * direction a picture has least to say about. So the video is here, on the same
 * clock, and the pose on the left has to be the pose on the right.
 *
 * ## The frame is one number, and it comes from the browser
 *
 * `useFramePlayer` already owns it: it seeks by the engine's measured
 * `SeekIndex` and reports the frame the browser says it **painted**, which is
 * what the rest of the app reads. The viewport reads the same value — so "the 3D
 * view is in sync with the video" is not a claim about two clocks staying
 * together, it is the same number rendered twice. There is no second timeline to
 * drift.
 *
 * What that leaves is the question a shared number cannot answer: whether the
 * pixels and the geometry are the same *recording*. The scene carries the
 * reference clip's `ContentKey` and the seek index carries the loaded clip's, and
 * `sceneMatchesClip` compares them. A mismatch stops the pairing rather than
 * scrubbing one swing's body against another's footage, which is a failure that
 * would look like a reconstruction error and never like a mistake about files.
 *
 * ## Why the clip has to be chosen by hand
 *
 * The project already knows its reference clip's path, and the WebView is still
 * not allowed to read it. Phase 14 shipped an asset scope that is **empty** and
 * a `choose_clip` command that opens its own dialog in Rust, deliberately with no
 * counterpart that takes a path — a grant has to be tied to something the
 * frontend could not have fabricated. Adding one here to save a click would undo
 * the only thing making that scope mean anything, so the clip is picked, and the
 * content key is what confirms the right one was.
 */

import type {
  EngineError,
  ProgressUpdate,
  ProjectList,
  ReconstructionScene,
  SeekIndex,
  VideoMetadata,
} from "@gsa/types";
import {
  Box,
  Camera,
  Compass,
  FolderOpen,
  Loader2,
  RotateCcw,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { OverlayCanvas } from "@/features/player/OverlayCanvas";
import { useFramePlayer } from "@/features/player/useFramePlayer";
import {
  MAGNIFICATIONS,
  orbitCamera,
  referenceCamera,
  type Magnification,
  type Orbit,
  type ViewCamera,
} from "@/features/scene/projection";
import {
  defaultDistance,
  frameAt,
  sceneMatchesClip,
  viewpointHonesty,
} from "@/features/scene/scene";
import { SceneViewport } from "@/features/scene/SceneViewport";
import { ViewpointPanel } from "@/features/scene/ViewpointPanel";
import {
  chooseClip,
  clipSource,
  listProjects,
  onProgress,
  probeVideo,
  progressFraction,
  reconstructScene,
  seekIndex as fetchSeekIndex,
} from "@/lib/ipc";

/**
 * Field of view for the free camera.
 *
 * Narrower than the reference camera's 69 degrees because a free view is usually
 * closer than the tripod was, and a wide angle at close range exaggerates depth
 * — which is the one thing this viewport must not do, since depth is the
 * direction the reconstruction knows least about.
 */
const FREE_FOV_DEG = 45;

/** Degrees of orbit per pixel dragged. A full turn is about a screen width. */
const DRAG_SENSITIVITY = 0.4;

const START: Orbit = { azimuthDeg: 0, elevationDeg: 0, distance: 0 };

type Busy = { label: string; update: ProgressUpdate | null } | null;

interface Clip {
  path: string;
  metadata: VideoMetadata;
  index: SeekIndex;
}

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

export function SceneScreen() {
  const [projects, setProjects] = useState<ProjectList | null>(null);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [scene, setScene] = useState<ReconstructionScene | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<EngineError | null>(null);

  const [orbit, setOrbit] = useState<Orbit>(START);
  const [locked, setLocked] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(true);
  // 20x by default. At true scale a well-conditioned joint's 1-sigma ellipse is
  // about two thirds of a screen pixel — see `MAGNIFICATIONS` — so the honest
  // drawing is an invisible one. The factor is always printed in the corner.
  const [magnification, setMagnification] = useState<Magnification>(20);
  const [showTrajectories, setShowTrajectories] = useState(true);
  const [showCameras, setShowCameras] = useState(true);
  const [standalone, setStandalone] = useState(0);

  const video = useRef<HTMLVideoElement | null>(null);
  const player = useFramePlayer(video, clip?.index ?? null);
  const unlisten = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  useEffect(() => {
    let live = true;
    void listProjects().then((result) => {
      if (!live) return;
      if (result.ok) {
        setProjects(result.value);
        setProjectId(
          (current) => current ?? result.value.projects?.[0]?.id ?? null,
        );
      } else {
        setError(result.error);
      }
    });
    return () => {
      live = false;
    };
  }, []);

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

  const reconstruct = useCallback(async () => {
    if (projectId === null) return;
    const built = await run("Triangulating", () => reconstructScene(projectId));
    if (!built) return;
    setScene(built);
    setOrbit({
      azimuthDeg: 0,
      elevationDeg: 0,
      distance: defaultDistance(built),
    });
    setLocked(true);
    setStandalone(built.start_frame);
  }, [projectId, run]);

  const loadClip = useCallback(async () => {
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
    setClip({ path: chosen, metadata, index });
  }, [run]);

  const matched = scene ? sceneMatchesClip(scene, clip?.index ?? null) : false;

  // **The one number.** With a clip loaded and proven to be the reference, it is
  // whatever the browser reported painting; without one, the viewport scrubs
  // itself. It is never two values reconciled, because two values are what drift.
  const frameIndex =
    clip && matched ? (player.residual?.landed ?? player.frame) : standalone;
  const frame = scene ? frameAt(scene, frameIndex) : null;

  const camera: ViewCamera | null = useMemo(() => {
    if (!scene) return null;
    return locked
      ? referenceCamera(scene)
      : orbitCamera(scene, orbit, FREE_FOV_DEG);
  }, [scene, locked, orbit]);

  const honesty = useMemo(
    () =>
      frame
        ? viewpointHonesty(frame, camera)
        : { drawn: 0, sigmaM: null, visible: null, apparentSigmaM: null },
    [frame, camera],
  );

  const onDrag = useCallback((dx: number, dy: number) => {
    // Any drag leaves the reference camera. Silently ignoring the drag while
    // locked would look broken; silently orbiting away from a view the reader
    // asked to keep would be worse, so the toggle follows the gesture and says
    // so on the button.
    setLocked(false);
    setOrbit((current) => ({
      azimuthDeg: current.azimuthDeg + dx * DRAG_SENSITIVITY,
      elevationDeg: Math.max(
        -85,
        Math.min(85, current.elevationDeg + dy * DRAG_SENSITIVITY),
      ),
      distance: current.distance,
    }));
  }, []);

  const onZoom = useCallback(
    (factor: number) => {
      if (!scene) return;
      setLocked(false);
      setOrbit((current) => ({
        ...current,
        distance: Math.max(
          scene.radius_m * 0.6,
          Math.min(current.distance * factor, scene.radius_m * 20),
        ),
      }));
    },
    [scene],
  );

  const resetView = useCallback(() => {
    if (!scene) return;
    setOrbit({
      azimuthDeg: 0,
      elevationDeg: 0,
      distance: defaultDistance(scene),
    });
    setLocked(true);
  }, [scene]);

  const entries = projects?.projects ?? [];
  const lastFrame = scene
    ? Math.max(scene.end_frame - 1, scene.start_frame)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">
            Three dimensions
          </h2>
          <p className="text-muted-foreground mt-1 text-sm">
            Two calibrated views, triangulated into metres. The only thing here
            that a single camera cannot produce.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-sm">
            <span className="sr-only">Session</span>
            <select
              aria-label="Session"
              className="border-input rounded-md border px-2 py-1 text-sm"
              value={projectId ?? ""}
              disabled={busy !== null || entries.length === 0}
              onChange={(event) => {
                setProjectId(Number(event.target.value));
                setScene(null);
              }}
            >
              {entries.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </label>
          <Button
            size="sm"
            disabled={busy !== null || projectId === null}
            onClick={() => void reconstruct()}
          >
            {busy ? <Loader2 className="animate-spin" /> : <Box />}
            Reconstruct
          </Button>
        </div>
      </div>

      {entries.length === 0 && !busy ? (
        <Card>
          <CardContent className="text-muted-foreground py-8 text-sm">
            No sessions yet. A reconstruction needs a session holding two clips
            of one swing, a calibration relating the two cameras, and an
            alignment relating their clocks — none of which can be recovered
            from a single file.
          </CardContent>
        </Card>
      ) : null}

      {busy ? (
        <Card>
          <CardContent className="space-y-2 py-6">
            <div className="text-muted-foreground flex justify-between text-sm">
              <span>{busy.label}</span>
              <span className="font-mono">
                {busy.update === null
                  ? "—"
                  : `${String(Math.round((progressFraction(busy.update) ?? 0) * 100))}%`}
              </span>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <EngineErrorPanel error={error} onRetry={() => void reconstruct()} />
      ) : null}

      {scene ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center justify-between gap-3">
                <span>Viewport</span>
                <span className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    variant={locked ? "default" : "outline"}
                    onClick={resetView}
                  >
                    <Camera /> Reference camera
                  </Button>
                  <Button size="sm" variant="outline" onClick={resetView}>
                    <RotateCcw /> Reset view
                  </Button>
                </span>
              </CardTitle>
              <CardDescription>
                {locked
                  ? "Looking from the camera that filmed the reference clip, with its own focal length — the same projection that produced the video beside it."
                  : `Orbited ${orbit.azimuthDeg.toFixed(0)}° round and ${orbit.elevationDeg.toFixed(0)}° up from that camera. Drag to turn, scroll to move closer.`}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
                <div
                  className="bg-muted overflow-hidden rounded-md"
                  style={{
                    aspectRatio: `${String(camera?.width ?? 16)} / ${String(camera?.height ?? 9)}`,
                  }}
                >
                  <SceneViewport
                    scene={scene}
                    frame={frame}
                    camera={camera}
                    showUncertainty={showUncertainty}
                    magnification={magnification}
                    showTrajectories={showTrajectories}
                    showCameras={showCameras}
                    onDrag={onDrag}
                    onZoom={onZoom}
                  />
                </div>
                <ViewpointPanel scene={scene} frame={frame} honesty={honesty} />
              </div>

              <div className="flex flex-wrap items-center gap-4 text-sm">
                <Toggle
                  label="Uncertainty"
                  checked={showUncertainty}
                  onChange={setShowUncertainty}
                />
                <label className="flex items-center gap-2">
                  <span className="sr-only">Uncertainty magnification</span>
                  <select
                    aria-label="Uncertainty magnification"
                    className="border-input rounded-md border px-2 py-1 text-sm"
                    value={magnification}
                    disabled={!showUncertainty}
                    onChange={(event) => {
                      setMagnification(
                        Number(event.target.value) as Magnification,
                      );
                    }}
                  >
                    {MAGNIFICATIONS.map((factor) => (
                      <option key={factor} value={factor}>
                        &times;{factor}
                      </option>
                    ))}
                  </select>
                </label>
                <Toggle
                  label="Hand paths"
                  checked={showTrajectories}
                  onChange={setShowTrajectories}
                />
                <Toggle
                  label="Cameras"
                  checked={showCameras}
                  onChange={setShowCameras}
                />
              </div>

              {(scene.warnings ?? []).length > 0 ? (
                <ul className="text-status-degraded space-y-2 text-sm">
                  {(scene.warnings ?? []).map((warning) => (
                    <li key={warning} className="flex gap-2">
                      <span aria-hidden="true">&middot;</span>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center justify-between gap-3">
                <span>The reference clip</span>
                <Button
                  size="sm"
                  variant={clip ? "outline" : "default"}
                  disabled={busy !== null}
                  onClick={() => void loadClip()}
                >
                  <FolderOpen /> {clip ? "Choose another" : "Choose the clip"}
                </Button>
              </CardTitle>
              <CardDescription>
                The frame numbers above are this clip&rsquo;s. Loading it puts
                the picture and the geometry on one number — the frame the
                browser reports painting.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {clip ? (
                <>
                  <div
                    className="bg-muted relative mx-auto w-full max-w-sm overflow-hidden rounded-md"
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
                      preload="auto"
                      data-testid="scene-video"
                    />
                    <OverlayCanvas
                      overlay={null}
                      frame={frameIndex}
                      className="pointer-events-none absolute inset-0 size-full"
                    />
                  </div>

                  {matched ? (
                    <p
                      className="text-muted-foreground text-xs"
                      data-testid="scene-clip-matched"
                    >
                      {basename(clip.path)} is the clip this reconstruction was
                      built from — matched by content, so a move or a rename
                      does not break it.
                    </p>
                  ) : (
                    <p
                      className="text-status-missing text-sm"
                      data-testid="scene-clip-mismatch"
                    >
                      {basename(clip.path)} is not the clip this reconstruction
                      was built from. Its contents hash differently from{" "}
                      {scene.reference_name}, so its frame numbers mean
                      something else — the viewport is scrubbing on its own
                      rather than showing this footage&rsquo;s geometry.
                    </p>
                  )}
                </>
              ) : (
                <p className="text-muted-foreground flex items-center gap-2 text-sm">
                  <Video className="size-4 opacity-50" aria-hidden="true" />
                  No clip loaded. The viewport scrubs on its own below; loading
                  the reference clip is what lets the two be checked against
                  each other.
                </p>
              )}

              <div className="flex items-center gap-3">
                <Compass
                  className="text-muted-foreground size-4"
                  aria-hidden="true"
                />
                <input
                  type="range"
                  className="flex-1"
                  aria-label="Frame"
                  min={scene.start_frame}
                  max={lastFrame}
                  value={frameIndex}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    if (clip && matched) player.seekTo(next);
                    else setStandalone(next);
                  }}
                />
                <span
                  className="w-28 text-right font-mono text-sm"
                  data-testid="scene-frame"
                >
                  {frameIndex}
                  <span className="text-muted-foreground">
                    {" / "}
                    {lastFrame}
                  </span>
                </span>
              </div>

              {clip && matched && player.observable ? (
                <p className="text-muted-foreground text-xs">
                  {player.residual && player.residual.delta !== 0
                    ? `The browser painted frame ${String(player.residual.landed)} for a request of ${String(player.residual.requested)}. The viewport is drawing what was painted, not what was asked for.`
                    : "Every seek has landed on the frame it asked for."}
                </p>
              ) : null}
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => {
          onChange(event.target.checked);
        }}
      />
      <span>{label}</span>
    </label>
  );
}
