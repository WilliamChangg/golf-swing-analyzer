/**
 * Typed wrappers over the Tauri command surface.
 *
 * Every engine call goes through here rather than calling `invoke` directly, so
 * that error normalisation happens in exactly one place and components only
 * ever see an `EngineResult`. React code therefore cannot forget to handle a
 * failure: the discriminated union forces it.
 */

import type {
  CoachingReport,
  EngineError,
  EngineResult,
  EnvironmentReport,
  MetricSet,
  PoseExtractionResult,
  PoseOverlay,
  ProgressUpdate,
  Project,
  ProjectList,
  CameraCalibration,
  CameraRig,
  CameraRole,
  SeekIndex,
  SwingPhases,
  SyncModel,
  VideoMetadata,
} from "@gsa/types";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** Tauri command names. Must match the `#[tauri::command]` functions in Rust. */
const COMMANDS = {
  doctor: "doctor",
  probeVideo: "probe_video",
  extractPoses: "extract_poses",
  detectPhases: "detect_phases",
  syncClips: "sync_clips",
  calibrateCamera: "calibrate_camera",
  getCalibration: "get_calibration",
  chooseClip: "choose_clip",
  seekIndex: "seek_index",
  poseOverlay: "pose_overlay",
  computeMetrics: "compute_metrics",
  coachSwing: "coach_swing",
  listProjects: "list_projects",
  getProject: "get_project",
  createProject: "create_project",
  deleteProject: "delete_project",
  addClip: "add_clip",
  removeClip: "remove_clip",
} as const;

/** Event Rust re-emits engine progress notifications on. */
const PROGRESS_EVENT = "engine://progress";

/**
 * Shape the Rust layer serialises its errors into.
 *
 * Kept local because it is an implementation detail of the boundary; callers
 * receive the normalised `EngineError` instead.
 */
interface RawEngineError {
  kind?: string;
  message?: string;
  code?: number;
  data?: Record<string, unknown>;
}

const ERROR_KINDS = new Set([
  "spawn",
  "transport",
  "protocol",
  "method",
  "timeout",
]);

/**
 * Coerce an unknown rejection into an `EngineError`.
 *
 * `invoke` rejects with whatever the Rust side serialised, but it can also
 * reject with a plain string (for example when the command name is unknown) or
 * a JS Error if something failed before reaching Rust. Treating an unrecognised
 * rejection as `transport` is deliberate: it is reported as a real failure
 * rather than being flattened into a generic message that hides the cause.
 */
export function normalizeError(raw: unknown): EngineError {
  if (typeof raw === "string") {
    return { kind: "transport", message: raw };
  }

  if (raw instanceof Error) {
    return { kind: "transport", message: raw.message };
  }

  if (typeof raw === "object" && raw !== null) {
    const candidate = raw as RawEngineError;
    const kind =
      candidate.kind && ERROR_KINDS.has(candidate.kind)
        ? (candidate.kind as EngineError["kind"])
        : "transport";

    const error: EngineError = {
      kind,
      message: candidate.message ?? "The analysis engine returned an error.",
    };
    if (candidate.code !== undefined) error.code = candidate.code;
    if (candidate.data !== undefined) error.data = candidate.data;
    return error;
  }

  return {
    kind: "transport",
    message: "Unknown error from the analysis engine.",
  };
}

async function call<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<EngineResult<T>> {
  try {
    // A command that takes no arguments is invoked with none, rather than with
    // an empty object: the engine rejects unexpected parameters, so what goes
    // on the wire is worth keeping exact.
    const value =
      args === undefined
        ? await invoke<T>(command)
        : await invoke<T>(command, args);
    return { ok: true, value };
  } catch (raw) {
    return { ok: false, error: normalizeError(raw) };
  }
}

/**
 * Run the environment health check.
 *
 * Spawns the Python worker if it is not already running, so this is also the
 * first real test that the engine is installed and reachable.
 */
export function doctor(): Promise<EngineResult<EnvironmentReport>> {
  return call<EnvironmentReport>(COMMANDS.doctor);
}

/**
 * Read a video file's container metadata without decoding it.
 *
 * Results are cached by content, so re-opening the same clip is immediate.
 * `refresh` forces a re-read, which is what a user asking to check the file
 * again expects.
 *
 * An unusable file (missing, empty, audio-only, corrupt) comes back as an
 * `EngineError` of kind "method" carrying `data.remediation` — the engine knows
 * what is wrong with the file, so it is the engine that says how to fix it.
 */
export function probeVideo(
  path: string,
  options: { refresh?: boolean } = {},
): Promise<EngineResult<VideoMetadata>> {
  return call<VideoMetadata>(COMMANDS.probeVideo, {
    path,
    refresh: options.refresh ?? false,
  });
}

/** The remediation the engine attached to a failure, when it attached one. */
export function remediationOf(error: EngineError): string | null {
  const value = error.data?.["remediation"];
  return typeof value === "string" && value.length > 0 ? value : null;
}

/**
 * Run pose estimation over a clip and store the landmarks.
 *
 * Long-running — minutes on a high-frame-rate clip. Subscribe with
 * `onProgress` before calling this to show real completion; the promise does
 * not settle until the whole extraction is done.
 *
 * The landmarks are not in the result. A 240 fps clip is tens of thousands of
 * frames of 33 landmarks in two coordinate spaces, so the engine writes them to
 * a Parquet file and returns its path.
 */
export function extractPoses(
  path: string,
  options: { model?: string } = {},
): Promise<EngineResult<PoseExtractionResult>> {
  return call<PoseExtractionResult>(COMMANDS.extractPoses, {
    path,
    model: options.model ?? null,
  });
}

/**
 * Locate the takeaway, top, impact and finish in a clip's extracted landmarks.
 *
 * Requires `extractPoses` to have run for the clip; the engine resolves the
 * landmarks through its content-keyed cache and returns an error naming the
 * extract command when there are none.
 *
 * **`detected: false` is a success.** A clip with no swing in it is an answer
 * the engine can give, and it arrives with warnings explaining what it found
 * instead. Treating it as a failure would file "nobody swung" alongside "the
 * worker crashed".
 *
 * `windowS` overrides the filter's smoothing window. Worth exposing because a
 * clip below about 60 fps cannot support the default, and the engine's own
 * message names the width that clip's frame rate would support.
 */
export function detectPhases(
  path: string,
  options: { model?: string; windowS?: number } = {},
): Promise<EngineResult<SwingPhases>> {
  return call<SwingPhases>(COMMANDS.detectPhases, {
    path,
    model: options.model ?? null,
    windowS: options.windowS ?? null,
  });
}

/** One instant a person identified in both clips of a pair. */
export interface ManualAnchor {
  label: string;
  referenceFrame: number;
  targetFrame: number;
}

/**
 * Relate two clips' clocks, and report how well the relation is known.
 *
 * Requires `extractPoses` to have run for both clips. Slow-motion factors are
 * per clip because two cameras in one session routinely differ; the smoothing
 * window is shared, because smoothing two clips differently would shift the
 * very features the alignment keys on — which means the *coarser* clip sets it,
 * and a 30 fps camera cannot support the engine's 0.10 s default.
 *
 * **`aligned: false` is a success**, exactly as `detected: false` is for
 * `detectPhases`. It arrives with a `refusal` saying what was found instead.
 *
 * Passing `anchors` switches the engine to the manual method: a person who has
 * looked at both frames is supplying the answer, not an estimate for the engine
 * to weigh against its own.
 */
export function syncClips(
  referencePath: string,
  targetPath: string,
  options: {
    model?: string;
    referenceSlowMotion?: number;
    targetSlowMotion?: number;
    windowS?: number;
    anchors?: ManualAnchor[];
  } = {},
): Promise<EngineResult<SyncModel>> {
  return call<SyncModel>(COMMANDS.syncClips, {
    referencePath,
    targetPath,
    model: options.model ?? null,
    referenceSlowMotion: options.referenceSlowMotion ?? null,
    targetSlowMotion: options.targetSlowMotion ?? null,
    windowS: options.windowS ?? null,
    anchors:
      options.anchors?.map((anchor) => ({
        label: anchor.label,
        reference_frame: anchor.referenceFrame,
        target_frame: anchor.targetFrame,
      })) ?? null,
  });
}

/**
 * Measure one camera's intrinsics from footage of a Charuco board.
 *
 * **`usable: false` is a success**, exactly as `aligned: false` is for
 * `syncClips` and `detected: false` for `detectPhases`. It arrives with a
 * `refusal` naming which bound the capture failed, and that is the useful part:
 * the numbers say what to reshoot, which an error would not.
 *
 * `projectId` stores the result on that project's rig. Without it the
 * calibration is computed and returned and nothing is written.
 */
export function calibrateCamera(
  source: string,
  options: {
    role?: CameraRole;
    projectId?: number;
    squaresX?: number;
    squaresY?: number;
    squareLengthMm?: number;
    stride?: number;
    notes?: string;
  } = {},
): Promise<EngineResult<CameraCalibration>> {
  return call<CameraCalibration>(COMMANDS.calibrateCamera, {
    source,
    role: options.role ?? "other",
    projectId: options.projectId ?? null,
    squaresX: options.squaresX ?? null,
    squaresY: options.squaresY ?? null,
    squareLengthMm: options.squareLengthMm ?? null,
    stride: options.stride ?? null,
    notes: options.notes ?? null,
  });
}

/**
 * What is known about a project's cameras, and therefore what it may claim.
 *
 * An uncalibrated project returns an empty rig whose `status` is `none`, rather
 * than null. That keeps "uncalibrated" a value every caller handles the same
 * way as any other status, instead of a null check each one writes separately
 * and one of them forgets.
 */
export function getCalibration(
  projectId: number,
): Promise<EngineResult<CameraRig>> {
  return call<CameraRig>(COMMANDS.getCalibration, { projectId });
}

/**
 * Open the native file picker, and admit the chosen clip for playback.
 *
 * Returns the chosen path, or null if the user cancelled — cancelling is not an
 * error and does not come back as one.
 *
 * **There is no `authorizeClip(path)` counterpart, and that absence is the
 * feature.** A video element means the WebView reads files, which every phase
 * before this one refused to allow. What makes it narrow is that the frontend
 * cannot name the file: the picker runs in Rust, and only what a human selected
 * there is added to the asset protocol's scope. A command that took a path and
 * granted access to it would hand the restricted party the key to its own
 * restriction. See `commands::choose_clip`.
 */
export function chooseClip(): Promise<EngineResult<string | null>> {
  return call<string | null>(COMMANDS.chooseClip);
}

/**
 * The URL a video element loads this clip from.
 *
 * Only resolvable for a clip that went through `chooseClip`; any other path
 * produces a URL the asset protocol refuses, which surfaces as a video that
 * will not load rather than as a thrown error.
 */
export function clipSource(path: string): string {
  return convertFileSrc(path);
}

/**
 * Every frame's presentation time, and the time to seek to to display it.
 *
 * The map the player runs on, and the reason it is not computed in the browser:
 * `frame / fps` is wrong on variable-rate footage, and measurably wrong even on
 * constant-rate footage whose container declares the wrong rate — which one of
 * this project's own reference clips does, by 8.4%.
 */
export function seekIndex(path: string): Promise<EngineResult<SeekIndex>> {
  return call<SeekIndex>(COMMANDS.seekIndex, { path });
}

/**
 * Filtered landmarks over a frame range, in the coordinates they are drawn in.
 *
 * A range, because the payload is 33 landmarks per frame and a canvas draws
 * one; the range exists so the UI makes one request per window rather than one
 * per frame. The engine refuses a range past its own limit rather than
 * truncating it.
 *
 * `withClub` costs a full decode pass — seconds, against milliseconds for the
 * skeleton — so it is a separate request rather than a flag on the first one.
 */
export function poseOverlay(
  path: string,
  range: { startFrame: number; endFrame?: number },
  options: AnalysisOptions & { withClub?: boolean } = {},
): Promise<EngineResult<PoseOverlay>> {
  return call<PoseOverlay>(COMMANDS.poseOverlay, {
    path,
    model: options.model ?? null,
    startFrame: range.startFrame,
    endFrame: range.endFrame ?? null,
    windowS: options.windowS ?? null,
    slowMotionFactor: options.slowMotionFactor ?? null,
    projectId: options.projectId ?? null,
    withClub: options.withClub ?? false,
  });
}

/**
 * Inputs shared by every analysis of one clip.
 *
 * One type rather than three copies because the three calls **must** agree: the
 * metrics panel, the findings panel and the overlay are read side by side, and
 * a different smoothing window in one of them would show a reader numbers that
 * disagree with the skeleton drawn under them, with nothing on screen to say
 * why.
 */
export interface AnalysisOptions {
  model?: string;
  /**
   * Smoothing window in seconds. Worth exposing because a clip below about
   * 60 fps cannot support the engine's 0.10 s default at all — it emits nothing
   * — and the engine's own message names the width that clip's rate supports.
   */
  windowS?: number;
  /** How many times slower than real time the clip plays. Supplied, never measured. */
  slowMotionFactor?: number;
  /** Apply this project's calibration, when the clip belongs to it. */
  projectId?: number;
}

/**
 * Measure the biomechanics metrics for a clip.
 *
 * **`computed: false` is a success**, and so is a long `refused` list. Phase 5
 * refuses three metrics outright on a down-the-line clip because that camera
 * position does not contain the measurement, which is a correct answer rather
 * than a failure.
 */
export function computeMetrics(
  path: string,
  options: AnalysisOptions = {},
): Promise<EngineResult<MetricSet>> {
  return call<MetricSet>(COMMANDS.computeMetrics, analysisArgs(path, options));
}

/**
 * Reach every conclusion this clip's measurements support, and refuse the rest.
 *
 * **An empty `findings` list is a success and is the ordinary outcome.** Across
 * four reference clips this engine produces at most two findings, and on the
 * 30 fps amateur clip it produces none — because one frame of ambiguity at the
 * top is worth 0.74 of the published tempo band. The `refused` list is where
 * nearly all of the information is, and the panel treats it that way.
 */
export function coachSwing(
  path: string,
  options: AnalysisOptions = {},
): Promise<EngineResult<CoachingReport>> {
  return call<CoachingReport>(COMMANDS.coachSwing, analysisArgs(path, options));
}

function analysisArgs(
  path: string,
  options: AnalysisOptions,
): Record<string, unknown> {
  return {
    path,
    model: options.model ?? null,
    windowS: options.windowS ?? null,
    slowMotionFactor: options.slowMotionFactor ?? null,
    projectId: options.projectId ?? null,
  };
}

/** Every project, with its clips. Stored alignments are omitted; see `getProject`. */
export function listProjects(): Promise<EngineResult<ProjectList>> {
  return call<ProjectList>(COMMANDS.listProjects);
}

/** One project, with its clips and its stored alignments. */
export function getProject(projectId: number): Promise<EngineResult<Project>> {
  return call<Project>(COMMANDS.getProject, { projectId });
}

/** Create an empty project. The name is not unique; the id is the identity. */
export function createProject(
  name: string,
  options: { notes?: string } = {},
): Promise<EngineResult<Project>> {
  return call<Project>(COMMANDS.createProject, {
    name,
    notes: options.notes ?? null,
  });
}

/**
 * Delete a project, its clips and its alignments. The video files are untouched.
 *
 * Returns the remaining projects rather than nothing, because a caller that has
 * just removed something is about to re-render the list.
 */
export function deleteProject(
  projectId: number,
): Promise<EngineResult<ProjectList>> {
  return call<ProjectList>(COMMANDS.deleteProject, { projectId });
}

/**
 * Attach a video to a project, identifying it **by content** rather than by path.
 *
 * `role` is where the camera was, as declared. Phase 6 measures the view from
 * the footage and may disagree; that disagreement is worth being able to state,
 * and cannot be stated unless the declaration was recorded.
 */
export function addClip(
  projectId: number,
  path: string,
  role: CameraRole,
  options: { slowMotionFactor?: number; label?: string } = {},
): Promise<EngineResult<Project>> {
  return call<Project>(COMMANDS.addClip, {
    projectId,
    path,
    role,
    slowMotionFactor: options.slowMotionFactor ?? null,
    label: options.label ?? null,
  });
}

/** Detach a clip from a project. Its stored alignments go with it. */
export function removeClip(
  projectId: number,
  clipId: number,
): Promise<EngineResult<Project>> {
  return call<Project>(COMMANDS.removeClip, { projectId, clipId });
}

/**
 * Subscribe to engine progress.
 *
 * Returns a promise of an unsubscribe function, which is Tauri's shape: the
 * listener is registered asynchronously, so unsubscribing has to wait for the
 * registration it is cancelling.
 *
 * Updates are filtered by `task` when one is given, because every long method
 * reports on the same channel and a screen only wants its own.
 */
export function onProgress(
  handler: (update: ProgressUpdate) => void,
  options: { task?: string } = {},
): Promise<() => void> {
  return listen<ProgressUpdate>(PROGRESS_EVENT, (event) => {
    if (options.task !== undefined && event.payload.task !== options.task)
      return;
    handler(event.payload);
  });
}

/**
 * Completed fraction of an update, or null when there is no total to divide by.
 *
 * Mirrors the `fraction` property on the Python contract, which is computed
 * rather than stored: a progress bar that invents a denominator is worse than
 * one that admits it does not have one.
 */
export function progressFraction(update: ProgressUpdate): number | null {
  if (update.total == null || update.total <= 0) return null;
  return Math.min(update.current / update.total, 1);
}
