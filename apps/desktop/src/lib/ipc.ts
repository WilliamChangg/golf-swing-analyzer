/**
 * Typed wrappers over the Tauri command surface.
 *
 * Every engine call goes through here rather than calling `invoke` directly, so
 * that error normalisation happens in exactly one place and components only
 * ever see an `EngineResult`. React code therefore cannot forget to handle a
 * failure: the discriminated union forces it.
 */

import type {
  EngineError,
  EngineResult,
  EnvironmentReport,
  PoseExtractionResult,
  ProgressUpdate,
  SwingPhases,
  VideoMetadata,
} from "@gsa/types";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** Tauri command names. Must match the `#[tauri::command]` functions in Rust. */
const COMMANDS = {
  doctor: "doctor",
  probeVideo: "probe_video",
  extractPoses: "extract_poses",
  detectPhases: "detect_phases",
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
