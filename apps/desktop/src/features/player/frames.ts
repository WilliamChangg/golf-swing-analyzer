/**
 * Converting between a frame index and a video element's clock.
 *
 * Pure functions over a `SeekIndex`, kept apart from the player so the
 * arithmetic can be tested without a DOM — and so that the one place the app
 * decides "which frame is on screen" is a function rather than a line buried in
 * an event handler.
 *
 * **Nothing here computes a timestamp.** The times come from the engine, which
 * read them out of the container. What these functions do is look them up, in
 * both directions: forward to seek, and backward to find out what a player
 * actually did. The backward direction is the one that matters, because asking
 * for a frame and getting it are different events and only the second is
 * observable.
 */

import type { SeekIndex } from "@gsa/types";

/**
 * How far below a frame's presentation time still counts as that frame.
 *
 * One microsecond, and it is not a fudge factor — it is the resolution of the
 * number being looked up. Chromium carries media time as whole microseconds, so
 * the `mediaTime` it reports for a frame at 0.13333333… seconds comes back as
 * 0.133333: **a third of a microsecond below the timestamp the container
 * actually holds.** Without this, a strict comparison puts that frame in the
 * previous one's interval, and the player reports a one-frame miss on roughly
 * half the frames of any clip whose frame times are not whole microseconds —
 * which is every clip at 30 fps.
 *
 * Measured, not assumed: `e2e/player.spec.ts` seeks a real clip in a real
 * browser and the differences observed are ±3.33e-7 s. A microsecond is
 * comfortably above that and four thousand times smaller than one frame at
 * 240 fps, so it can never merge two frames.
 */
const MEDIA_TIME_QUANTUM_S = 1e-6;

/**
 * Which frame is displayed at time `t`, by the rule every player implements.
 *
 * The last frame whose presentation time has been reached. Binary search
 * because this runs on every displayed frame during playback — up to 240 times
 * a second on the footage the capture protocol asks for — and a linear scan of
 * 14,400 timestamps there would be the most expensive thing on the screen.
 *
 * Clamped at zero: seeking before the first frame shows the first frame, which
 * is what a player does, rather than an index of −1 that every caller would
 * have to guard.
 */
export function frameAt(timestamps: number[], t: number): number {
  if (timestamps.length === 0) return 0;

  let low = 0;
  let high = timestamps.length - 1;
  let found = 0;

  while (low <= high) {
    const mid = (low + high) >>> 1;
    // `<=` rather than `<`: a time exactly on a frame's presentation stamp is
    // that frame, not the one before it. This is the boundary the seek targets
    // are built to avoid asking a decoder about, and the rule has to be stated
    // somewhere even so. The quantum widens it by the resolution of the clock
    // the time arrived on.
    if ((timestamps[mid] ?? 0) <= t + MEDIA_TIME_QUANTUM_S) {
      found = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return found;
}

/** Clamp a frame index into the clip. */
export function clampFrame(index: SeekIndex, frame: number): number {
  const last = Math.max(index.frame_count - 1, 0);
  if (!Number.isFinite(frame)) return 0;
  return Math.min(Math.max(Math.round(frame), 0), last);
}

/**
 * The time to seek to in order to display `frame`.
 *
 * The midpoint of the interval that frame is displayed for, computed by the
 * engine. Falling back to the frame's own presentation time when the index is
 * short would be seeking to a boundary, so an out-of-range frame is clamped
 * instead and the caller gets a real target for a real frame.
 */
export function seekTargetFor(index: SeekIndex, frame: number): number {
  const clamped = clampFrame(index, frame);
  return index.seek_targets_s[clamped] ?? index.timestamps_s[clamped] ?? 0;
}

/**
 * What the player did, against what it was asked to do.
 *
 * `requested` is the frame the UI wanted; `mediaTime` is the presentation time
 * of the frame the browser reports it actually painted. The difference between
 * the two, in frames, is the only honest measure of whether this player is
 * frame-accurate on this clip in this browser — and it is a measurement rather
 * than an assumption, which is why it is displayed instead of assumed to be
 * zero.
 */
export interface SeekResidual {
  requested: number;
  landed: number;
  /** `landed - requested`. Zero when the player showed the frame asked for. */
  delta: number;
}

export function residualFor(
  index: SeekIndex,
  requested: number,
  mediaTime: number,
): SeekResidual {
  const landed = frameAt(index.timestamps_s, mediaTime);
  return { requested, landed, delta: landed - requested };
}

/**
 * Whether this browser can say which frame it is showing.
 *
 * `requestVideoFrameCallback` is the only API that reports the presentation
 * time of the frame actually painted. Without it a player can still seek, but
 * it cannot check, and `currentTime` is not a substitute: browsers commonly
 * leave it at the value that was requested, so reading it back would confirm
 * the seek against itself.
 *
 * Exposed so the UI can say "unverified" rather than show a residual of zero it
 * has not measured.
 */
export function canObserveFrames(video: HTMLVideoElement | null): boolean {
  return (
    video !== null &&
    typeof (video as Partial<VideoFrameCallbackHost>)
      .requestVideoFrameCallback === "function"
  );
}

/** The slice of the `requestVideoFrameCallback` API this app uses. */
export interface VideoFrameCallbackHost {
  requestVideoFrameCallback: (
    callback: (
      now: DOMHighResTimeStamp,
      metadata: { mediaTime: number },
    ) => void,
  ) => number;
  cancelVideoFrameCallback: (handle: number) => void;
}
