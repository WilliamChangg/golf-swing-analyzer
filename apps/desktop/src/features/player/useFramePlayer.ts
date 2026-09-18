/**
 * A video element driven by frame index rather than by time.
 *
 * Drives one `<video>` and presents it as something the rest of the app can talk
 * to in the only unit its panels use: frames. Every seek goes through the
 * engine's measured `SeekIndex`, and every seek is **checked** — the frame the
 * browser reports painting is looked back up, and the difference from the frame
 * that was asked for is kept as `residual` and shown.
 *
 * That check is the point. A player that seeks to `frame / fps` and displays
 * the frame number it asked for looks correct on every clip, including the ones
 * where it is five frames out. This one can be wrong and say so.
 *
 * Playback is left to the element. Stepping through frames by hand with
 * `requestVideoFrameCallback` would reimplement a decoder's pacing in
 * JavaScript and lose audio sync, dropped-frame handling and the browser's own
 * rate control; the element plays, and the hook follows along by observing
 * which frame each painted frame was.
 */

import type { SeekIndex } from "@gsa/types";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  canObserveFrames,
  clampFrame,
  frameAt,
  residualFor,
  seekTargetFor,
  type SeekResidual,
  type VideoFrameCallbackHost,
} from "@/features/player/frames";

/**
 * Playback rates offered.
 *
 * Slower than real time only. A golf swing's downswing is about a quarter of a
 * second, so the useful direction is entirely downward, and offering 2x would
 * be offering a way to see less of the thing being examined.
 */
export const PLAYBACK_RATES = [0.25, 0.5, 1] as const;

export type PlaybackRate = (typeof PLAYBACK_RATES)[number];

export interface FramePlayer {
  /** The frame the UI last asked for. */
  frame: number;
  /** What the browser reported painting, and how far that was from the request. */
  residual: SeekResidual | null;
  /**
   * Whether this browser can report the frame it painted. False means seeks are
   * issued but unverified, and the UI says so rather than showing a zero it did
   * not measure.
   */
  observable: boolean;
  playing: boolean;
  rate: PlaybackRate;
  seekTo: (frame: number) => void;
  step: (delta: number) => void;
  togglePlay: () => void;
  setRate: (rate: PlaybackRate) => void;
}

/**
 * `ref` is passed in rather than created here, and attached to the `<video>` by
 * the caller. A hook that returned a ref inside its result object would make
 * that object ref-like, so every component receiving it — the transport, the
 * panels that seek on a click — would be reading a ref during render.
 */
export function useFramePlayer(
  ref: React.RefObject<HTMLVideoElement | null>,
  index: SeekIndex | null,
): FramePlayer {
  const [frame, setFrame] = useState(0);
  const [residual, setResidual] = useState<SeekResidual | null>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRateState] = useState<PlaybackRate>(1);
  const [observable, setObservable] = useState(false);

  // The frame last asked for, held in a ref because the frame callback reads it
  // on every painted frame and re-registering that callback on every seek would
  // tear down and rebuild the observer sixty times a second.
  const requested = useRef(0);

  const seekTo = useCallback(
    (target: number) => {
      const video = ref.current;
      if (!index || !video) return;

      const clamped = clampFrame(index, target);
      requested.current = clamped;
      setFrame(clamped);
      video.currentTime = seekTargetFor(index, clamped);
    },
    [index, ref],
  );

  const step = useCallback(
    (delta: number) => {
      // Stepping is relative to the frame that is **on screen**, not to the one
      // last asked for. Those differ exactly when a seek landed somewhere else,
      // and stepping from the request would then repeat the same miss — press
      // "next frame" twice on a player that is one frame behind and it stays
      // one frame behind forever.
      const from = residual?.landed ?? requested.current;
      seekTo(from + delta);
    },
    [residual, seekTo],
  );

  const togglePlay = useCallback(() => {
    const video = ref.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, [ref]);

  const setRate = useCallback(
    (next: PlaybackRate) => {
      setRateState(next);
      if (ref.current) ref.current.playbackRate = next;
    },
    [ref],
  );

  // Keep the element's rate in step with the state across remounts and source
  // changes, which reset `playbackRate` to 1 without telling anyone.
  useEffect(() => {
    if (ref.current) ref.current.playbackRate = rate;
  }, [rate, index, ref]);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;

    const onPlay = () => {
      setPlaying(true);
    };
    const onPause = () => {
      setPlaying(false);
    };
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
    };
  }, [index, ref]);

  // The observer. Re-registers itself after every painted frame, which is how
  // `requestVideoFrameCallback` is meant to be used — it is one-shot, like
  // `requestAnimationFrame`.
  useEffect(() => {
    const video = ref.current;
    if (!video || !index) return;

    const supported = canObserveFrames(video);
    setObservable(supported);
    if (!supported) {
      setResidual(null);
      return;
    }

    const host = video as unknown as VideoFrameCallbackHost;
    let handle = 0;
    let live = true;

    const onFrame = (
      _now: DOMHighResTimeStamp,
      metadata: { mediaTime: number },
    ) => {
      if (!live) return;
      const landed = frameAt(index.timestamps_s, metadata.mediaTime);
      // While playing, the element is the authority on where we are and the
      // "request" follows it, so the residual is zero by construction and means
      // nothing. It is only a measurement across a seek, which is why the
      // request is re-pointed at the playhead here rather than left to
      // accumulate a difference that measures nothing.
      if (!video.paused) {
        requested.current = landed;
        setFrame(landed);
      }
      setResidual(residualFor(index, requested.current, metadata.mediaTime));
      handle = host.requestVideoFrameCallback(onFrame);
    };

    handle = host.requestVideoFrameCallback(onFrame);
    return () => {
      live = false;
      host.cancelVideoFrameCallback(handle);
    };
  }, [index, ref]);

  return {
    frame,
    residual,
    observable,
    playing,
    rate,
    seekTo,
    step,
    togglePlay,
    setRate,
  };
}
