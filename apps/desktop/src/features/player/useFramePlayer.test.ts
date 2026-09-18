/**
 * The player hook.
 *
 * Driven against a stub video element rather than a real one. jsdom has no
 * decoder and no `requestVideoFrameCallback`, so a real element could not
 * report a painted frame — and the painted frame is the entire subject. The
 * stub lets a test say "the browser painted frame 2 when you asked for frame 3"
 * and check that the player says so too, which is the behaviour that separates
 * this design from one that prints the number it requested.
 *
 * `ref` being a parameter is what makes this possible at all.
 */

import type { SeekIndex } from "@gsa/types";
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useFramePlayer } from "./useFramePlayer";

function index(): SeekIndex {
  return {
    schema_version: 1,
    path: "/data/swing.mov",
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest: "a".repeat(64),
      size_bytes: 1,
    },
    source: "decoded_frames",
    frame_count: 6,
    timestamps_s: [0, 0.0333, 0.0666, 0.0999, 0.1666, 0.2333],
    seek_targets_s: [0.0166, 0.0499, 0.0832, 0.1332, 0.1999, 0.2666],
    last_interval_s: 0.0666,
    warnings: [],
  };
}

/**
 * A video element that can be told what it painted.
 *
 * `paint` invokes the registered frame callback with a presentation time, which
 * is how the browser reports the frame it actually displayed. Nothing else in
 * the app can produce that number, and nothing else can check a seek.
 */
function stubVideo(options: { observable?: boolean } = {}) {
  let callback:
    ((now: number, metadata: { mediaTime: number }) => void) | null = null;

  const video = {
    currentTime: 0,
    playbackRate: 1,
    paused: true,
    play: vi.fn(() => Promise.resolve()),
    pause: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    ...(options.observable === false
      ? {}
      : {
          requestVideoFrameCallback: (
            handler: (now: number, metadata: { mediaTime: number }) => void,
          ) => {
            callback = handler;
            return 1;
          },
          cancelVideoFrameCallback: vi.fn(),
        }),
  };

  return {
    ref: { current: video as unknown as HTMLVideoElement },
    video,
    paint(mediaTime: number) {
      callback?.(0, { mediaTime });
    },
  };
}

describe("useFramePlayer", () => {
  it("seeks to the midpoint of the frame's display interval", () => {
    // Not to the frame's own timestamp, which is a boundary a decoder's
    // rounding decides the side of.
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.seekTo(3);
    });

    expect(stub.video.currentTime).toBe(0.1332);
  });

  it("clamps a seek past the end of the clip", () => {
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.seekTo(99);
    });

    expect(stub.video.currentTime).toBe(0.2666);
  });

  it("reports a residual of zero when the browser painted the frame asked for", () => {
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.seekTo(3);
    });
    act(() => {
      stub.paint(0.1332);
    });

    expect(result.current.residual).toEqual({
      requested: 3,
      landed: 3,
      delta: 0,
    });
  });

  it("reports the miss when the browser painted a different frame", () => {
    // The behaviour the whole design exists for. A player that displayed the
    // requested number would look correct here.
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.seekTo(3);
    });
    act(() => {
      stub.paint(0.0666);
    });

    expect(result.current.residual).toEqual({
      requested: 3,
      landed: 2,
      delta: -1,
    });
  });

  it("steps from the frame on screen, not from the one last asked for", () => {
    // Otherwise "next frame" on a player that is one frame behind repeats the
    // same miss forever and never advances.
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.seekTo(3);
    });
    act(() => {
      stub.paint(0.0666); // landed on 2
    });
    act(() => {
      result.current.step(1);
    });

    // 2 + 1, not 3 + 1.
    expect(stub.video.currentTime).toBe(0.1332);
  });

  it("says when the browser cannot report which frame it painted", () => {
    // Without `requestVideoFrameCallback` the seeks still happen and nothing can
    // confirm them. A residual of zero here would be a number never measured.
    const stub = stubVideo({ observable: false });
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    expect(result.current.observable).toBe(false);
    expect(result.current.residual).toBeNull();
  });

  it("follows the element while it is playing", () => {
    // During playback the element decides where we are, so the request follows
    // the playhead instead of accumulating a difference that measures nothing.
    const stub = stubVideo();
    stub.video.paused = false;
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      stub.paint(0.1666);
    });

    expect(result.current.frame).toBe(4);
    expect(result.current.residual?.delta).toBe(0);
  });

  it("applies a playback rate to the element", () => {
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, index()));

    act(() => {
      result.current.setRate(0.25);
    });

    expect(stub.video.playbackRate).toBe(0.25);
    expect(result.current.rate).toBe(0.25);
  });

  it("does nothing without an index rather than seeking to zero", () => {
    // A clip whose frame times have not arrived yet must not be scrubbed to the
    // start by a stray interaction.
    const stub = stubVideo();
    const { result } = renderHook(() => useFramePlayer(stub.ref, null));

    act(() => {
      result.current.seekTo(3);
    });

    expect(stub.video.currentTime).toBe(0);
  });
});
