import type { SeekIndex } from "@gsa/types";
import { describe, expect, it } from "vitest";

import { clampFrame, frameAt, residualFor, seekTargetFor } from "./frames";

/**
 * A clip whose frames are *not* evenly spaced.
 *
 * Deliberately variable, and deliberately asymmetric, because on a
 * constant-rate clip every off-by-one and every dropped aspect of the lookup
 * produces an answer that is still nearly right. The intervals below are
 * 33.3 ms for four frames and then 66.7 ms, which is what a phone does when it
 * runs short of light.
 */
function index(overrides: Partial<SeekIndex> = {}): SeekIndex {
  const timestamps = [0, 0.0333, 0.0666, 0.0999, 0.1666, 0.2333];
  const targets = [0.0166, 0.0499, 0.0832, 0.1332, 0.1999, 0.2666];
  return {
    schema_version: 1,
    path: "/data/swing.mov",
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest: "a".repeat(64),
      size_bytes: 1,
    },
    source: "decoded_frames",
    frame_count: timestamps.length,
    timestamps_s: timestamps,
    seek_targets_s: targets,
    last_interval_s: 0.0666,
    warnings: [],
    ...overrides,
  };
}

describe("frameAt", () => {
  it("returns the last frame whose presentation time has been reached", () => {
    const times = index().timestamps_s;

    expect(frameAt(times, 0.05)).toBe(1);
    expect(frameAt(times, 0.12)).toBe(3);
    expect(frameAt(times, 0.2)).toBe(4);
  });

  it("treats a time exactly on a presentation stamp as that frame", () => {
    // The boundary the seek targets exist to avoid asking a decoder about. The
    // rule still has to be defined here, because a `mediaTime` coming back from
    // the browser can land on one.
    const times = index().timestamps_s;

    expect(frameAt(times, 0.0333)).toBe(1);
    expect(frameAt(times, 0.1666)).toBe(4);
  });

  it("clamps before the first frame rather than returning -1", () => {
    // Seeking before the start shows the first frame, which is what a player
    // does. An index of -1 would have to be guarded by every caller.
    expect(frameAt(index().timestamps_s, -5)).toBe(0);
  });

  it("holds on the last frame past the end of the clip", () => {
    expect(frameAt(index().timestamps_s, 99)).toBe(5);
  });

  it("answers for an empty clip instead of throwing", () => {
    expect(frameAt([], 1.5)).toBe(0);
  });

  it("agrees with a linear scan on every frame", () => {
    // The binary search exists because this runs on every painted frame, up to
    // 240 times a second. This pins it against the obvious implementation it
    // replaced.
    const times = index().timestamps_s;
    const linear = (t: number) => {
      let found = 0;
      times.forEach((stamp, i) => {
        if (stamp <= t) found = i;
      });
      return found;
    };

    for (let t = -0.1; t < 0.35; t += 0.0031) {
      expect(frameAt(times, t)).toBe(linear(t));
    }
  });
});

describe("seekTargetFor", () => {
  it("returns the midpoint the engine computed, not the frame's own time", () => {
    // The distinction this whole contract exists for. Returning the frame's own
    // timestamp would be seeking to a boundary and letting a decoder's rounding
    // decide which side of it we land on.
    const map = index();

    expect(seekTargetFor(map, 2)).toBe(0.0832);
    expect(seekTargetFor(map, 2)).not.toBe(map.timestamps_s[2]);
  });

  it("clamps rather than returning undefined past the end", () => {
    const map = index();

    expect(seekTargetFor(map, 99)).toBe(0.2666);
    expect(seekTargetFor(map, -4)).toBe(0.0166);
  });

  it("every target displays its own frame", () => {
    // The round trip, over every frame. This is the property the whole player
    // rests on, asserted here in the units the UI works in.
    const map = index();

    map.seek_targets_s.forEach((target, frame) => {
      expect(frameAt(map.timestamps_s, target)).toBe(frame);
    });
  });
});

describe("clampFrame", () => {
  it("keeps a frame inside the clip", () => {
    const map = index();

    expect(clampFrame(map, -3)).toBe(0);
    expect(clampFrame(map, 99)).toBe(5);
    expect(clampFrame(map, 3)).toBe(3);
  });

  it("rounds a fractional frame rather than truncating toward zero", () => {
    expect(clampFrame(index(), 2.6)).toBe(3);
  });

  it("survives a non-finite input", () => {
    // A range input with an empty value yields NaN, which would otherwise be
    // passed on to `currentTime` and throw.
    expect(clampFrame(index(), Number.NaN)).toBe(0);
  });
});

describe("residualFor", () => {
  it("is zero when the player showed the frame asked for", () => {
    const map = index();

    expect(residualFor(map, 3, 0.1332)).toEqual({
      requested: 3,
      landed: 3,
      delta: 0,
    });
  });

  it("reports the direction and size of a miss", () => {
    // What the transport displays. A player that landed elsewhere has to be
    // able to say so, which is the difference between this design and one that
    // prints the frame it requested.
    const map = index();

    expect(residualFor(map, 3, 0.0666)).toEqual({
      requested: 3,
      landed: 2,
      delta: -1,
    });
    expect(residualFor(map, 1, 0.2)).toEqual({
      requested: 1,
      landed: 4,
      delta: 3,
    });
  });
});
