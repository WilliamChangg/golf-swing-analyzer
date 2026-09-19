/**
 * The two drawing rules, asserted on the strings they produce.
 *
 * Both are about the same thing from different directions: a plot must not put a
 * curve where nothing was measured, and must not leave a reader to guess how
 * much of a gap is the clock rather than the swing.
 */

import type { ChannelSample } from "@gsa/types";
import { describe, expect, it } from "vitest";

import {
  bandOf,
  pathOf,
  scaleFor,
  type PlotBox,
} from "@/features/compare/paths";
import { HOLE, channel } from "@/features/compare/comparison.fixture";

const BOX: PlotBox = {
  width: 100,
  height: 100,
  pad: { top: 0, right: 0, bottom: 0, left: 0 },
};

function sample(overrides: Partial<ChannelSample>): ChannelSample {
  return {
    position: 0,
    reference: null,
    target: null,
    difference: null,
    bracket: null,
    resolved: false,
    ...overrides,
  };
}

describe("scaleFor", () => {
  it("returns null when nothing is drawable", () => {
    expect(scaleFor([sample({}), sample({ position: 3 })], BOX)).toBeNull();
  });

  it("puts the takeaway at the left edge and the finish at the right", () => {
    const scale = scaleFor([sample({ position: 0, reference: 1 })], BOX);

    expect(scale?.x(0)).toBe(0);
    expect(scale?.x(3)).toBe(100);
  });

  it("covers the band and not just the two lines", () => {
    // A bracket drawn past the top of the plot is a bracket the reader silently
    // under-reads, which inverts the point of drawing one.
    const scale = scaleFor(
      [
        sample({ position: 0, reference: 1, bracket: 5 }),
        sample({ position: 3, reference: 1, bracket: 5 }),
      ],
      BOX,
    );

    expect(scale?.y(6)).toBeCloseTo(0);
    expect(scale?.y(-4)).toBeCloseTo(100);
  });

  it("reports the value range it covers, so the plot can label it", () => {
    // Without labels a reader sees the shape of a difference and not its size,
    // which on a comparison is most of what they came for.
    const scale = scaleFor(
      [
        sample({ position: 0, reference: 1, bracket: 0.5 }),
        sample({ position: 3, reference: 4 }),
      ],
      BOX,
    );

    expect(scale?.domain).toEqual([0.5, 4]);
  });

  it("gives a flat series an axis rather than dividing by zero", () => {
    const scale = scaleFor(
      [
        sample({ position: 0, reference: 2 }),
        sample({ position: 3, reference: 2 }),
      ],
      BOX,
    );

    expect(scale?.y(2)).toBeCloseTo(50);
    expect(Number.isFinite(scale?.y(2) ?? NaN)).toBe(true);
  });
});

describe("pathOf", () => {
  it("breaks the path where the series is null rather than bridging it", () => {
    const scale = scaleFor(channel().samples ?? [], BOX);
    expect(scale).not.toBeNull();

    const drawn = pathOf(
      channel().samples ?? [],
      (entry) => entry.reference,
      scale!,
    );

    // Two subpaths: one before the hole and one after. A bridged gap would be one.
    expect(drawn.match(/M/g)).toHaveLength(2);
  });

  it("draws nothing at all for a series with no values", () => {
    const scale = scaleFor([sample({ position: 0, reference: 1 })], BOX);

    expect(
      pathOf([sample({}), sample({})], (entry) => entry.target, scale!),
    ).toBe("");
  });

  it("starts a fresh subpath after every gap, however many there are", () => {
    const scale = scaleFor([sample({ position: 0, reference: 1 })], BOX);
    const series = [
      sample({ position: 0, reference: 1 }),
      sample({}),
      sample({ position: 1, reference: 1 }),
      sample({}),
      sample({ position: 2, reference: 1 }),
    ];

    expect(
      pathOf(series, (entry) => entry.reference, scale!).match(/M/g),
    ).toHaveLength(3);
  });
});

describe("bandOf", () => {
  it("closes one area per unbroken run", () => {
    const samples = channel().samples ?? [];
    const scale = scaleFor(samples, BOX);

    const area = bandOf(samples, scale!);

    expect(area.match(/Z/g)).toHaveLength(2);
    expect(samples[HOLE.from]?.reference).toBeNull();
  });

  it("draws nothing where a sample has a value but no bracket", () => {
    // An unbracketed sample is one whose window held no usable neighbour. Shading
    // it at zero width would claim the value is exactly determined there, which
    // is the opposite of what the null means.
    const scale = scaleFor([sample({ position: 0, reference: 1 })], BOX);
    const series = [
      sample({ position: 0, reference: 1 }),
      sample({ position: 1, reference: 1 }),
    ];

    expect(bandOf(series, scale!)).toBe("");
  });

  it("drops a run of one, which cannot enclose anything", () => {
    const scale = scaleFor([sample({ position: 0, reference: 1 })], BOX);
    const series = [
      sample({ position: 0, reference: 1, bracket: 0.1 }),
      sample({}),
      sample({ position: 2, reference: 1, bracket: 0.1 }),
    ];

    expect(bandOf(series, scale!)).toBe("");
  });
});
