/**
 * A small hand-built comparison, for the tests that are about presentation.
 *
 * Deliberately **not** a real one. `python/tests/test_comparison_*.py` check the
 * arithmetic against swings whose relationship is known by construction; what is
 * left for these tests is what the app does with the answer — whether a gap in a
 * series stays a gap, whether a refusal arrives as words rather than an enum
 * name, and whether anything on screen ranks the two swings.
 *
 * Three things about it are load-bearing:
 *
 * * **One channel has a hole in the middle.** A plot that joined its endpoints
 *   would draw a curve at a position where one clip was never measured, and
 *   nothing on screen would say so.
 * * **One difference resolves and one does not**, so the two presentations are
 *   both exercised — and the unresolved one carries its numbers, because what is
 *   refused is the claim that the swings differ, not the values.
 * * **The two clips disagree about the camera**, which is the state that refuses
 *   most of a real report and therefore the one the panel has to explain well.
 *
 * It lives beside the tests rather than inside one so the plot and the panel can
 * assert against the same report. Nothing in the app imports it.
 */

import type {
  ChannelSample,
  ClipSummary,
  SwingComparison,
  TrajectoryComparison,
} from "@gsa/types";

export const SAMPLE_COUNT = 13;

/** Where the hole sits, as indices into a `SAMPLE_COUNT`-long series. */
export const HOLE = { from: 5, to: 8 };

function clip(name: string, digest: string, span: number): ClipSummary {
  return {
    path: `/clips/${name}`,
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest,
      size_bytes: 1024,
    },
    frames: 300,
    view: "face_on",
    lead_side: "left",
    calibration: "none",
    slow_motion_factor: 1,
    torso_length: span,
    clock: {
      usable: true,
      frame_interval_s: 1 / 120,
      slow_motion_factor: 1,
      methodology: "Piecewise-linear between the four detected events.",
      knots: [
        {
          event: "takeaway",
          position: 0,
          frame_index: 30,
          timestamp_s: 0.25,
          confidence: 0.9,
          ambiguity_s: 1 / 120,
          ambiguity_source: "One frame interval.",
        },
        {
          event: "top",
          position: 1,
          frame_index: 120,
          timestamp_s: 1.0,
          confidence: 0.8,
          ambiguity_s: 1 / 120,
          ambiguity_source: "One frame interval.",
        },
        {
          event: "impact",
          position: 2,
          frame_index: 168,
          timestamp_s: 1.4,
          confidence: 0.7,
          ambiguity_s: 0.04,
          ambiguity_source:
            "Two independent estimates disagree by more than one frame.",
        },
        {
          event: "finish",
          position: 3,
          frame_index: 240,
          timestamp_s: 2.0,
          confidence: 0.4,
          ambiguity_s: 1 / 120,
          ambiguity_source: "One frame interval.",
        },
      ],
    },
  };
}

function samples(options: {
  hole: boolean;
  resolvedFrom: number;
}): ChannelSample[] {
  return Array.from({ length: SAMPLE_COUNT }, (_, index) => {
    const position = (index / (SAMPLE_COUNT - 1)) * 3;
    const blocked = options.hole && index >= HOLE.from && index < HOLE.to;
    if (blocked) {
      return {
        position,
        reference: null,
        target: null,
        difference: null,
        bracket: null,
        resolved: false,
      };
    }
    const reference = Math.sin((position / 3) * Math.PI);
    const target = reference + (index >= options.resolvedFrom ? 0.5 : 0.01);
    const bracket = 0.05;
    const difference = target - reference;
    return {
      position,
      reference,
      target,
      difference,
      bracket,
      resolved: Math.abs(difference) > bracket,
    };
  });
}

export function channel(
  overrides: Partial<TrajectoryComparison> = {},
): TrajectoryComparison {
  const drawn = samples({ hole: true, resolvedFrom: 9 });
  return {
    channel: "hand_speed",
    label: "Hand speed",
    unit: "torso_lengths_per_s",
    basis: "image_plane",
    samples: drawn,
    resolved_fraction:
      drawn.filter((sample) => sample.resolved).length / drawn.length,
    largest_difference: 0.5,
    largest_at: 3,
    refusal: null,
    reason: "",
    ...overrides,
  };
}

export function comparisonFixture(
  overrides: Partial<SwingComparison> = {},
): SwingComparison {
  return {
    schema_version: 1,
    computed: true,
    reference: clip("before.mov", "a".repeat(64), 0.31),
    target: clip("after.mov", "b".repeat(64), 0.3),
    camera: {
      reference_span: 0.83,
      target_span: 0.6,
      span_disagreement: 0.32,
      reference_openness: 0.99,
      target_openness: 0.95,
      reference_azimuth_deg: 8.1,
      target_azimuth_deg: 18.2,
      azimuth_separation_deg: 26.3,
      consistent: false,
      methodology: "Projected shoulder span at address, in torso lengths.",
    },
    differences: [
      {
        name: "tempo_ratio",
        label: "Tempo ratio",
        group: "timing",
        unit: "ratio",
        basis: "temporal",
        event: null,
        reference_value: 3.43,
        target_value: 2.61,
        difference: -0.82,
        direction: "lower",
        bracket: 0.3,
        terms: [
          {
            source: "reference clip",
            value: 0.18,
            reason: "One frame of ambiguity at the top.",
          },
          {
            source: "target clip",
            value: 0.12,
            reason: "One frame of ambiguity at the top.",
          },
        ],
        margin: 0.52,
        confidence: 0.61,
        reference_frames: [30, 120, 168],
        target_frames: [28, 110, 160],
        interpretation:
          "The backswing divided by the downswing. A camera position cannot change it.",
        methodology: "Two event timings divided.",
      },
    ],
    refused: [
      {
        name: "shoulder_turn",
        label: "Shoulder turn at the top",
        event: "top",
        refusal: "camera_moved",
        reason:
          "The shoulders span 0.83 torso lengths at address in one and 0.60 in the other.",
        reference_value: null,
        target_value: null,
        bracket: null,
      },
      {
        name: "backswing_duration",
        label: "Backswing duration",
        event: null,
        refusal: "unresolved",
        reason:
          "The two values differ by less than these recordings can resolve.",
        reference_value: 0.8,
        target_value: 0.81,
        bracket: 0.04,
      },
    ],
    trajectories: [channel()],
    hand_path: null,
    metrics_considered: 3,
    config: {
      samples: SAMPLE_COUNT,
      min_confidence: 0.25,
      max_span_disagreement: 0.1,
    },
    warnings: [],
    ...overrides,
  };
}
