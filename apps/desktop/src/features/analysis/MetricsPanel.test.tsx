/**
 * The metrics panel.
 *
 * The tests worth having here are about what a number is shown *with*, not
 * about whether it renders. A projected angle and a three-dimensional one print
 * identically, and a refused metric and a missing one look the same from
 * outside — both of those are the panel's job to keep apart.
 */

import type { Metric, MetricSet, RefusedMetric } from "@gsa/types";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { MetricsPanel } from "./MetricsPanel";

function metric(overrides: Partial<Metric> = {}): Metric {
  return {
    name: "shoulder_turn",
    group: "rotation",
    label: "Shoulder turn (top)",
    value: 53.0,
    unit: "degrees",
    basis: "foreshortened_angle",
    event: "top",
    phase: null,
    source_frames: [356, 357, 358],
    view: "face_on",
    interpretation:
      "Turn away from the camera, inferred from how much the shoulder line shortened.",
    uncertainty: 5.5,
    confidence: {
      overall: 0.49,
      observation: 1.0,
      anchor: 0.64,
      method: 0.77,
    },
    methodology:
      "Arccosine of the projected shoulder span against the address baseline.",
    ...overrides,
  };
}

function refused(overrides: Partial<RefusedMetric> = {}): RefusedMetric {
  return {
    name: "x_factor",
    event: "top",
    reason:
      "A down-the-line camera does not contain this measurement: the shoulders project 0.10 torso lengths at address.",
    ...overrides,
  };
}

function metrics(overrides: Partial<MetricSet> = {}): MetricSet {
  return {
    schema_version: 1,
    computed: true,
    metrics: [metric()],
    refused: [],
    view: {
      view: "face_on",
      confidence: 1.0,
      shoulder_span_ratio: 0.83,
      hip_span_ratio: 0.6,
      openness: 0.9,
      frames: [0, 1, 2],
      methodology: "Projected shoulder width at address.",
    },
    lead_side: null,
    references: [],
    torso_length: 0.31,
    geometry: { width: 720, height: 1280 },
    frames: 68,
    calibration: "none",
    slow_motion_factor: 1,
    config: {},
    warnings: [],
    ...overrides,
  };
}

describe("MetricsPanel", () => {
  it("shows the measured uncertainty where one exists", () => {
    render(<MetricsPanel result={metrics()} onSeek={vi.fn()} />);

    expect(screen.getByText("53.0° ± 5.5°")).toBeVisible();
  });

  it("does not print an uncertainty where none was measured", () => {
    // Phase 6 quantifies uncertainty for the foreshortened rotations and for
    // nothing else. A "± 0" on the rest would claim a precision never computed.
    render(
      <MetricsPanel
        result={metrics({
          metrics: [
            metric({
              uncertainty: null,
              basis: "temporal",
              unit: "ratio",
              value: 2.67,
            }),
          ],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(screen.getByText("2.67 : 1")).toBeVisible();
    expect(screen.queryByText(/±/)).toBeNull();
  });

  it("explains what kind of claim the number is when asked", async () => {
    // `basis` is the field Phase 5 said does the most work. A reader has to be
    // able to find out that this angle is not an angle of the body.
    render(<MetricsPanel result={metrics()} onSeek={vi.fn()} />);

    await userEvent.click(
      screen.getByRole("button", {
        name: /How Shoulder turn \(top\) was measured/,
      }),
    );

    expect(screen.getByText(/one camera cannot see it directly/)).toBeVisible();
    expect(
      screen.getByText(/Arccosine of the projected shoulder span/),
    ).toBeVisible();
  });

  it("decomposes the confidence rather than showing only the product", async () => {
    render(<MetricsPanel result={metrics()} onSeek={vi.fn()} />);
    await userEvent.click(
      screen.getByRole("button", {
        name: /How Shoulder turn \(top\) was measured/,
      }),
    );

    expect(
      screen.getByText(/observation 1.00.*anchor 0.64.*method 0.77/),
    ).toBeVisible();
  });

  it("seeks to the frames a metric was measured over", async () => {
    const onSeek = vi.fn();
    render(<MetricsPanel result={metrics()} onSeek={onSeek} />);

    await userEvent.click(
      screen.getByRole("button", { name: "Shoulder turn (top)" }),
    );

    expect(onSeek).toHaveBeenCalledWith(356);
  });

  it("gives refusals their own list rather than dropping them", () => {
    // On a down-the-line clip Phase 5 refuses three metrics outright. Showing
    // the survivors and dropping those three would leave a reader unable to
    // discover that the number they came for was deliberately absent.
    render(
      <MetricsPanel
        result={metrics({
          refused: [refused(), refused({ name: "pelvis_turn" })],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(screen.getByText("Refused (2)")).toBeVisible();
    expect(screen.getByText("x_factor")).toBeVisible();
    expect(screen.getByText("pelvis_turn")).toBeVisible();
  });

  it("does not offer a jump for a metric that cites no frames", () => {
    render(
      <MetricsPanel
        result={metrics({ metrics: [metric({ source_frames: [] })] })}
        onSeek={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Shoulder turn (top)" }),
    ).toBeDisabled();
  });

  it("surfaces the measurement warnings", () => {
    render(
      <MetricsPanel
        result={metrics({
          warnings: ["This clip was measured without a camera calibration."],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(
      screen.getByText(/measured without a camera calibration/),
    ).toBeVisible();
  });
});
