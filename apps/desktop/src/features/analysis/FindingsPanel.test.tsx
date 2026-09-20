/**
 * The findings panel, tested against the shape its own data actually has.
 *
 * Across four reference clips this engine produces at most two findings and
 * refuses ten or eleven, and on the 30 fps amateur clip it produces none. So the
 * cases worth pinning are the empty one and the refusals — a panel that only
 * looked right with findings in it would look broken on most real input.
 */

import type { CoachingReport, Finding, RefusedFinding } from "@gsa/types";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FindingsPanel } from "./FindingsPanel";

/** The one finding the tour face-on clip produces, at a slow-motion factor of 7. */
function tempoFinding(overrides: Partial<Finding> = {}): Finding {
  return {
    rule_id: "tempo.ratio",
    title: "Backswing-to-downswing tempo",
    observation:
      "The backswing took 1.83 times as long as the downswing, below the 2.43 to 3.80 this source reports for tour professionals.",
    comparison: "below",
    value: 1.83,
    unit: "ratio",
    bracket: 0.04,
    band_low: 2.43,
    band_high: 3.8,
    margin: 0.6,
    source: {
      citation: "Novosel, Tour Tempo (2004)",
      year: 2004,
      population: "tour professionals",
      sample_size: 54,
      method: "two_d_video",
      measures: "Backswing and downswing frame counts at 30 fps.",
      permitted_bases: ["temporal"],
      filmed_from: null,
      note: "Published as frame counts, not as a ratio.",
    },
    evidence: [
      {
        metric: "backswing_duration",
        label: "Backswing duration",
        value: 0.8,
        unit: "seconds",
        basis: "temporal",
        event: null,
        phase: "backswing",
        uncertainty: null,
        confidence: 0.9,
        frames: [13, 14, 15, 38],
        timestamps_s: [0.433, 0.466, 0.5, 1.267],
        methodology: "Takeaway to top, on the clip's own clock.",
      },
    ],
    confidence: 0.82,
    phrased: null,
    ...overrides,
  };
}

function refused(overrides: Partial<RefusedFinding> = {}): RefusedFinding {
  return {
    rule_id: "rotation.x_factor_top",
    title: "X-factor at the top",
    refusal: "basis_not_permitted",
    reason:
      "The published figure is a threshold on a spatial measurement; this clip measures a foreshortened angle.",
    source: {
      citation: "Golf Magazine (1992)",
      year: 1992,
      population: "unstated",
      sample_size: null,
      method: "convention",
      measures: "Difference between shoulder and pelvis turn at the top.",
      permitted_bases: [],
      filmed_from: null,
      note: "No measurement protocol is published.",
    },
    ...overrides,
  };
}

function report(overrides: Partial<CoachingReport> = {}): CoachingReport {
  return {
    schema_version: 1,
    computed: true,
    findings: [],
    refused: [],
    rules_considered: 12,
    view: "face_on",
    frame_interval_s: 0.0333,
    phrasing: { mode: "off", attempted: 0, accepted: 0, rejected: 0 },
    warnings: [],
    ...overrides,
  };
}

describe("FindingsPanel", () => {
  it("presents an empty result as a result, not as a failure", () => {
    render(
      <FindingsPanel
        report={report({ refused: [refused()] })}
        onSeek={vi.fn()}
      />,
    );

    expect(screen.getByText(/That is a result, not a failure/)).toBeVisible();
  });

  it("lists every refusal with the reason it stopped", () => {
    // The refusals are where nearly all the information is on real footage.
    // Dropping them would leave a reader unable to tell a number that is absent
    // on purpose from one that is absent by accident.
    render(
      <FindingsPanel
        report={report({
          refused: [
            refused(),
            refused({
              rule_id: "posture.spine_tilt_impact",
              refusal: "no_uncertainty",
            }),
          ],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(screen.getByText("Not concluded (2)")).toBeVisible();
    expect(
      screen.getByText(/measured on a different kind of quantity/),
    ).toBeVisible();
    expect(screen.getByText(/never been quantified/)).toBeVisible();
  });

  it("shows the bracket a comparison had to clear", () => {
    // The number that explains why this panel is empty on a 30 fps clip: one
    // frame at the top is worth 0.74 of a band 1.37 wide.
    render(
      <FindingsPanel
        report={report({ findings: [tempoFinding()] })}
        onSeek={vi.fn()}
      />,
    );

    expect(screen.getByText("Bracket it cleared")).toBeVisible();
    expect(screen.getByText("0.04")).toBeVisible();
  });

  it("seeks to the frames a piece of evidence cites", async () => {
    // The whole argument for citing frames: a finding that carries them is one
    // somebody can discover is wrong.
    const onSeek = vi.fn();
    render(
      <FindingsPanel
        report={report({ findings: [tempoFinding()] })}
        onSeek={onSeek}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: /frames 13–38/ }));

    expect(onSeek).toHaveBeenCalledWith(13);
  });

  it("names the source, its population and how it was measured", () => {
    render(
      <FindingsPanel
        report={report({ findings: [tempoFinding()] })}
        onSeek={vi.fn()}
      />,
    );

    expect(
      screen.getByText(
        /Novosel, Tour Tempo \(2004\).*tour professionals.*n = 54/,
      ),
    ).toBeVisible();
    expect(screen.getByText(/Measured by two d video/)).toBeVisible();
  });

  it("keeps the engine's own sentence when a rewording is present", () => {
    // `phrased` never replaces `observation`. The guard can tell a rewording
    // invents no numbers; it cannot tell the rewording is about this finding
    // rather than the one below it.
    render(
      <FindingsPanel
        report={report({
          findings: [
            tempoFinding({
              phrased: "Your backswing is quick relative to your downswing.",
            }),
          ],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(
      screen.getByText(/below the 2.43 to 3.80 this source reports/),
    ).toBeVisible();
    expect(
      screen.getByText("Your backswing is quick relative to your downswing."),
    ).toBeVisible();
  });

  it("presents no score, severity or grade as a field", () => {
    // Phase 13 asserts the absence of those field names in the contract. This is
    // the same assertion one layer up, where a presentation could reintroduce
    // one by deriving it from fields that do exist -- `margin` and `confidence`
    // are both a short step from a five-star rating.
    //
    // Asserted over the labels rather than over all the text, because the panel
    // *says* there is no score and that sentence is the claim, not a violation
    // of it.
    const { container } = render(
      <FindingsPanel
        report={report({ findings: [tempoFinding()], refused: [refused()] })}
        onSeek={vi.fn()}
      />,
    );

    const labels = [...container.querySelectorAll("dt, th, h4, h5")].map(
      (node) => node.textContent ?? "",
    );

    expect(labels.length).toBeGreaterThan(0);
    for (const label of labels) {
      expect(label).not.toMatch(/severity|score|grade|rating|rank/i);
    }
  });

  it("surfaces the report's warnings", () => {
    render(
      <FindingsPanel
        report={report({
          warnings: [
            "This clip's slow-motion factor was supplied, not measured.",
          ],
        })}
        onSeek={vi.fn()}
      />,
    );

    expect(
      screen.getByText(/slow-motion factor was supplied, not measured/),
    ).toBeVisible();
  });
});
