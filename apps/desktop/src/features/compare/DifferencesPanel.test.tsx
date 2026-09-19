/**
 * What the panel says, and the things it must never say.
 *
 * The negative assertions are the point. A comparison screen is the most natural
 * place in this whole application for a score to appear — two swings, one
 * number, everybody understands it — and there is no measurement behind one. So
 * the absence is pinned here as well as in the Python contract, because a
 * component could perfectly well invent one out of fields that are individually
 * honest.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DifferencesPanel } from "@/features/compare/DifferencesPanel";
import { comparisonFixture } from "@/features/compare/comparison.fixture";

describe("DifferencesPanel", () => {
  it("shows a difference with its direction and both values", () => {
    render(<DifferencesPanel result={comparisonFixture()} />);

    expect(screen.getByText("Tempo ratio")).toBeInTheDocument();
    expect(screen.getByText("3.43")).toBeInTheDocument();
    expect(screen.getByText("2.61")).toBeInTheDocument();
    expect(screen.getByText(/the target measures lower/)).toBeInTheDocument();
  });

  it("itemises the bracket rather than summing it into one number", () => {
    // A clock term is fixed by a faster camera and a camera term by moving the
    // tripod back. One total tells a reader neither.
    render(<DifferencesPanel result={comparisonFixture()} />);

    expect(screen.getByText("reference clip")).toBeInTheDocument();
    expect(screen.getByText("target clip")).toBeInTheDocument();
    expect(screen.getByText("Had to clear")).toBeInTheDocument();
  });

  it("gives every refusal words rather than an enum name", () => {
    render(<DifferencesPanel result={comparisonFixture()} />);

    const refusals = screen.getAllByTestId("refusal");
    expect(refusals).toHaveLength(2);
    for (const entry of refusals) {
      expect(entry.textContent).not.toMatch(/camera_moved|unresolved/);
    }
  });

  it("shows the numbers behind an unresolved refusal", () => {
    // What is refused is the claim that the swings differ, not the values. A
    // reader who cannot see them cannot see how close the call was.
    render(<DifferencesPanel result={comparisonFixture()} />);

    expect(screen.getByText(/0\.80 vs 0\.81/)).toBeInTheDocument();
    expect(screen.getByText(/bracket 0\.04/)).toBeInTheDocument();
  });

  it("says plainly that an unresolved pair is not an agreement", () => {
    render(<DifferencesPanel result={comparisonFixture()} />);

    expect(
      screen.getByText(/not a finding that the swings agree/),
    ).toBeInTheDocument();
  });

  it("explains a moved camera before the refusals it caused", () => {
    render(<DifferencesPanel result={comparisonFixture()} />);

    const note = screen.getByTestId("camera-note");
    expect(note.textContent).toMatch(
      /do not agree about where the camera stood/,
    );
    expect(note.textContent).toMatch(/0\.83 against 0\.60/);
    expect(note.textContent).toMatch(/26\.3° apart/);
  });

  it("says the cameras agree when they do", () => {
    const result = comparisonFixture();
    result.camera = {
      ...result.camera,
      consistent: true,
      span_disagreement: 0.01,
    };

    render(<DifferencesPanel result={result} />);

    expect(screen.getByTestId("camera-note").textContent).toMatch(
      /close enough together/,
    );
  });

  it("reads as a result rather than a failure when nothing differs", () => {
    const result = comparisonFixture({ differences: [] });

    render(<DifferencesPanel result={result} />);

    expect(screen.queryByTestId("difference")).not.toBeInTheDocument();
    expect(
      screen.getByText(/Nothing measured in both clips differs by more than/),
    ).toBeInTheDocument();
  });

  it("puts no verdict anywhere on screen, and says so in as many words", () => {
    const { container } = render(
      <DifferencesPanel result={comparisonFixture()} />,
    );

    const text = (container.textContent ?? "").toLowerCase();
    for (const word of ["better", "worse", "grade", "rating", "improved"]) {
      expect(text).not.toContain(word);
    }
    // "score" appears exactly once, in the sentence denying there is one. A
    // panel that simply never mentioned it would leave a reader assuming the
    // absence was an oversight.
    expect(text.match(/score/g)).toHaveLength(1);
    expect(text).toContain("there is no score");
  });

  it("cites the frames on both sides of a difference", () => {
    render(<DifferencesPanel result={comparisonFixture()} />);

    expect(screen.getByText("30–168")).toBeInTheDocument();
    expect(screen.getByText("28–160")).toBeInTheDocument();
  });
});
