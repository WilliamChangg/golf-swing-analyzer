/**
 * The screen where two clips become one report.
 *
 * What is worth pinning here is what the screen does *around* the engine call:
 * that both clips arrive through the Rust picker rather than by name, that
 * choosing one does not clear the other, that the shared window and the per-clip
 * factors reach the engine the way the contract expects, and that a comparison
 * which could not be computed reads as a result rather than a failure.
 *
 * The arithmetic is checked in Python, and the drawing rules in `paths.test.ts`.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";
import { comparisonFixture } from "@/features/compare/comparison.fixture";

const chooseClipMock = vi.hoisted(() => vi.fn());
const compareSwingsMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  chooseClip: chooseClipMock,
  compareSwings: compareSwingsMock,
  onProgress: () => Promise.resolve(() => undefined),
}));

const { CompareScreen } = await import("./CompareScreen");

async function pickBoth() {
  const user = userEvent.setup();
  chooseClipMock.mockResolvedValueOnce({
    ok: true,
    value: "/clips/before.mov",
  });
  await user.click(screen.getByTestId("choose-reference"));
  chooseClipMock.mockResolvedValueOnce({ ok: true, value: "/clips/after.mov" });
  await user.click(screen.getByTestId("choose-target"));
  return user;
}

describe("CompareScreen", () => {
  beforeEach(() => {
    chooseClipMock.mockReset();
    compareSwingsMock.mockReset();
    compareSwingsMock.mockResolvedValue({
      ok: true,
      value: comparisonFixture(),
    });
  });

  it("will not compare until both clips are chosen", async () => {
    const user = userEvent.setup();
    render(<CompareScreen />);

    expect(screen.getByRole("button", { name: /Compare/ })).toBeDisabled();

    chooseClipMock.mockResolvedValueOnce({
      ok: true,
      value: "/clips/before.mov",
    });
    await user.click(screen.getByTestId("choose-reference"));

    expect(screen.getByRole("button", { name: /Compare/ })).toBeDisabled();
  });

  it("keeps the first clip when the second is chosen", async () => {
    render(<CompareScreen />);
    await pickBoth();

    expect(screen.getByText("before.mov")).toBeInTheDocument();
    expect(screen.getByText("after.mov")).toBeInTheDocument();
  });

  it("treats a cancelled picker as neither a choice nor an error", async () => {
    const user = userEvent.setup();
    render(<CompareScreen />);

    chooseClipMock.mockResolvedValueOnce({ ok: true, value: null });
    await user.click(screen.getByTestId("choose-reference"));

    // Both sides still say so: cancelling chose nothing, and it is not an error.
    expect(screen.getAllByText("No clip chosen.")).toHaveLength(2);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("sends one window for the pair and a factor for each clip", async () => {
    render(<CompareScreen />);
    const user = await pickBoth();

    const factors = screen.getAllByRole("spinbutton");
    await user.clear(factors[0]!);
    await user.type(factors[0]!, "7");

    await user.click(screen.getByRole("button", { name: /Compare/ }));

    await waitFor(() => {
      expect(compareSwingsMock).toHaveBeenCalled();
    });
    const [reference, target, options] = compareSwingsMock.mock.calls[0] as [
      string,
      string,
      Record<string, unknown>,
    ];
    expect(reference).toBe("/clips/before.mov");
    expect(target).toBe("/clips/after.mov");
    expect(options.referenceSlowMotion).toBe(7);
    expect(options.targetSlowMotion).toBe(1);
  });

  it("shows the clocks, which is what the normalisation divided out", async () => {
    render(<CompareScreen />);
    const user = await pickBoth();
    await user.click(screen.getByRole("button", { name: /Compare/ }));

    await waitFor(() => {
      expect(screen.getByText("The clocks")).toBeInTheDocument();
    });
    // Every knot from both clips, so a reader can see the timing the plots
    // below have had removed from them. Scoped to the table, because the plot
    // axis names the same four instants and that is not what is being counted.
    const clocks = within(screen.getByTestId("clock-table"));
    expect(clocks.getAllByText("takeaway")).toHaveLength(2);
    expect(clocks.getAllByText("impact")).toHaveLength(2);
  });

  it("draws a channel that was comparable and explains one that was not", async () => {
    const result = comparisonFixture();
    result.trajectories = [
      ...(result.trajectories ?? []),
      {
        channel: "shoulder_angle",
        label: "Shoulder line tilt",
        unit: "degrees",
        basis: "projected_angle",
        samples: [],
        resolved_fraction: 0,
        largest_difference: null,
        largest_at: null,
        refusal: "camera_moved",
        reason: "These two recordings disagree about where the camera stood.",
      },
    ];
    compareSwingsMock.mockResolvedValue({ ok: true, value: result });

    render(<CompareScreen />);
    const user = await pickBoth();
    await user.click(screen.getByRole("button", { name: /Compare/ }));

    await waitFor(() => {
      expect(screen.getByTestId("plot-hand_speed")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("plot-shoulder_angle")).not.toBeInTheDocument();
    expect(
      screen.getByText(/disagree about where the camera stood/),
    ).toBeInTheDocument();
  });

  it("reads an uncomputable pair as a result rather than a failure", async () => {
    compareSwingsMock.mockResolvedValue({
      ok: true,
      value: comparisonFixture({
        computed: false,
        trajectories: [],
        differences: [],
        refused: [],
        warnings: ["/clips/after.mov: No swing was detected in this clip."],
      }),
    });

    render(<CompareScreen />);
    const user = await pickBoth();
    await user.click(screen.getByRole("button", { name: /Compare/ }));

    await waitFor(() => {
      expect(
        screen.getByText(/cannot be put on one clock/),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/No swing was detected/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("surfaces an engine failure as an error, not as an empty report", async () => {
    compareSwingsMock.mockResolvedValue({
      ok: false,
      error: { kind: "method", message: "No extracted poses for after.mov." },
    });

    render(<CompareScreen />);
    const user = await pickBoth();
    await user.click(screen.getByRole("button", { name: /Compare/ }));

    await waitFor(() => {
      expect(screen.getByText(/No extracted poses/)).toBeInTheDocument();
    });
  });

  it("clears a stale report when either clip is swapped", async () => {
    render(<CompareScreen />);
    const user = await pickBoth();
    await user.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => {
      expect(screen.getByText("The clocks")).toBeInTheDocument();
    });

    chooseClipMock.mockResolvedValueOnce({
      ok: true,
      value: "/clips/third.mov",
    });
    await user.click(screen.getByTestId("choose-target"));

    expect(screen.queryByText("The clocks")).not.toBeInTheDocument();
  });
});
