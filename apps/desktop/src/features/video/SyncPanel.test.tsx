import type { SyncModel } from "@gsa/types";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";

const syncClipsMock = vi.hoisted(() => vi.fn());
const openMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  syncClips: syncClipsMock,
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({ open: openMock }));

const { SyncPanel } = await import("./SyncPanel");

const REFERENCE = "/Users/example/data/face-on/swing.mov";
const TARGET = "/Users/example/data/dtl/swing.mov";

function clip(name: string, frames: number, interval: number) {
  return {
    path: `/Users/example/${name}`,
    name,
    frames,
    start_s: 0,
    duration_s: (frames - 1) * interval,
    median_interval_s: interval,
    slow_motion_factor: 1,
  };
}

/** An alignment shaped like a clean 120 fps pair: offset from the correlation. */
function aligned(overrides: Partial<SyncModel> = {}): SyncModel {
  return {
    schema_version: 1,
    aligned: true,
    method: "combined",
    reference: clip("face-on.mov", 312, 1 / 120),
    target: clip("dtl.mov", 300, 1 / 120),
    time_map: {
      offset_s: 0.4,
      rate: 1,
      rate_estimated: false,
      pivot_s: 1.4,
      offset_uncertainty_s: 0.0034,
      rate_uncertainty: null,
      support_start_s: 0.53,
      support_end_s: 2.09,
    },
    anchors: [
      {
        label: "takeaway",
        event: "takeaway",
        source: "detected",
        reference_frame: 64,
        target_frame: 112,
        reference_s: 0.533,
        target_s: 0.933,
        confidence: 0.95,
      },
      {
        label: "impact",
        event: "impact",
        source: "detected",
        reference_frame: 204,
        target_frame: 252,
        reference_s: 1.7,
        target_s: 2.1,
        confidence: 0.94,
      },
    ],
    residuals: [
      {
        label: "takeaway",
        reference_s: 0.533,
        observed_target_s: 0.933,
        predicted_target_s: 0.933,
        residual_ms: -0.4,
      },
      {
        label: "impact",
        reference_s: 1.7,
        observed_target_s: 2.1,
        predicted_target_s: 2.1008,
        residual_ms: -0.8,
      },
    ],
    quality: {
      residual_rms_ms: 0.6,
      residual_max_ms: 0.8,
      degrees_of_freedom: 1,
      quantisation_floor_ms: 3.4,
      method_disagreement_ms: 0.5,
    },
    confidence: {
      overall: 0.34,
      agreement: 0.36,
      anchors: 0.95,
      stability: 1,
    },
    correlation: {
      peak_correlation: 0.991,
      peak_offset_s: 0.4,
      rival_correlation: 0.63,
      rival_offset_s: 1.24,
      grid_interval_s: 1 / 120,
      overlap_s: 2.2,
      samples: 265,
      sub_grid_shift_s: 0.0004,
    },
    overlap: {
      start_s: 0.4,
      end_s: 2.59,
      duration_s: 2.19,
      reference_fraction: 0.85,
      target_fraction: 1,
    },
    config: {},
    refusal: null,
    warnings: [
      "Synchronisation aligns two swing-shaped signals; it cannot tell that both cameras filmed the same swing.",
    ],
    ...overrides,
  };
}

async function chooseTarget(): Promise<void> {
  openMock.mockResolvedValue(TARGET);
  await userEvent.click(screen.getByRole("button", { name: /second clip/i }));
  await waitFor(() => {
    expect(syncClipsMock).toHaveBeenCalled();
  });
}

beforeEach(() => {
  syncClipsMock.mockReset();
  openMock.mockReset();
  syncClipsMock.mockResolvedValue({ ok: true, value: aligned() });
});

describe("SyncPanel", () => {
  it("says what it is for before a second clip is chosen", () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    expect(screen.getByText(/Phases 8 and 9 build on/i)).toBeInTheDocument();
    expect(screen.queryByTestId("alignment-timeline")).not.toBeInTheDocument();
  });

  it("sends both paths and the per-clip slow-motion factors", async () => {
    render(<SyncPanel referencePath={REFERENCE} referenceSlowMotion={7} />);

    fireEvent.change(
      screen.getByLabelText(/slow-motion factor of the second clip/i),
      { target: { value: "5" } },
    );
    await chooseTarget();

    expect(syncClipsMock).toHaveBeenCalledWith(
      REFERENCE,
      TARGET,
      expect.objectContaining({
        referenceSlowMotion: 7,
        targetSlowMotion: 5,
      }),
    );
  });

  it("shows the offset with its uncertainty", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText(/\+400\.0 ms ± 3\.4 ms/)).toBeInTheDocument();
  });

  it("states that an unestimated clock rate was assumed rather than measured", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText(/assumed, not measured/i)).toBeInTheDocument();
  });

  it("shows a fitted clock rate as a value rather than as an assumption", async () => {
    syncClipsMock.mockResolvedValue({
      ok: true,
      value: aligned({
        time_map: {
          offset_s: -0.13,
          rate: 0.90901,
          rate_estimated: true,
          pivot_s: 1.4,
          offset_uncertainty_s: 0.0017,
          rate_uncertainty: 0.0029,
          support_start_s: 0.53,
          support_end_s: 2.09,
        },
      }),
    });
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText("0.90901")).toBeInTheDocument();
    expect(
      screen.queryByText(/assumed, not measured/i),
    ).not.toBeInTheDocument();
  });

  it("puts the residual next to the floor it should be judged against", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText(/0\.6 ms rms \(0\.2× floor\)/)).toBeInTheDocument();
    expect(screen.getByText("3.4 ms")).toBeInTheDocument();
  });

  it("flags a residual well above the floor", async () => {
    syncClipsMock.mockResolvedValue({
      ok: true,
      value: aligned({
        quality: {
          residual_rms_ms: 95.8,
          residual_max_ms: 115.7,
          degrees_of_freedom: 3,
          quantisation_floor_ms: 2.4,
          method_disagreement_ms: 0.1,
        },
      }),
    });
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    const residual = screen.getByText(/95\.8 ms rms \(39\.9× floor\)/);
    expect(residual).toHaveClass("text-status-degraded");
  });

  it("explains a missing residual rather than showing zero", async () => {
    syncClipsMock.mockResolvedValue({
      ok: true,
      value: aligned({
        quality: {
          residual_rms_ms: null,
          residual_max_ms: null,
          degrees_of_freedom: 0,
          quantisation_floor_ms: 3.4,
          method_disagreement_ms: null,
        },
      }),
    });
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText("not measurable")).toBeInTheDocument();
    expect(
      screen.getByText(/no spare degrees of freedom/i),
    ).toBeInTheDocument();
  });

  it("decomposes the confidence rather than showing only the product", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText("0.34 = 0.36 × 0.95 × 1.00")).toBeInTheDocument();
  });

  it("draws both clips on one clock", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    const timeline = screen.getByTestId("alignment-timeline");
    expect(within(timeline).getByText("face-on.mov")).toBeInTheDocument();
    expect(within(timeline).getByText("dtl.mov")).toBeInTheDocument();
    // Both bars are placed through the fitted map, so the target's span is
    // stated in the reference's clock rather than in its own.
    expect(within(timeline).getByText(/-0\.40/)).toBeInTheDocument();
  });

  it("renders a refusal as a finding, not as an error", async () => {
    syncClipsMock.mockResolvedValue({
      ok: true,
      value: aligned({
        aligned: false,
        method: null,
        time_map: null,
        quality: null,
        confidence: null,
        anchors: [],
        residuals: [],
        refusal: "The anchors cannot be reconciled.",
      }),
    });
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(screen.getByText("Not aligned")).toBeInTheDocument();
    expect(
      screen.getByText("The anchors cannot be reconciled."),
    ).toBeInTheDocument();
    // A refusal still carries the clip descriptions, so the manual picker works.
    expect(screen.getByLabelText("Frame in face-on.mov")).toBeInTheDocument();
  });

  it("surfaces an engine failure with a retry", async () => {
    syncClipsMock.mockResolvedValue({
      ok: false,
      error: { kind: "method", message: "No extracted poses for dtl.mov." },
    });
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(
      screen.getByText(/No extracted poses for dtl\.mov\./),
    ).toBeInTheDocument();
  });

  it("carries every warning through", async () => {
    render(<SyncPanel referencePath={REFERENCE} />);
    await chooseTarget();

    expect(
      screen.getByText(/cannot tell that both cameras filmed the same swing/i),
    ).toBeInTheDocument();
  });

  describe("manual anchors", () => {
    it("pins the current frame of each clip and re-aligns on the picks", async () => {
      render(<SyncPanel referencePath={REFERENCE} />);
      await chooseTarget();

      await userEvent.click(
        screen.getByRole("button", { name: /next frame in face-on\.mov/i }),
      );
      await userEvent.click(
        screen.getByRole("button", { name: /next frame in dtl\.mov/i }),
      );
      await userEvent.click(screen.getByRole("button", { name: /pin this/i }));

      expect(screen.getByText("anchor 1: 1 → 1")).toBeInTheDocument();

      syncClipsMock.mockClear();
      await userEvent.click(screen.getByRole("button", { name: /re-align/i }));

      await waitFor(() => {
        expect(syncClipsMock).toHaveBeenCalledWith(
          REFERENCE,
          TARGET,
          expect.objectContaining({
            anchors: [{ label: "anchor 1", referenceFrame: 1, targetFrame: 1 }],
          }),
        );
      });
    });

    it("says that a single pick is unchecked", async () => {
      render(<SyncPanel referencePath={REFERENCE} />);
      await chooseTarget();
      await userEvent.click(screen.getByRole("button", { name: /pin this/i }));

      expect(screen.getByText(/nothing checks it/i)).toBeInTheDocument();
    });

    it("cannot re-align with nothing pinned", async () => {
      render(<SyncPanel referencePath={REFERENCE} />);
      await chooseTarget();

      expect(screen.getByRole("button", { name: /re-align/i })).toBeDisabled();
    });

    it("bounds the picker by the frame count the engine reported", async () => {
      render(<SyncPanel referencePath={REFERENCE} />);
      await chooseTarget();

      const slider = screen.getByLabelText("Frame in face-on.mov");
      expect(slider).toHaveAttribute("max", "311");
      expect(
        screen.getByRole("button", { name: /previous frame in face-on\.mov/i }),
      ).toBeDisabled();
    });

    it("clears the pinned instants", async () => {
      render(<SyncPanel referencePath={REFERENCE} />);
      await chooseTarget();
      await userEvent.click(screen.getByRole("button", { name: /pin this/i }));
      await userEvent.click(
        screen.getByRole("button", { name: /clear pinned instants/i }),
      );

      expect(screen.queryByText(/anchor 1:/)).not.toBeInTheDocument();
    });
  });
});
