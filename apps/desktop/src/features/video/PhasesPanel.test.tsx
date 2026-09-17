import type { SwingPhases } from "@gsa/types";
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

const detectPhasesMock = vi.hoisted(() => vi.fn());
const onProgressMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  detectPhases: detectPhasesMock,
  onProgress: onProgressMock,
}));

const { PhasesPanel } = await import("./PhasesPanel");

const VIDEO = "/Users/example/data/raw/faceon.mov";

function confidence(overall: number) {
  return {
    overall,
    margin: overall,
    visibility: 1,
    resolution: 1,
  };
}

/** A detected swing shaped like the face-on reference clip. */
function detected(overrides: Partial<SwingPhases> = {}): SwingPhases {
  return {
    schema_version: 1,
    detected: true,
    events: [
      {
        event: "takeaway",
        frame_index: 13,
        timestamp_s: 0.433,
        confidence: confidence(0.97),
        methodology: "End of the last still stretch before the backswing.",
        corroboration_frame: null,
        corroboration_delta_s: null,
      },
      {
        event: "top",
        frame_index: 38,
        timestamp_s: 1.267,
        confidence: confidence(0.64),
        methodology: "Minimum hand speed near the highest hand position.",
        corroboration_frame: null,
        corroboration_delta_s: null,
      },
      {
        event: "impact",
        frame_index: 48,
        timestamp_s: 1.6,
        confidence: confidence(0.67),
        methodology: "Maximum hand speed. A kinematic estimate.",
        corroboration_frame: 47,
        corroboration_delta_s: -0.033,
      },
      {
        event: "finish",
        frame_index: 65,
        timestamp_s: 2.167,
        confidence: confidence(0),
        methodology: "Not reached: the clip ended first.",
        corroboration_frame: null,
        corroboration_delta_s: null,
      },
    ],
    phases: [
      {
        phase: "address",
        start_frame: 0,
        end_frame: 13,
        start_s: 0,
        end_s: 0.4,
        duration_s: 0.4,
        confidence: 0.97,
      },
      {
        phase: "backswing",
        start_frame: 13,
        end_frame: 38,
        start_s: 0.433,
        end_s: 1.233,
        duration_s: 0.8,
        confidence: 0.64,
      },
      {
        phase: "downswing",
        start_frame: 38,
        end_frame: 48,
        start_s: 1.267,
        end_s: 1.567,
        duration_s: 0.3,
        confidence: 0.64,
      },
      {
        phase: "follow_through",
        start_frame: 48,
        end_frame: 66,
        start_s: 1.6,
        end_s: 2.167,
        duration_s: 0.567,
        confidence: 0,
      },
    ],
    hand: {
      source: "right_wrist",
      valid_frames: 64,
      total_frames: 68,
      peak_speed: 2.624,
      travel: 0.338,
      torso_length: 0.088,
      travel_ratio: 3.86,
    },
    frames: 68,
    config: {
      moving_fraction: 0.05,
      min_travel_ratio: 0.5,
      min_phase_travel_ratio: 0.15,
      min_backswing_s: 0.2,
      min_downswing_s: 0.06,
      max_downswing_s: 1,
      transition_search_s: 0.25,
      confidence_window_s: 0.1,
      min_still_s: 0.1,
      min_tracking_gap_s: 0.15,
    },
    warnings: [],
    ...overrides,
  };
}

/** A clip the engine looked at and found no swing in. */
function undetected(): SwingPhases {
  return detected({
    detected: false,
    events: [],
    phases: [],
    warnings: [
      "No swing detected. The hands ranged over 0.02 torso lengths, below the 0.5 a swing requires.",
    ],
  });
}

async function run() {
  await userEvent.click(screen.getByRole("button", { name: /detect swing/i }));
}

describe("PhasesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    onProgressMock.mockResolvedValue(vi.fn());
  });

  it("detects nothing until asked", () => {
    render(<PhasesPanel videoPath={VIDEO} />);
    expect(screen.getByText(/nothing detected yet/i)).toBeInTheDocument();
    expect(detectPhasesMock).not.toHaveBeenCalled();
  });

  it("shows the located events with their confidence factors", async () => {
    detectPhasesMock.mockResolvedValue({ ok: true, value: detected() });
    render(<PhasesPanel videoPath={VIDEO} />);
    await run();

    expect(await screen.findByText("Swing detected.")).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("Impact");
    expect(table).toHaveTextContent("1.600s");
    expect(table).toHaveTextContent("0.67");
  });

  it("reports a clip with no swing as a finding, not a failure", async () => {
    // "Nobody swung" and "the worker crashed" must not render the same way.
    detectPhasesMock.mockResolvedValue({ ok: true, value: undetected() });
    render(<PhasesPanel videoPath={VIDEO} />);
    await run();

    expect(
      await screen.findByText(/no swing detected in this clip/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/0.02 torso lengths/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Frame")).not.toBeInTheDocument();
  });

  it("renders an engine failure with its remedy", async () => {
    detectPhasesMock.mockResolvedValue({
      ok: false,
      error: {
        kind: "method",
        message: "No extracted poses for faceon.mov.",
        code: -31001,
        data: { remediation: "Run: analyzer extract faceon.mov" },
      },
    });
    render(<PhasesPanel videoPath={VIDEO} />);
    await run();

    expect(await screen.findByText(/no extracted poses/i)).toBeInTheDocument();
    expect(
      screen.getByText("Run: analyzer extract faceon.mov"),
    ).toBeInTheDocument();
  });

  it("shows a failure rather than hanging when the bridge itself throws", async () => {
    detectPhasesMock.mockRejectedValue(new Error("worker died"));
    render(<PhasesPanel videoPath={VIDEO} />);
    await run();

    expect(await screen.findByText(/worker died/)).toBeInTheDocument();
  });

  it("unsubscribes from progress once the work is done", async () => {
    const unlisten = vi.fn();
    onProgressMock.mockResolvedValue(unlisten);
    detectPhasesMock.mockResolvedValue({ ok: true, value: detected() });

    render(<PhasesPanel videoPath={VIDEO} />);
    await run();

    await waitFor(() => {
      expect(unlisten).toHaveBeenCalled();
    });
  });

  describe("frame-by-frame inspector", () => {
    beforeEach(() => {
      detectPhasesMock.mockResolvedValue({ ok: true, value: detected() });
    });

    it("opens on impact, because that is the frame most worth checking", async () => {
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      const scrubber = await screen.findByLabelText("Frame");
      expect(scrubber).toHaveValue("48");
      // Scoped to the status readout: the phase name also appears in the
      // legend and on the timeline's jump buttons.
      expect(
        within(screen.getByRole("status")).getByText("Follow-through"),
      ).toBeInTheDocument();
    });

    it("names the phase the current frame falls in", async () => {
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      const scrubber = await screen.findByLabelText("Frame");
      // A range input is dragged, not typed into, so the change is fired
      // directly rather than going through keyboard simulation.
      fireEvent.change(scrubber, { target: { value: "20" } });

      // Frame 20 is inside the backswing (13-38).
      await waitFor(() => {
        expect(
          within(screen.getByRole("status")).getByText("Backswing"),
        ).toBeInTheDocument();
      });
    });

    it("steps one frame at a time across a phase boundary", async () => {
      // The reason the inspector exists: an event's frame number is only
      // checkable by seeing where the phase changes.
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      await userEvent.click(
        await screen.findByRole("button", { name: /^top$/i }),
      );
      expect(screen.getByLabelText("Frame")).toHaveValue("38");
      expect(
        within(screen.getByRole("status")).getByText("Downswing"),
      ).toBeInTheDocument();

      await userEvent.click(
        screen.getByRole("button", { name: /previous frame/i }),
      );
      expect(screen.getByLabelText("Frame")).toHaveValue("37");
      expect(
        within(screen.getByRole("status")).getByText("Backswing"),
      ).toBeInTheDocument();
    });

    it("jumps to an event and says which one the frame is", async () => {
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      await userEvent.click(
        await screen.findByRole("button", { name: /^takeaway$/i }),
      );
      expect(screen.getByLabelText("Frame")).toHaveValue("13");
      expect(screen.getByText(/confidence 0.97/)).toBeInTheDocument();
    });

    it("cannot step before the first frame", async () => {
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      const scrubber = await screen.findByLabelText("Frame");
      fireEvent.change(scrubber, { target: { value: "0" } });

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /previous frame/i }),
        ).toBeDisabled();
      });
    });

    it("offers each phase on the timeline as a jump target", async () => {
      render(<PhasesPanel videoPath={VIDEO} />);
      await run();

      await userEvent.click(
        await screen.findByRole("button", { name: /jump to downswing/i }),
      );
      expect(screen.getByLabelText("Frame")).toHaveValue("38");
    });
  });
});
