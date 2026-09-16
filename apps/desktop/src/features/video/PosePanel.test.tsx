import type { PoseExtractionResult, ProgressUpdate } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";

const extractPosesMock = vi.hoisted(() => vi.fn());
const onProgressMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  // `progressFraction` and `remediationOf` stay real: the rendering under test
  // depends on them, and stubbing them would test different functions than the
  // app runs.
  ...(await importOriginal<typeof IpcModule>()),
  extractPoses: extractPosesMock,
  onProgress: onProgressMock,
}));

const { PosePanel } = await import("./PosePanel");

const VIDEO = "/Users/example/data/raw/faceon.mov";

function result(
  overrides: Partial<PoseExtractionResult> = {},
): PoseExtractionResult {
  return {
    schema_version: 1,
    video_path: VIDEO,
    output_path: "/cache/poses/abc/pose_landmarker_full.parquet",
    model: {
      name: "pose_landmarker_full",
      variant: "full",
      precision: "float16",
      sha256:
        "4eaa5eb7a9836522aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      delegate: "cpu",
      min_pose_detection_confidence: 0.5,
      min_pose_presence_confidence: 0.5,
      min_tracking_confidence: 0.5,
    },
    extracted_at: "2026-09-16T00:00:00Z",
    stats: {
      frames_processed: 600,
      frames_detected: 594,
      detection_rate: 0.99,
      elapsed_s: 9.2,
      ms_per_frame: 15.3,
      mean_visibility: 0.912,
    },
    warnings: [],
    ...overrides,
  };
}

function update(overrides: Partial<ProgressUpdate> = {}): ProgressUpdate {
  return {
    schema_version: 1,
    request_id: 1,
    task: "extract_poses",
    stage: "estimating",
    current: 150,
    total: 600,
    elapsed_s: 2.5,
    detail: null,
    ...overrides,
  };
}

/** Capture the handler the panel subscribes with, so a test can drive it. */
function captureProgressHandler(): () => (u: ProgressUpdate) => void {
  let handler: ((u: ProgressUpdate) => void) | undefined;
  onProgressMock.mockImplementation((fn: (u: ProgressUpdate) => void) => {
    handler = fn;
    return Promise.resolve(() => undefined);
  });
  return () => {
    if (!handler) throw new Error("the panel did not subscribe to progress");
    return handler;
  };
}

describe("PosePanel", () => {
  beforeEach(() => {
    extractPosesMock.mockReset();
    onProgressMock.mockReset();
    onProgressMock.mockResolvedValue(() => undefined);
  });

  it("starts with nothing extracted rather than implying a result", () => {
    render(<PosePanel videoPath={VIDEO} />);
    expect(screen.getByText(/No landmarks extracted/i)).toBeInTheDocument();
    expect(extractPosesMock).not.toHaveBeenCalled();
  });

  it("extracts the clip it was given", async () => {
    extractPosesMock.mockResolvedValue({ ok: true, value: result() });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(extractPosesMock).toHaveBeenCalledWith(VIDEO);
    });
  });

  it("subscribes to progress before starting, not after", async () => {
    /* A clip short enough to finish immediately would otherwise report nothing,
       because the listener would be registered after the work was done. */
    const order: string[] = [];
    onProgressMock.mockImplementation(() => {
      order.push("subscribe");
      return Promise.resolve(() => undefined);
    });
    extractPosesMock.mockImplementation(() => {
      order.push("extract");
      return Promise.resolve({ ok: true, value: result() });
    });

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(order).toEqual(["subscribe", "extract"]);
    });
  });

  it("listens only for its own task", async () => {
    extractPosesMock.mockResolvedValue({ ok: true, value: result() });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(onProgressMock).toHaveBeenCalledWith(expect.any(Function), {
        task: "extract_poses",
      });
    });
  });

  it("shows measured progress while running", async () => {
    const handler = captureProgressHandler();
    extractPosesMock.mockReturnValue(new Promise(() => undefined)); // never settles

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeInTheDocument();
    });

    handler()(update());

    expect(await screen.findByText("25%")).toBeInTheDocument();
    expect(screen.getByText(/frame 150 of 600/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "25",
    );
  });

  it("shows no percentage when the engine gave no total", async () => {
    /* A bar that invents a denominator is worse than one that admits it has none. */
    const handler = captureProgressHandler();
    extractPosesMock.mockReturnValue(new Promise(() => undefined));

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeInTheDocument();
    });

    handler()(update({ total: null }));

    expect(await screen.findByText("—")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).not.toHaveAttribute(
      "aria-valuenow",
    );
  });

  it("names the loading stage distinctly from the estimating stage", async () => {
    const handler = captureProgressHandler();
    extractPosesMock.mockReturnValue(new Promise(() => undefined));

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeInTheDocument();
    });

    handler()(update({ stage: "starting", current: 0 }));
    expect(await screen.findByText(/Loading the model/)).toBeInTheDocument();
  });

  it("reports what the engine measured when it finishes", async () => {
    extractPosesMock.mockResolvedValue({ ok: true, value: result() });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    expect(await screen.findByText("600")).toBeInTheDocument();
    expect(screen.getByText("594 (99.0%)")).toBeInTheDocument();
    expect(screen.getByText("15.3 ms")).toBeInTheDocument();
    expect(
      screen.getByText("pose_landmarker_full (float16)"),
    ).toBeInTheDocument();
  });

  it("says mean visibility was not measured rather than showing zero", async () => {
    /* An average over nothing does not exist; rendering 0.000 would invent one. */
    extractPosesMock.mockResolvedValue({
      ok: true,
      value: result({
        stats: { ...result().stats, frames_detected: 0, mean_visibility: null },
      }),
    });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );
    expect(await screen.findByText("not measured")).toBeInTheDocument();
  });

  it("surfaces the engine's warnings", async () => {
    const warning = "No pose was detected in any frame.";
    extractPosesMock.mockResolvedValue({
      ok: true,
      value: result({ warnings: [warning] }),
    });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );
    expect(await screen.findByText(warning)).toBeInTheDocument();
  });

  it("shows the engine's remediation when extraction fails", async () => {
    extractPosesMock.mockResolvedValue({
      ok: false,
      error: {
        kind: "method",
        message: "Model 'pose_landmarker_full' is not downloaded.",
        code: -31001,
        data: { remediation: "Run `python scripts/download_models.py`." },
      },
    });
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    expect(await screen.findByText(/is not downloaded/)).toBeInTheDocument();
    expect(
      screen.getByText("Run `python scripts/download_models.py`."),
    ).toBeInTheDocument();
  });

  it("unsubscribes once the work is done", async () => {
    const unlisten = vi.fn();
    onProgressMock.mockResolvedValue(unlisten);
    extractPosesMock.mockResolvedValue({ ok: true, value: result() });

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(unlisten).toHaveBeenCalled();
    });
  });

  it("unsubscribes even when extraction fails", async () => {
    const unlisten = vi.fn();
    onProgressMock.mockResolvedValue(unlisten);
    extractPosesMock.mockRejectedValue(new Error("worker died"));

    render(<PosePanel videoPath={VIDEO} />);
    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(unlisten).toHaveBeenCalled();
    });
  });

  it("disables the button while running so a second run cannot start", async () => {
    extractPosesMock.mockReturnValue(new Promise(() => undefined));
    render(<PosePanel videoPath={VIDEO} />);

    await userEvent.click(
      screen.getByRole("button", { name: /extract pose/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /extract pose/i }),
      ).toBeDisabled();
    });
  });
});
