import type { EnvironmentReport } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const doctorMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/ipc", () => ({ doctor: doctorMock }));

const { HealthScreen } = await import("./HealthScreen");

function report(overrides: Partial<EnvironmentReport> = {}): EnvironmentReport {
  return {
    schema_version: 1,
    generated_at: "2026-09-16T00:00:00Z",
    overall_status: "ok",
    platform: {
      system: "Darwin",
      release: "25.4.0",
      machine: "arm64",
      cpu_count: 10,
      python_version: "3.12.14",
      python_executable: "/tmp/.venv/bin/python",
    },
    components: [
      {
        name: "ffmpeg",
        status: "ok",
        detail: "Found at /opt/homebrew/bin/ffmpeg.",
        version: "9.0.1",
      },
    ],
    compute: {
      torch_version: "2.14.0",
      mps_available: true,
      mps_built: true,
      cuda_available: false,
      selected_device: "mps",
      cpu_fallback_available: true,
      mediapipe_delegate: "cpu",
      ffmpeg_hwaccels: ["videotoolbox"],
    },
    warnings: [],
    ...overrides,
  };
}

describe("HealthScreen", () => {
  beforeEach(() => {
    doctorMock.mockReset();
  });

  it("renders each component with its measured version", async () => {
    doctorMock.mockResolvedValue({ ok: true, value: report() });

    render(<HealthScreen />);

    expect(await screen.findByText("ffmpeg")).toBeInTheDocument();
    expect(screen.getByText("9.0.1")).toBeInTheDocument();
  });

  it("shows remediation text for a component that needs action", async () => {
    doctorMock.mockResolvedValue({
      ok: true,
      value: report({
        overall_status: "missing",
        components: [
          {
            name: "model:pose_landmarker_full",
            status: "missing",
            detail: "pose_landmarker_full.task is not present.",
            remediation: "Run `python scripts/download_models.py`.",
          },
        ],
      }),
    });

    render(<HealthScreen />);

    expect(
      await screen.findByText(/Run `python scripts\/download_models\.py`\./),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Missing").length).toBeGreaterThan(0);
  });

  it("reports MediaPipe's delegate separately from the torch device", async () => {
    // Regression guard: collapsing these into one "GPU available" indicator
    // would claim GPU pose inference on a platform that has none.
    doctorMock.mockResolvedValue({ ok: true, value: report() });

    render(<HealthScreen />);

    expect(await screen.findByText("MediaPipe delegate")).toBeInTheDocument();
    expect(screen.getByText("cpu")).toBeInTheDocument();
    expect(screen.getByText("mps")).toBeInTheDocument();
  });

  it("surfaces warnings returned by the engine", async () => {
    doctorMock.mockResolvedValue({
      ok: true,
      value: report({ warnings: ["Pose inference runs on CPU."] }),
    });

    render(<HealthScreen />);

    expect(
      await screen.findByText("Pose inference runs on CPU."),
    ).toBeInTheDocument();
  });

  it("renders a failure instead of an empty healthy-looking list", async () => {
    doctorMock.mockResolvedValue({
      ok: false,
      error: { kind: "spawn", message: "uv not found on PATH" },
    });

    render(<HealthScreen />);

    expect(
      await screen.findByText("The analysis engine could not be started"),
    ).toBeInTheDocument();
    expect(screen.getByText(/uv not found on PATH/)).toBeInTheDocument();
    // Nothing may be presented as OK when the check itself failed.
    expect(screen.queryByText("OK")).not.toBeInTheDocument();
  });

  it("offers a retry that re-runs the check", async () => {
    doctorMock
      .mockResolvedValueOnce({
        ok: false,
        error: { kind: "transport", message: "worker exited" },
      })
      .mockResolvedValue({ ok: true, value: report() });

    const { default: userEvent } = await import("@testing-library/user-event");
    const user = userEvent.setup();

    render(<HealthScreen />);
    await user.click(await screen.findByRole("button", { name: /retry/i }));

    await waitFor(() => expect(doctorMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("ffmpeg")).toBeInTheDocument();
  });

  it("shows a loading state before the first result arrives", () => {
    doctorMock.mockReturnValue(new Promise(() => {}));

    render(<HealthScreen />);

    expect(screen.getByText(/Probing the analysis engine/)).toBeInTheDocument();
  });
});
