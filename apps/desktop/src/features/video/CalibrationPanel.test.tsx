import type { CameraCalibration } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";

const calibrateCameraMock = vi.hoisted(() => vi.fn());
const onProgressMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  calibrateCamera: calibrateCameraMock,
  onProgress: onProgressMock,
}));

const { CalibrationPanel } = await import("./CalibrationPanel");

const SOURCE = "/Users/example/data/calibration/faceon";

function calibration(
  overrides: Partial<CameraCalibration> = {},
): CameraCalibration {
  return {
    schema_version: 1,
    role: "face_on",
    intrinsics: {
      fx: 1399.4,
      fy: 1399.7,
      cx: 957.7,
      cy: 540.3,
      distortion: [-0.28, 0.1208, 0.0013, -0.0004],
      model: "radial_tangential_4",
      image_width: 1920,
      image_height: 1080,
      fx_uncertainty: 4.65,
      fy_uncertainty: 4.6,
      cx_uncertainty: 1.2,
      cy_uncertainty: 1.1,
    },
    quality: {
      rms_reprojection_px: 0.237,
      max_reprojection_px: 0.72,
      per_view_rms_px: [0.2, 0.25, 0.28],
      coverage: {
        views: 14,
        corners: 321,
        image_fraction: 0.82,
        edge_fraction: 0.16,
        tilt_range_deg: 41.8,
        scale_range: 2.46,
        methodology: "grid over the frame",
      },
      degrees_of_freedom: 550,
    },
    detection: {
      frames_scanned: 40,
      frames_with_board: 20,
      views_used: 14,
      corners_total: 321,
      board: {
        squares_x: 7,
        squares_y: 5,
        square_length_m: 0.035,
        marker_length_m: 0.026,
        family: "DICT_5X5_100",
        legacy_pattern: false,
      },
      observations: [
        {
          frame: 0,
          corners: 24,
          reprojection_rms_px: 0.2,
          tilt_deg: 30,
          distance_m: 0.6,
          centroid_x: 400,
          centroid_y: 300,
          used: true,
        },
        {
          frame: 1,
          corners: 24,
          reprojection_rms_px: null,
          tilt_deg: null,
          distance_m: null,
          centroid_x: 1500,
          centroid_y: 800,
          used: false,
          dropped_reason: "a view already covered this position and scale",
        },
      ],
      warnings: [],
    },
    calibrated_at: "2026-09-17T00:00:00Z",
    source: SOURCE,
    notes: "",
    usable: true,
    refusal: null,
    warnings: [],
    ...overrides,
  };
}

/** The degenerate capture: fits beautifully, determines nothing, is refused. */
function refused(): CameraCalibration {
  const base = calibration();
  return {
    ...base,
    usable: false,
    refusal:
      "The board was held within 2 degrees of one orientation throughout, and 20 degrees of spread is required. Held square to the camera, a board cannot separate focal length from distance.",
    quality: {
      ...base.quality,
      // Deliberately *better* than the usable one, which is the whole point.
      rms_reprojection_px: 0.222,
      coverage: {
        ...base.quality.coverage,
        tilt_range_deg: 1.9,
        image_fraction: 0.17,
        edge_fraction: 0,
        scale_range: 1.09,
      },
    },
  };
}

beforeEach(() => {
  calibrateCameraMock.mockReset();
  onProgressMock.mockReset();
  onProgressMock.mockResolvedValue(() => undefined);
});

describe("CalibrationPanel", () => {
  it("does not calibrate until asked", () => {
    render(<CalibrationPanel source={SOURCE} />);
    expect(
      screen.getByRole("button", { name: /find the board/i }),
    ).toBeInTheDocument();
    expect(calibrateCameraMock).not.toHaveBeenCalled();
  });

  it("reports what was measured and how well it is known", async () => {
    calibrateCameraMock.mockResolvedValue({ ok: true, value: calibration() });
    render(<CalibrationPanel source={SOURCE} />);

    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByText(/this calibration may be used/i),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("fx 1399.4, fy 1399.7 px")).toBeInTheDocument();
    expect(screen.getByText("0.237 px rms")).toBeInTheDocument();
    expect(screen.getByText("4.65 px (0.33%)")).toBeInTheDocument();
  });

  it("states the field of view, which is the one number a person can check", async () => {
    calibrateCameraMock.mockResolvedValue({ ok: true, value: calibration() });
    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    // 2 * atan(1920 / (2 * 1399.4)) = 68.9 degrees.
    await waitFor(() => {
      expect(screen.getByText("68.9° across")).toBeInTheDocument();
    });
  });

  it("shows a refusal as a result rather than as an error", async () => {
    calibrateCameraMock.mockResolvedValue({ ok: true, value: refused() });
    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByText(/this calibration will not be used/i),
      ).toBeInTheDocument();
    });
    // The numbers are still there: they are what say what to reshoot.
    expect(
      screen.getByText(/cannot separate focal length from distance/i),
    ).toBeInTheDocument();
    expect(screen.getByText("0.222 px rms")).toBeInTheDocument();
    // The coverage numbers that caused the refusal are shown, not just the words.
    expect(screen.getByText("2°")).toBeInTheDocument();
    expect(screen.getByText("17%")).toBeInTheDocument();
  });

  it("does not let a good residual look like a good calibration", async () => {
    calibrateCameraMock.mockResolvedValue({ ok: true, value: refused() });
    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    await waitFor(() => {
      expect(screen.getByText("0.222 px rms")).toBeInTheDocument();
    });

    // The residual is labelled with what it actually measures, right next to it.
    expect(
      screen.getByText(/how well the model fits these views/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/a capture that constrains nothing fits better/i),
    ).toBeInTheDocument();
  });

  it("draws every detected view, including the ones it did not use", async () => {
    calibrateCameraMock.mockResolvedValue({ ok: true, value: calibration() });
    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    const map = await screen.findByRole("img", {
      name: /1 view\(s\) used of 2 detected/i,
    });
    // Both observations are drawn: a dropped view is evidence about the
    // capture, not noise to hide.
    expect(map.querySelectorAll("circle")).toHaveLength(2);
  });

  it("surfaces an engine failure with a retry", async () => {
    calibrateCameraMock.mockResolvedValue({
      ok: false,
      error: { kind: "method", message: "Unknown method 'calibrate_camera'" },
    });
    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    await waitFor(() => {
      expect(screen.getByText(/unknown method/i)).toBeInTheDocument();
    });
  });

  it("reports progress while the board is being looked for", async () => {
    let emit: ((update: unknown) => void) | null = null;
    onProgressMock.mockImplementation((handler: (update: unknown) => void) => {
      emit = handler;
      return Promise.resolve(() => undefined);
    });
    calibrateCameraMock.mockReturnValue(new Promise(() => undefined));

    render(<CalibrationPanel source={SOURCE} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the board/i }),
    );

    await waitFor(() => {
      expect(emit).not.toBeNull();
    });
    emit?.({
      schema_version: 1,
      request_id: 1,
      task: "detect_board",
      stage: "scanning",
      current: 20,
      total: 40,
      elapsed_s: 1.2,
      detail: "6 views",
    });

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toHaveAttribute(
        "aria-valuenow",
        "50",
      );
    });
    expect(screen.getByText(/6 views/)).toBeInTheDocument();
  });
});
