import type * as IpcModule from "@/lib/ipc";
import type { VideoMetadata } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const probeVideoMock = vi.hoisted(() => vi.fn());
const openMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  // `remediationOf` is real: the error panel's behaviour depends on it, and
  // stubbing it would test a different function than the app runs.
  ...(await importOriginal<typeof IpcModule>()),
  probeVideo: probeVideoMock,
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: openMock }));

const { VideoScreen } = await import("./VideoScreen");

const PATH = "/Users/example/data/raw/faceon.mov";

function metadata(overrides: Partial<VideoMetadata> = {}): VideoMetadata {
  return {
    schema_version: 1,
    path: PATH,
    file_size_bytes: 48921,
    content_key: {
      algorithm: "sha256-sampled-v1",
      digest: "831f54a67850b694",
      size_bytes: 48921,
    },
    probed_at: "2026-09-16T00:00:00Z",
    container_format: "mov,mp4,m4a,3gp,3g2,mj2",
    stream: {
      codec_name: "h264",
      codec_long_name: "H.264 / AVC",
      profile: "High",
      pix_fmt: "yuv420p",
      coded_width: 1920,
      coded_height: 1080,
      display_width: 1920,
      display_height: 1080,
      rotation_ccw_degrees: 0,
      rotation_source: null,
      sample_aspect_ratio: "1:1",
      bit_rate: 189696,
      time_base: "1/15360",
    },
    timing: {
      source: "decoded_frames",
      frame_count: 60,
      first_timestamp_s: 0,
      last_timestamp_s: 1.966667,
      timestamp_span_s: 1.966667,
      duration_s: 2,
      container_duration_s: 2,
      nominal_fps: 30,
      measured_fps: 30,
      intervals: {
        median_s: 0.033333,
        min_s: 0.033333,
        max_s: 0.033333,
        quantum_s: 0.0000651,
        irregular_count: 0,
        irregular_fraction: 0,
      },
      is_vfr: false,
    },
    warnings: [],
    ...overrides,
  };
}

/** Choose a file through the picker and wait for the probe to settle. */
async function importClip(result: unknown, path = PATH): Promise<void> {
  openMock.mockResolvedValue(path);
  probeVideoMock.mockResolvedValue(result);
  render(<VideoScreen />);
  await userEvent.click(screen.getByRole("button", { name: /choose video/i }));
  await waitFor(() => {
    expect(probeVideoMock).toHaveBeenCalled();
  });
}

describe("VideoScreen", () => {
  beforeEach(() => {
    probeVideoMock.mockReset();
    openMock.mockReset();
  });

  it("starts empty rather than implying a clip is loaded", () => {
    render(<VideoScreen />);
    expect(screen.getByText(/no clip loaded/i)).toBeInTheDocument();
    expect(probeVideoMock).not.toHaveBeenCalled();
  });

  it("probes the path the picker returned", async () => {
    await importClip({ ok: true, value: metadata() });
    expect(probeVideoMock).toHaveBeenCalledWith(PATH, { refresh: false });
  });

  it("does not probe when the picker is cancelled", async () => {
    openMock.mockResolvedValue(null);
    render(<VideoScreen />);
    await userEvent.click(
      screen.getByRole("button", { name: /choose video/i }),
    );
    expect(probeVideoMock).not.toHaveBeenCalled();
  });

  it("renders the measured frame rate, not the declared one", async () => {
    await importClip({
      ok: true,
      value: metadata({
        timing: {
          ...metadata().timing,
          measured_fps: 22.759,
          nominal_fps: 23.684,
        },
      }),
    });
    expect(await screen.findByText("22.759 fps")).toBeInTheDocument();
    expect(screen.getByText("23.684 fps")).toBeInTheDocument();
  });

  it("calls a constant-rate clip constant", async () => {
    await importClip({ ok: true, value: metadata() });
    expect(await screen.findByText("Constant")).toBeInTheDocument();
  });

  it("calls a variable-rate clip variable", async () => {
    await importClip({
      ok: true,
      value: metadata({ timing: { ...metadata().timing, is_vfr: true } }),
    });
    expect(await screen.findByText("Variable")).toBeInTheDocument();
  });

  it("says unknown when the engine could not determine the frame rate", async () => {
    /* A null verdict means no timestamps were found. Rendering that as
       "Constant" would turn an absence of evidence into a finding. */
    await importClip({
      ok: true,
      value: metadata({
        timing: {
          ...metadata().timing,
          is_vfr: null,
          source: "container_rate",
          intervals: null,
        },
      }),
    });
    expect(await screen.findByText("Unknown")).toBeInTheDocument();
    expect(screen.queryByText("Constant")).not.toBeInTheDocument();
  });

  it("shows both stored and display size for a rotated clip", async () => {
    await importClip({
      ok: true,
      value: metadata({
        stream: {
          ...metadata().stream,
          display_width: 1080,
          display_height: 1920,
          rotation_ccw_degrees: 90,
          rotation_source: "display_matrix",
        },
      }),
    });
    expect(await screen.findByText("1080×1920")).toBeInTheDocument();
    expect(screen.getByText("1920×1080")).toBeInTheDocument();
    expect(screen.getByText("90° ccw (display_matrix)")).toBeInTheDocument();
  });

  it("omits the stored size when it is the same as the display size", async () => {
    await importClip({ ok: true, value: metadata() });
    await screen.findByText("1920×1080");
    expect(screen.queryByText("Stored size")).not.toBeInTheDocument();
  });

  it("surfaces the engine's caveats verbatim", async () => {
    const warning = "Variable frame rate: 14 of 44 intervals differ.";
    await importClip({
      ok: true,
      value: metadata({ warnings: [warning] }),
    });
    expect(await screen.findByText(warning)).toBeInTheDocument();
  });

  it("shows the engine's remediation when a file is rejected", async () => {
    await importClip({
      ok: false,
      error: {
        kind: "method",
        message: "faceon.mov contains no video stream.",
        code: -31001,
        data: { remediation: "Select a video recording." },
      },
    });
    expect(
      await screen.findByText(/contains no video stream/),
    ).toBeInTheDocument();
    expect(screen.getByText("Select a video recording.")).toBeInTheDocument();
  });

  it("re-reads with refresh when asked to", async () => {
    await importClip({ ok: true, value: metadata() });
    await screen.findByText("Constant");

    await userEvent.click(screen.getByRole("button", { name: /re-read/i }));
    expect(probeVideoMock).toHaveBeenLastCalledWith(PATH, { refresh: true });
  });

  it("offers no re-read before anything is loaded", () => {
    render(<VideoScreen />);
    expect(
      screen.queryByRole("button", { name: /re-read/i }),
    ).not.toBeInTheDocument();
  });
});
