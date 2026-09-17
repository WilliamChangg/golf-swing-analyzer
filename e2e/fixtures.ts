import type { Page } from "@playwright/test";

/**
 * A complete, realistic environment report.
 *
 * Values mirror what the engine actually returns on the reference machine
 * (Apple M1 Pro, macOS 26) so the rendered layout is exercised at the real
 * content lengths rather than with short placeholder strings.
 */
export const HEALTHY_REPORT = {
  schema_version: 1,
  generated_at: "2026-09-16T06:39:00Z",
  overall_status: "ok",
  platform: {
    system: "Darwin",
    release: "25.4.0",
    machine: "arm64",
    cpu_count: 10,
    python_version: "3.12.14",
    python_executable:
      "/Users/example/Documents/golf-swing-analyzer/python/.venv/bin/python",
  },
  components: [
    {
      name: "python",
      status: "ok",
      detail: "Python 3.12.14 at /Users/example/python/.venv/bin/python",
      version: "3.12.14",
    },
    {
      name: "numpy",
      status: "ok",
      detail: "`numpy` imported successfully.",
      version: "2.5.3",
    },
    {
      name: "opencv",
      status: "ok",
      detail: "`cv2` imported successfully.",
      version: "4.14.0",
    },
    {
      name: "ffmpeg",
      status: "ok",
      detail: "Found at /opt/homebrew/bin/ffmpeg.",
      version: "9.0.1",
    },
    {
      name: "model:pose_landmarker_full",
      status: "ok",
      detail:
        "pose_landmarker_full.task present and sha256 matches the manifest.",
      version: "4eaa5eb7a9836522",
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
  warnings: [
    "Pose inference runs on CPU: the MediaPipe Tasks Python API provides no GPU delegate on this platform. The torch device below does not change this.",
  ],
};

/**
 * A realistic `probe_video` result: a rotated, variable-rate phone clip.
 *
 * Deliberately the awkward case rather than the clean one, because the two
 * things the screen exists to communicate — rotation and non-uniform frame
 * timing — are only visible on a clip that has them.
 */
export const VFR_ROTATED_METADATA = {
  schema_version: 1,
  path: "/Users/example/data/raw/2026-09-15/faceon.mov",
  file_size_bytes: 41175,
  content_key: {
    algorithm: "sha256-sampled-v1",
    digest: "93c22821d3c1b694a0f1c2d3e4f5a6b7",
    size_bytes: 41175,
  },
  probed_at: "2026-09-16T06:39:00Z",
  container_format: "mov,mp4,m4a,3gp,3g2,mj2",
  stream: {
    codec_name: "h264",
    codec_long_name: "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
    profile: "High",
    pix_fmt: "yuv420p",
    coded_width: 1920,
    coded_height: 1080,
    display_width: 1080,
    display_height: 1920,
    rotation_ccw_degrees: 90,
    rotation_source: "display_matrix",
    sample_aspect_ratio: "1:1",
    bit_rate: 189696,
    time_base: "1/15360",
  },
  timing: {
    source: "decoded_frames",
    frame_count: 45,
    first_timestamp_s: 0,
    last_timestamp_s: 1.933333,
    timestamp_span_s: 1.933333,
    duration_s: 2,
    container_duration_s: 1.9,
    nominal_fps: 23.684211,
    measured_fps: 22.758621,
    intervals: {
      median_s: 0.033333,
      min_s: 0.033333,
      max_s: 0.066667,
      quantum_s: 0.0000651,
      irregular_count: 14,
      irregular_fraction: 0.318182,
    },
    is_vfr: true,
  },
  warnings: [
    "Variable frame rate: 14 of 44 intervals differ from the median by more than one time-base tick (from 33.33 ms to 66.67 ms). Frame times must come from presentation timestamps; frame_index / fps is not valid for this clip.",
    "The container requests a 90 degree counter-clockwise display rotation (stored 1920x1080, presented 1080x1920). Frames are returned already rotated.",
  ],
};

/** A completed pose extraction, as the engine reports one. */
export const POSE_RESULT = {
  schema_version: 1,
  video_path: "/Users/example/data/raw/2026-09-15/faceon.mov",
  output_path:
    "/Users/example/Library/Caches/golf-swing-analyzer/poses/sha256-sampled-v1-93c22821/pose_landmarker_full.parquet",
  model: {
    name: "pose_landmarker_full",
    variant: "full",
    precision: "float16",
    sha256: "4eaa5eb7a9836522aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    delegate: "cpu",
    min_pose_detection_confidence: 0.5,
    min_pose_presence_confidence: 0.5,
    min_tracking_confidence: 0.5,
  },
  extracted_at: "2026-09-16T06:41:00Z",
  stats: {
    frames_processed: 45,
    frames_detected: 44,
    detection_rate: 0.9778,
    elapsed_s: 0.69,
    ms_per_frame: 15.3,
    mean_visibility: 0.912,
  },
  warnings: [],
};

/** What one stubbed command does when the app calls it. */
export interface CommandStub {
  result?: unknown;
  error?: unknown;
}

/**
 * Stub Tauri's IPC bridge before the app's scripts run.
 *
 * `invoke` from @tauri-apps/api calls through `window.__TAURI_INTERNALS__`,
 * which only exists inside the real WebView. Installing it here lets the app
 * run unmodified in a browser.
 *
 * Commands are stubbed by name. The file picker goes through the same bridge as
 * the engine — the dialog plugin's `open` is `invoke("plugin:dialog|open")` —
 * so an import flow can be driven end to end without a native dialog.
 */
export async function stubEngine(
  page: Page,
  handlers: Record<string, CommandStub> = {},
): Promise<void> {
  const commands: Record<string, CommandStub> = {
    doctor: { result: HEALTHY_REPORT },
    // Tauri's event API rides the same bridge as commands. Answering it lets a
    // screen that subscribes to progress get past its `await` — without this,
    // any flow that reports progress would stall before it started.
    "plugin:event|listen": { result: 1 },
    "plugin:event|unlisten": { result: null },
    ...handlers,
  };

  await page.addInitScript((stubs) => {
    (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
      invoke: (cmd: string) => {
        const stub = stubs[cmd];
        if (stub === undefined) {
          // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- Tauri rejects an unknown command with a plain string, not an Error.
          return Promise.reject(`command ${cmd} not found`);
        }
        return "error" in stub && stub.error !== undefined
          ? // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- Tauri rejects with the plain object Rust serialised, not an Error. Rejecting with an Error here would test a shape the app never actually receives.
            Promise.reject(stub.error)
          : Promise.resolve(stub.result);
      },
      transformCallback: (cb: unknown) => cb,
      // Called by the unlisten function Tauri's `listen` returns. Absent, it
      // throws an unhandled rejection inside the page — which passes the test
      // while leaving the app's teardown path untested.
      unregisterListener: () => undefined,
    };
  }, commands);
}
