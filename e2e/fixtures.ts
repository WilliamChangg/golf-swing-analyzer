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

/**
 * A detected swing, shaped like the face-on reference clip.
 *
 * Frame numbers and confidences are the ones the engine actually produces on
 * `data/face-on/PW_face-on.mp4` with a 0.15 s window, so the inspector is
 * exercised against a real swing's proportions rather than round numbers — and
 * against a truncated finish, which is what that clip really has.
 */
export const SWING_PHASES = {
  schema_version: 1,
  detected: true,
  events: [
    {
      event: "takeaway",
      frame_index: 13,
      timestamp_s: 0.433,
      confidence: {
        overall: 0.97,
        margin: 1.0,
        visibility: 0.97,
        resolution: 1.0,
      },
      methodology:
        "End of the last stretch in which hand speed stayed below 5% of its peak before the top of the backswing.",
      corroboration_frame: null,
      corroboration_delta_s: null,
    },
    {
      event: "top",
      frame_index: 38,
      timestamp_s: 1.267,
      confidence: {
        overall: 0.64,
        margin: 0.84,
        visibility: 0.92,
        resolution: 0.83,
      },
      methodology:
        "Minimum hand speed within 0.25 s of the highest point the hands reached before impact.",
      corroboration_frame: null,
      corroboration_delta_s: null,
    },
    {
      event: "impact",
      frame_index: 48,
      timestamp_s: 1.6,
      confidence: {
        overall: 0.67,
        margin: 1.0,
        visibility: 0.81,
        resolution: 0.83,
      },
      methodology:
        "Maximum hand speed. A kinematic estimate: nothing here observes the ball or the club.",
      corroboration_frame: 47,
      corroboration_delta_s: -0.033,
    },
    {
      event: "finish",
      frame_index: 65,
      timestamp_s: 2.167,
      confidence: {
        overall: 0.0,
        margin: 0.0,
        visibility: 0.63,
        resolution: 1.0,
      },
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
      start_s: 0.0,
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
      confidence: 0.0,
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
    max_downswing_s: 1.0,
    transition_search_s: 0.25,
    confidence_window_s: 0.1,
    min_still_s: 0.1,
    min_tracking_gap_s: 0.15,
  },
  warnings: [
    "The hands had not come to rest when the clip ended, so the finish is reported at the last tracked frame rather than located. Its confidence is reduced accordingly.",
  ],
};

/** The same clip with no swing in it: a result, not an error. */
export const NO_SWING = {
  ...SWING_PHASES,
  detected: false,
  events: [],
  phases: [],
  hand: { ...SWING_PHASES.hand, travel: 0.004, travel_ratio: 0.04 },
  warnings: [
    "No swing detected. The hands ranged over 0.04 torso lengths, below the 0.5 a swing requires. Either the clip contains no swing, or the hands were not tracked through the part where one happened.",
  ],
};

/**
 * An alignment shaped like the real down-the-line/face-on pair.
 *
 * The numbers come from running `scripts/benchmark_sync.py --real` on the two
 * tour clips in `data/`, which are *not* the same swing — so the residual is
 * 40x the frame-rate floor and the confidence is 0.10. That is deliberately the
 * fixture: a flow that only ever renders a clean alignment never exercises the
 * presentation of the case a reader most needs to notice.
 */
export const SYNC_MODEL = {
  schema_version: 1,
  aligned: true,
  method: "combined",
  reference: {
    path: "/Users/example/data/face-on/rory_face_on.mp4",
    name: "rory_face_on.mp4",
    frames: 525,
    start_s: 0,
    duration_s: 3.1,
    median_interval_s: 0.004762,
    slow_motion_factor: 7,
  },
  target: {
    path: "/Users/example/data/dtl/rory_dtl.mp4",
    name: "rory_dtl.mp4",
    frames: 422,
    start_s: 0,
    duration_s: 2.813,
    median_interval_s: 0.006667,
    slow_motion_factor: 5,
  },
  time_map: {
    offset_s: -0.2577,
    rate: 1,
    rate_estimated: false,
    pivot_s: 1.631,
    offset_uncertainty_s: 0.0024,
    rate_uncertainty: null,
    support_start_s: 0.767,
    support_end_s: 2.5,
  },
  anchors: [
    {
      label: "takeaway",
      event: "takeaway",
      source: "detected",
      reference_frame: 161,
      target_frame: 64,
      reference_s: 0.767,
      target_s: 0.427,
      confidence: 0.95,
    },
    {
      label: "top",
      event: "top",
      source: "detected",
      reference_frame: 303,
      target_frame: 164,
      reference_s: 1.443,
      target_s: 1.093,
      confidence: 0.88,
    },
    {
      label: "impact",
      event: "impact",
      source: "detected",
      reference_frame: 381,
      target_frame: 247,
      reference_s: 1.814,
      target_s: 1.647,
      confidence: 0.64,
    },
    {
      label: "finish",
      event: "finish",
      source: "detected",
      reference_frame: 525,
      target_frame: 319,
      reference_s: 2.5,
      target_s: 2.127,
      confidence: 0.62,
    },
  ],
  residuals: [
    {
      label: "takeaway",
      reference_s: 0.767,
      observed_target_s: 0.427,
      predicted_target_s: 0.509,
      residual_ms: -82.3,
    },
    {
      label: "top",
      reference_s: 1.443,
      observed_target_s: 1.093,
      predicted_target_s: 1.185,
      residual_ms: -91.8,
    },
    {
      label: "impact",
      reference_s: 1.814,
      observed_target_s: 1.647,
      predicted_target_s: 1.557,
      residual_ms: 90.1,
    },
    {
      label: "finish",
      reference_s: 2.5,
      observed_target_s: 2.127,
      predicted_target_s: 2.242,
      residual_ms: -115.7,
    },
  ],
  quality: {
    residual_rms_ms: 95.8,
    residual_max_ms: 115.7,
    degrees_of_freedom: 3,
    quantisation_floor_ms: 2.4,
    method_disagreement_ms: 0.1,
  },
  confidence: {
    overall: 0.1,
    agreement: 0.12,
    anchors: 0.77,
    stability: 1,
  },
  correlation: {
    peak_correlation: 0.716,
    peak_offset_s: -0.2577,
    rival_correlation: 0.593,
    rival_offset_s: 2.195,
    grid_interval_s: 0.004762,
    overlap_s: 2.45,
    samples: 515,
    sub_grid_shift_s: 0.0002,
  },
  overlap: {
    start_s: 0,
    end_s: 2.792,
    duration_s: 2.792,
    reference_fraction: 0.9,
    target_fraction: 0.99,
  },
  config: {},
  refusal: null,
  warnings: [
    "Synchronisation aligns two swing-shaped signals; it cannot tell that both cameras filmed the same swing. The residual is the only evidence on that question, because two swings of different tempo cannot be aligned by any offset and any clock rate.",
  ],
};
