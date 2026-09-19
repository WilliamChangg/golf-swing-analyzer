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
      // The asset protocol, which is how a `<video>` reaches a local file. In
      // the real app this returns an `asset:` URL that Rust serves, and only
      // for a path already admitted to the scope. Here it points at a route
      // Playwright fulfils with the real fixture bytes, so Chromium genuinely
      // decodes a real clip — which is what makes the seek assertions evidence
      // about a player rather than about the arithmetic.
      convertFileSrc: () => "/e2e-asset/vfr_30_to_15fps.mp4",
    };

    // The event plugin keeps its own global, and the unlisten function Tauri's
    // `listen` returns reaches for it by name. Without this, tearing down a
    // progress subscription throws an unhandled rejection inside the page --
    // which passes the test while leaving the app's teardown path untested,
    // and that is exactly what it was doing before Phase 14 put four
    // subscribe/unsubscribe cycles in one click.
    (
      window as unknown as Record<string, unknown>
    ).__TAURI_EVENT_PLUGIN_INTERNALS__ = {
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

/**
 * A calibration from a capture that determines what it measures.
 *
 * The numbers are the ones `scripts/benchmark_calibration.py` produces on a
 * well-spread 14-view capture, so the panel is exercised against a real
 * calibration's proportions.
 */
export const GOOD_CALIBRATION = {
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
      methodology: "Area is the fraction of a 12x8 grid containing a corner.",
    },
    degrees_of_freedom: 550,
  },
  detection: {
    frames_scanned: 200,
    frames_with_board: 60,
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
        reprojection_rms_px: 0.21,
        tilt_deg: 32,
        distance_m: 0.62,
        centroid_x: 420,
        centroid_y: 310,
        used: true,
      },
      {
        frame: 12,
        corners: 18,
        reprojection_rms_px: 0.29,
        tilt_deg: 11,
        distance_m: 0.88,
        centroid_x: 1520,
        centroid_y: 760,
        used: true,
      },
      {
        frame: 20,
        corners: 24,
        reprojection_rms_px: null,
        tilt_deg: null,
        distance_m: null,
        centroid_x: 900,
        centroid_y: 540,
        used: false,
        dropped_reason: "a view already covered this position and scale",
      },
    ],
    warnings: [],
  },
  calibrated_at: "2026-09-17T00:00:00Z",
  source: "/Users/example/data/calibration/faceon.mov",
  notes: "",
  usable: true,
  refusal: null,
  warnings: [],
};

/**
 * The capture this whole phase exists to catch.
 *
 * Its reprojection error is *better* than `GOOD_CALIBRATION`'s and its focal
 * length is wrong by a third. Used to assert that the UI never lets the
 * residual read as the verdict.
 */
export const DEGENERATE_CALIBRATION = {
  ...GOOD_CALIBRATION,
  intrinsics: { ...GOOD_CALIBRATION.intrinsics, fx: 1854.4, fy: 1854.9 },
  usable: false,
  refusal:
    "The board was held within 2 degrees of one orientation throughout, and 20 degrees of spread is required. Held square to the camera, a board cannot separate focal length from distance -- a longer lens further away makes the same picture -- so the fit is free to choose badly while fitting well. Tilt the board substantially between views.",
  quality: {
    ...GOOD_CALIBRATION.quality,
    rms_reprojection_px: 0.222,
    coverage: {
      ...GOOD_CALIBRATION.quality.coverage,
      image_fraction: 0.17,
      edge_fraction: 0,
      tilt_range_deg: 1.9,
      scale_range: 1.09,
    },
  },
};

// --- the player (Phase 14) -------------------------------------------------

/** Where the real fixture bytes are served from in these tests. */
export const CLIP_URL = "/e2e-asset/vfr_30_to_15fps.mp4";

/** The fixture on disk, relative to the repo root. */
export const CLIP_FIXTURE = "python/tests/fixtures/video/vfr_30_to_15fps.mp4";

/**
 * The seek index the engine really produces for that fixture.
 *
 * Copied from `analyzer.ingestion.seek_index`, not invented, because the whole
 * point of the player spec is that a browser lands on the frame these numbers
 * ask for. Made-up timestamps would test the arithmetic against itself.
 *
 * At full precision, deliberately. Rounding them to six decimals -- which is
 * what ffprobe prints, and what looks tidy here -- pushes each value a few
 * hundred nanoseconds off the presentation time it names, and that is enough to
 * land a seek on the wrong side of a frame boundary. Phase 1 made the same
 * decision one layer down by reading integer time-base ticks instead of
 * ffprobe's printed `pts_time`.
 *
 * The clip changes rate from 30 fps to 15 fps half-way through, and its
 * container declares an average of 23.684 — so `frame / declared fps` misses on
 * 39 of its 45 frames. That is what makes it the right fixture for a player.
 */
export const VFR_SEEK_INDEX = {
  schema_version: 1,
  path: "/Users/example/data/vfr_30_to_15fps.mp4",
  content_key: {
    algorithm: "sha256-sampled-v1",
    digest: "93c22821d3c1b69428e681001d0546cf8e34641d7f61fa3f423ef219641784a8",
    size_bytes: 41175,
  },
  source: "decoded_frames",
  frame_count: 45,
  timestamps_s: [
    0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333,
    0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3,
    0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335,
    0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6,
    0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333,
    0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9,
    0.9333333333333333, 0.9666666666666667, 1.0, 1.0666666666666667,
    1.1333333333333333, 1.2, 1.2666666666666666, 1.3333333333333333, 1.4,
    1.4666666666666666, 1.5333333333333334, 1.6, 1.6666666666666667,
    1.7333333333333334, 1.8, 1.8666666666666667, 1.9333333333333333,
  ],

  seek_targets_s: [
    0.016666666666666666, 0.05, 0.08333333333333334, 0.11666666666666667, 0.15,
    0.18333333333333335, 0.21666666666666667, 0.25, 0.2833333333333333,
    0.31666666666666665, 0.35, 0.3833333333333333, 0.4166666666666667, 0.45,
    0.48333333333333334, 0.5166666666666666, 0.55, 0.5833333333333333,
    0.6166666666666667, 0.6499999999999999, 0.6833333333333333,
    0.7166666666666666, 0.75, 0.7833333333333334, 0.8166666666666667,
    0.8500000000000001, 0.8833333333333333, 0.9166666666666667, 0.95,
    0.9833333333333334, 1.0333333333333332, 1.1, 1.1666666666666665,
    1.2333333333333334, 1.2999999999999998, 1.3666666666666667,
    1.4333333333333331, 1.5, 1.5666666666666669, 1.6333333333333333,
    1.7000000000000002, 1.7666666666666666, 1.8333333333333335, 1.9,
    1.9666666666666668,
  ],

  last_interval_s: 0.06666666666666665,
  warnings: [],
};

/** The container's declared average rate, which is what a naive player would use. */
export const VFR_DECLARED_FPS = 23.68421052631579;

/** Metadata for the same fixture, so the player can size its frame. */
export const VFR_CLIP_METADATA = {
  ...VFR_ROTATED_METADATA,
  path: VFR_SEEK_INDEX.path,
  content_key: VFR_SEEK_INDEX.content_key,
  stream: {
    ...VFR_ROTATED_METADATA.stream,
    coded_width: 320,
    coded_height: 240,
    display_width: 320,
    display_height: 240,
    rotation_ccw_degrees: 0,
    rotation_source: null,
  },
  timing: {
    ...VFR_ROTATED_METADATA.timing,
    frame_count: 45,
    nominal_fps: VFR_DECLARED_FPS,
    measured_fps: 22.758620689655171,
    is_vfr: true,
  },
};

/**
 * A metric set with one measurement and one refusal.
 *
 * Small on purpose: the panel's behaviour under a refusal is what these tests
 * are for, and forty metrics would only make the assertions harder to read.
 */
export const METRIC_SET = {
  schema_version: 1,
  computed: true,
  metrics: [
    {
      name: "timing.tempo_ratio",
      group: "timing",
      label: "Tempo ratio",
      value: 2.67,
      unit: "ratio",
      basis: "temporal",
      event: null,
      phase: null,
      source_frames: [13, 38, 46],
      view: "face_on",
      interpretation: "Backswing duration divided by downswing duration.",
      uncertainty: null,
      confidence: {
        overall: 0.57,
        observation: 1.0,
        anchor: 0.64,
        method: 0.89,
      },
      methodology:
        "Takeaway-to-top over top-to-impact, on the clip's own clock.",
    },
  ],
  refused: [
    {
      name: "rotation.x_factor",
      event: "top",
      reason:
        "A down-the-line camera does not contain this measurement: the shoulders project 0.10 torso lengths at address.",
    },
  ],
  view: {
    view: "face_on",
    confidence: 1.0,
    shoulder_span_ratio: 0.83,
    hip_span_ratio: 0.6,
    openness: 0.9,
    frames: [0, 1, 2],
    methodology: "Projected shoulder width at address.",
  },
  lead_side: null,
  references: [],
  torso_length: 0.31,
  warnings: [],
};

/**
 * A coaching report with no findings and eleven refusals.
 *
 * The ordinary outcome, and the one the panel is laid out around: on the 30 fps
 * amateur clip this engine reaches no conclusion at all.
 */
export const COACHING_REPORT = {
  schema_version: 1,
  computed: true,
  findings: [],
  refused: [
    {
      rule_id: "rotation.x_factor_top",
      title: "X-factor at the top",
      refusal: "basis_not_permitted",
      reason:
        "The published figure is a threshold on a spatial measurement; this clip measures a foreshortened angle.",
      source: {
        citation: "Golf Magazine (1992)",
        year: 1992,
        population: "unstated",
        sample_size: null,
        method: "convention",
        measures: "Difference between shoulder and pelvis turn at the top.",
        permitted_bases: [],
        filmed_from: null,
        note: "No measurement protocol is published.",
      },
    },
  ],
  rules_considered: 12,
  view: "face_on",
  frame_interval_s: 0.0333,
  phrasing: { mode: "off", attempted: 0, accepted: 0, rejected: 0 },
  warnings: [],
};

/** An overlay covering the fixture's frames, with one landmark blocked. */
export const POSE_OVERLAY = {
  schema_version: 1,
  video_path: VFR_SEEK_INDEX.path,
  content_key: VFR_SEEK_INDEX.content_key,
  geometry: { width: 320, height: 240 },
  start_frame: 0,
  end_frame: 45,
  frames: Array.from({ length: 45 }, (_, index) => ({
    frame_index: index,
    timestamp_s: VFR_SEEK_INDEX.timestamps_s[index] ?? 0,
    points: [
      {
        landmark: 11,
        x: 0.4 + index * 0.002,
        y: 0.35,
        state: "observed",
        visibility: 0.98,
      },
      {
        landmark: 12,
        x: 0.55 + index * 0.002,
        y: 0.35,
        state: "observed",
        visibility: 0.97,
      },
      // Blocked throughout, which is what the down-the-line clip really does
      // where motion blur loses a wrist.
      { landmark: 15, x: null, y: null, state: "blocked", visibility: 0.1 },
    ],
    shaft: null,
  })),
  landmarks: [11, 12, 15],
  connections: [[11, 12]],
  slow_motion_factor: 1,
  undistorted: false,
  club_tracked: false,
  warnings: [],
};

/** An empty session list, which is what a first run shows. */
export const NO_PROJECTS = { schema_version: 1, projects: [] };

// --- three dimensions (Phase 15) -------------------------------------------

/** One session, so the Three-D screen has something to reconstruct. */
export const ONE_PROJECT = {
  schema_version: 1,
  projects: [
    {
      schema_version: 1,
      id: 3,
      name: "Range session",
      notes: "",
      created_at: "2026-09-18T10:00:00Z",
      clips: [],
      syncs: [],
      rig: null,
    },
  ],
  database_path: "/Users/example/Library/Application Support/gsa/projects.db",
};

/** Landmarks the scene fixture carries: two shoulders and two elbows. */
const SCENE_LANDMARKS = [11, 12, 13, 14];

/**
 * A reconstruction over the same 45 frames the real fixture clip has.
 *
 * **The content key is the fixture clip's**, which is the whole point: the
 * viewport refuses to scrub against a recording it cannot prove is the one that
 * was reconstructed, and a scene carrying a made-up key would exercise the
 * refusal instead of the pairing. `scene.spec.ts` flips it to test the other
 * branch.
 *
 * The geometry is synthetic and deliberately simple — a body 2.5 m down the
 * reference camera's axis, sliding sideways one centimetre a frame — because
 * what this fixture is for is the *frame the viewport draws*, and a joint that
 * moves a known amount per frame is one whose drawn position says which frame is
 * on screen. The engine's real output is checked against a body whose 3D
 * positions are inputs in `python/tests/test_scene.py`, and the projection that
 * draws it is pinned to the engine's own in `projection.test.ts`.
 *
 * One frame is left entirely unreconstructed, because a viewport that held the
 * previous pose across it would look perfectly correct.
 */
export const SCENE_EMPTY_FRAME = 20;

export const RECONSTRUCTION_SCENE = {
  schema_version: 1,
  space: "camera",
  reference_content_key: VFR_SEEK_INDEX.content_key,
  reference_name: "vfr_30_to_15fps.mp4",
  target_name: "down_the_line.mp4",
  reference_role: "face_on",
  target_role: "down_the_line",
  start_frame: 0,
  end_frame: 45,
  frames: Array.from({ length: 45 }, (_, index) => {
    const empty = index === SCENE_EMPTY_FRAME;
    const points = SCENE_LANDMARKS.map((landmark) => ({
      landmark,
      position: empty
        ? null
        : {
            // One centimetre of travel per frame: over 45 frames that is 45 cm,
            // which at fx = 1400 and 2.5 m is about 250 px on screen. A frame
            // out is visibly a frame out.
            x: (landmark % 2 === 1 ? -0.2 : 0.2) + index * 0.01,
            y: landmark < 13 ? -0.25 : 0.05,
            z: 2.5,
          },
      refused: empty ? "not_seen" : null,
      uncertainty: empty
        ? null
        : {
            sigma_m: 0.02,
            xx: 1e-5,
            yy: 1e-5,
            zz: 4e-4,
            xy: 0,
            xz: 0,
            yz: 0,
          },
      convergence_deg: empty ? null : 88,
      reprojection_px: empty ? null : 0.9,
      visibility: empty ? 0.2 : 0.95,
    }));
    return {
      frame_index: index,
      timestamp_s: VFR_SEEK_INDEX.timestamps_s[index] ?? 0,
      points,
      reconstructed: empty ? 0 : points.length,
    };
  }),
  landmarks: SCENE_LANDMARKS,
  connections: [
    [11, 12],
    [11, 13],
    [12, 14],
  ],
  trajectories: [],
  cameras: [
    {
      kind: "reference",
      role: "face_on",
      name: "vfr_30_to_15fps.mp4",
      position: { x: 0, y: 0, z: 0 },
      forward: { x: 0, y: 0, z: 1 },
      up: { x: 0, y: -1, z: 0 },
      right: { x: 1, y: 0, z: 0 },
      fx: 1400,
      fy: 1400,
      cx: 960,
      cy: 540,
      image_width: 1920,
      image_height: 1080,
      horizontal_fov_deg: 69,
    },
    {
      kind: "target",
      role: "down_the_line",
      name: "down_the_line.mp4",
      position: { x: -2.5, y: 0, z: 2.5 },
      forward: { x: 1, y: 0, z: 0 },
      up: { x: 0, y: -1, z: 0 },
      right: { x: 0, y: 0, z: -1 },
      fx: 1500,
      fy: 1500,
      cx: 960,
      cy: 540,
      image_width: 1920,
      image_height: 1080,
      horizontal_fov_deg: 65,
    },
  ],
  calibration: "stereo",
  centroid: { x: 0.02, y: -0.1, z: 2.5 },
  radius_m: 0.45,
  slow_motion_factor: 1,
  pixel_sigma_px: 2.29,
  report: {
    schema_version: 1,
    reconstructed: true,
    space: "camera",
    reference_role: "face_on",
    target_role: "down_the_line",
    reference_name: "vfr_30_to_15fps.mp4",
    target_name: "down_the_line.mp4",
    frames: 45,
    reconstructed_frames: 44,
    slow_motion_factor: 1,
    calibration: "stereo",
    baseline_m: 3.54,
    convergence_deg: 90,
    quality: {
      points_attempted: 180,
      points_reconstructed: 176,
      coverage: 0.978,
      median_reprojection_px: 0.9,
      max_reprojection_px: 2.4,
      median_convergence_deg: 88,
      min_convergence_deg: 71,
      median_uncertainty_m: 0.02,
      p95_uncertainty_m: 0.031,
      bones: [],
      symmetry: [],
      worst_bone_variation: 0.04,
      pixel_sigma_px: 2.29,
      methodology: "DLT, refined on reprojection error in both views",
    },
    pairing: null,
    landmarks: [],
    reconstructed_at: null,
    config: {},
    refusal: null,
    warnings: [],
  },
  warnings: [],
};
