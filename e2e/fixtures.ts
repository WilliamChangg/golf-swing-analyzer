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
 * Stub Tauri's IPC bridge before the app's scripts run.
 *
 * `invoke` from @tauri-apps/api calls through `window.__TAURI_INTERNALS__`,
 * which only exists inside the real WebView. Installing it here lets the app
 * run unmodified in a browser.
 */
export async function stubEngine(
  page: Page,
  handler: { result?: unknown; error?: unknown } = { result: HEALTHY_REPORT },
): Promise<void> {
  await page.addInitScript((payload) => {
    (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
      invoke: (_cmd: string) =>
        "error" in payload && payload.error !== undefined
          ? // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- Tauri rejects with the plain object Rust serialised, not an Error. Rejecting with an Error here would test a shape the app never actually receives.
            Promise.reject(payload.error)
          : Promise.resolve(payload.result),
      transformCallback: (cb: unknown) => cb,
    };
  }, handler);
}
