/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * Outcome of a single environment probe.
 *
 * OK        - present and usable
 * DEGRADED  - present but limited (e.g. a slower fallback path is in use)
 * MISSING   - not installed / not found; the user can fix this
 * ERROR     - the probe itself failed unexpectedly
 */
export type HealthStatus = "ok" | "degraded" | "missing" | "error";

/**
 * Complete environment health report returned by the `doctor` method.
 */
export interface EnvironmentReport {
  schema_version?: number;
  generated_at: string;
  /**
   * Worst status across all components, in the order OK < DEGRADED < MISSING < ERROR.
   */
  overall_status: HealthStatus;
  platform: PlatformInfo;
  components: ComponentStatus[];
  compute: ComputeInfo;
  /**
   * Non-fatal caveats the user should know about, e.g. CPU-only inference.
   */
  warnings?: string[];
}
/**
 * Facts about the host, read from the interpreter and OS.
 */
export interface PlatformInfo {
  system: string;
  release: string;
  machine: string;
  cpu_count: number | null;
  python_version: string;
  python_executable: string;
}
/**
 * Result of probing one required component of the analysis environment.
 */
export interface ComponentStatus {
  /**
   * Stable identifier, e.g. 'ffmpeg' or 'torch'.
   */
  name: string;
  status: HealthStatus;
  /**
   * Human-readable result of the probe.
   */
  detail: string;
  /**
   * Version string as reported by the component itself, never inferred.
   */
  version?: string | null;
  /**
   * Concrete action the user can take when status is not OK.
   */
  remediation?: string | null;
}
/**
 * Measured acceleration capabilities.
 *
 * `mediapipe_delegate` is reported separately from the torch device because the
 * two do not share a backend: on macOS the MediaPipe Tasks Python API has no
 * GPU delegate, so pose inference runs on CPU even when torch has MPS.
 */
export interface ComputeInfo {
  torch_version?: string | null;
  /**
   * torch.backends.mps.is_available()
   */
  mps_available: boolean;
  /**
   * torch.backends.mps.is_built()
   */
  mps_built: boolean;
  /**
   * torch.cuda.is_available()
   */
  cuda_available: boolean;
  /**
   * Device the engine will actually use.
   */
  selected_device: string;
  cpu_fallback_available: boolean;
  /**
   * Delegate MediaPipe will use on this platform ('cpu' or 'gpu').
   */
  mediapipe_delegate: string;
  /**
   * Hardware decoders reported by `ffmpeg -hwaccels`.
   */
  ffmpeg_hwaccels?: string[];
}
