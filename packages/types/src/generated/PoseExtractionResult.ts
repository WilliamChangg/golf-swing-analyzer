/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * What the desktop app is told about a completed extraction.
 *
 * Deliberately excludes the landmarks themselves. A 240 fps clip carries tens
 * of thousands of frames of 33 landmarks in two spaces, which belongs in a
 * columnar file on disk rather than in a JSON-RPC response; the app is given
 * the path to read and the numbers to display.
 */
export interface PoseExtractionResult {
  schema_version?: number;
  video_path: string;
  /**
   * Parquet file holding the landmarks.
   */
  output_path: string;
  model: PoseModelInfo;
  extracted_at: string;
  stats: PoseExtractionStats;
  warnings?: string[];
}
/**
 * Which model produced a sequence, pinned so results stay attributable.
 */
export interface PoseModelInfo {
  /**
   * Manifest name, e.g. 'pose_landmarker_full'.
   */
  name: string;
  /**
   * lite / full / heavy.
   */
  variant: string;
  /**
   * Weight precision as the manifest records it.
   */
  precision: string;
  /**
   * Digest of the model file actually loaded.
   */
  sha256: string;
  /**
   * Where inference ran, as measured: 'cpu' or 'gpu'.
   */
  delegate: string;
  min_pose_detection_confidence: number;
  min_pose_presence_confidence: number;
  min_tracking_confidence: number;
}
/**
 * Measured facts about one extraction run.
 *
 * `detection_rate` is a count, not a quality score. A low rate means the model
 * found nobody in most frames; it says nothing about whether the landmarks it
 * did find are correct, and no accuracy claim may be built on it.
 */
export interface PoseExtractionStats {
  frames_processed: number;
  frames_detected: number;
  detection_rate: number;
  elapsed_s: number;
  ms_per_frame: number;
  /**
   * Mean visibility across detected landmarks. None when nothing was detected.
   */
  mean_visibility?: number | null;
}
