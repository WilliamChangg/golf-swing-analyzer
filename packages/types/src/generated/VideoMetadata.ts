/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * How a `ContentKey` digest was computed.
 *
 * SHA256          - the whole file. Use when the answer must be an integrity
 *                   check, as it is for model weights.
 * SHA256_SAMPLED  - size, plus sha256 over three fixed windows of the file.
 *                   **This is not an integrity check.** It exists because
 *                   hashing every byte of a multi-gigabyte recording on every
 *                   open costs seconds, and a cache key only has to change
 *                   when the file changes, not prove that it did not.
 */
export type HashAlgorithm = "sha256" | "sha256-sampled-v1";
/**
 * Where a clip's per-frame times came from.
 *
 * PACKET_PTS      - presentation timestamps read from the container index.
 *                   Authoritative: these are the times the frames are to be
 *                   shown at, whether or not they are evenly spaced.
 * CONTAINER_RATE  - synthesised from the container's declared average frame
 *                   rate because no usable timestamps were found. A fallback,
 *                   and one that makes variable frame rate undetectable, so
 *                   anything derived from it is flagged.
 */
export type TimestampSource = "packet_pts" | "container_rate";

/**
 * Everything known about a video file without decoding it.
 *
 * Produced by `probe_video`. Every field is read from the container by ffprobe
 * or computed from timestamps that were; nothing here is assumed.
 */
export interface VideoMetadata {
  schema_version?: number;
  path: string;
  file_size_bytes: number;
  content_key: ContentKey;
  probed_at: string;
  /**
   * ffprobe's format_name, e.g. 'mov,mp4,m4a,3gp,3g2,mj2'.
   */
  container_format: string;
  stream: VideoStreamInfo;
  timing: VideoTiming;
  /**
   * Caveats that affect how the clip may be analysed, e.g. variable frame rate.
   */
  warnings?: string[];
}
/**
 * Identity of a file's contents, used to key derived artifacts.
 *
 * The algorithm travels with the digest so a cache written under one scheme is
 * never silently compared against a digest produced by another.
 */
export interface ContentKey {
  algorithm: HashAlgorithm;
  /**
   * Lowercase hex sha256 digest.
   */
  digest: string;
  /**
   * File size at the time the digest was taken.
   */
  size_bytes: number;
}
/**
 * The video stream's coding and geometry, as the container declares them.
 */
export interface VideoStreamInfo {
  codec_name: string;
  codec_long_name?: string | null;
  profile?: string | null;
  pix_fmt?: string | null;
  /**
   * Width as stored, before any display rotation.
   */
  coded_width: number;
  /**
   * Height as stored, before any display rotation.
   */
  coded_height: number;
  /**
   * Width after rotation. What a frame source yields.
   */
  display_width: number;
  /**
   * Height after rotation. What a frame source yields.
   */
  display_height: number;
  /**
   * Counter-clockwise rotation to apply to a stored frame to display it upright, taken from the container's display matrix.
   */
  rotation_ccw_degrees: 0 | 90 | 180 | 270;
  /**
   * Which container field the rotation came from: 'display_matrix' or 'rotate_tag'.
   */
  rotation_source?: string | null;
  sample_aspect_ratio?: string | null;
  bit_rate?: number | null;
  /**
   * Stream time base as a rational string, e.g. '1/15360'.
   */
  time_base: string;
}
/**
 * When each frame is presented, and whether that is evenly spaced.
 */
export interface VideoTiming {
  source: TimestampSource;
  /**
   * Frames found in the container index.
   */
  frame_count: number;
  first_timestamp_s: number;
  last_timestamp_s: number;
  /**
   * last - first. Measured exactly; assumes nothing about the final frame.
   */
  timestamp_span_s: number;
  /**
   * Estimated playback duration: the timestamp span plus the last observed interval, since how long the final frame is displayed is not recorded.
   */
  duration_s: number;
  /**
   * Duration as declared by the container, for comparison. Not used for analysis.
   */
  container_duration_s?: number | null;
  /**
   * The container's declared average frame rate. Display only.
   */
  nominal_fps?: number | null;
  /**
   * (frame_count - 1) / timestamp_span_s. Meaningful as an average even on variable-rate clips; None when there are fewer than two frames.
   */
  measured_fps?: number | null;
  /**
   * None when timestamps were synthesised, or when there are fewer than two frames.
   */
  intervals?: IntervalStats | null;
  /**
   * True when any interval differs from the median by more than one time-base tick. None means it could not be determined, which is not the same as False.
   */
  is_vfr?: boolean | null;
}
/**
 * Measured spacing between consecutive presentation timestamps.
 *
 * Reported so the `is_vfr` verdict can be checked rather than trusted: a clip
 * with one irregular interval out of 7000 and a clip that switches rate
 * half-way through are both "variable", and these numbers are what tells them
 * apart.
 */
export interface IntervalStats {
  median_s: number;
  min_s: number;
  max_s: number;
  /**
   * One tick of the container's time base. Intervals cannot be resolved more finely than this, so it is the tolerance below which a difference is rounding rather than a real rate change.
   */
  quantum_s: number;
  /**
   * Intervals differing from the median by more than one time-base tick.
   */
  irregular_count: number;
  irregular_fraction: number;
}
