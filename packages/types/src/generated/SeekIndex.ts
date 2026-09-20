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
 * DECODED_FRAMES  - presentation timestamps of the frames a decoder actually
 *                   emits, accounting for any edit list the container applies.
 *                   Authoritative: these are the times the frames are shown at,
 *                   whether or not they are evenly spaced.
 *
 *                   Deliberately not the container's *packet* timestamps. Those
 *                   are cheaper to read and are wrong on real recordings: a
 *                   clip with an MP4 edit list has packets a decoder never
 *                   emits, and packet times offset from presentation times.
 *
 * CONTAINER_RATE  - synthesised from the container's declared average frame
 *                   rate because no usable timestamps were found. A fallback,
 *                   and one that makes variable frame rate undetectable, so
 *                   anything derived from it is flagged.
 */
export type TimestampSource = "decoded_frames" | "container_rate";

/**
 * Every frame's presentation time, and the time to seek to in order to land on it.
 *
 * Named for what it is used for rather than for what it holds, because
 * `ingestion.probe.FrameIndex` already holds the raw version -- the timestamps
 * and the keyframe flags, as read. This is that, plus the one derived quantity
 * a player needs and nothing else in the system does.
 *
 * **A video element cannot be asked for a frame.** It is asked for a *time*,
 * and its decoder picks whichever frame is being displayed then. Every panel
 * this project has built since Phase 4 reports frame indices -- impact at
 * frame 46, the address baseline over frames 0 to 12 -- so putting a picture
 * behind those numbers needs a map from an index to a time that lands on it.
 *
 * `frame / fps` is not that map, and this is the same argument Phase 1 is
 * built on one layer up: on variable-rate footage the interval between frames
 * is not constant, so dividing by an average rate accumulates error until the
 * frame under the playhead is not the frame the panel is talking about. The
 * timestamps here are the presentation times of the frames a decoder actually
 * emits, which is what `probe` already reads and until now discarded at the
 * contract boundary.
 *
 * **`seek_targets_s` is not `timestamps_s`, and the difference is the point of
 * this contract.** Frame *i* is displayed from `timestamps_s[i]` until
 * `timestamps_s[i + 1]`, so its own timestamp is the instant the frame appears
 * -- a boundary, with frame *i - 1* on the other side of it. Asking for that
 * exact time is asking a decoder to resolve a tie, and which side it falls is
 * decided by rounding nobody controls: the container's rational time base, the
 * double the time is carried as, and the decoder's own comparison. The
 * midpoint of the interval is the furthest point from both boundaries, so it
 * is the target that survives all three, and that is what is stored here.
 *
 * Both lists are relative to the first frame, matching `PoseFrame.timestamp_s`
 * and every other time in this system. They are also the inverse map: a player
 * that observes the presentation time it actually landed on can look it up
 * here and report the frame it really showed, rather than the frame it asked
 * for. Those differ, and a player that assumed otherwise would be the third
 * place in this project where an unmeasured clock quietly biased a number.
 */
export interface SeekIndex {
  schema_version?: number;
  path: string;
  content_key: ContentKey;
  /**
   * Where the times came from. `container_rate` means they were synthesised, so seeking by them is a guess of exactly the kind this contract exists to avoid -- see `warnings`.
   */
  source: TimestampSource;
  frame_count: number;
  /**
   * Presentation time of each frame, relative to the first. Strictly increasing.
   */
  timestamps_s: number[];
  /**
   * Where to seek to in order to land on each frame: the midpoint of the interval that frame is displayed for.
   */
  seek_targets_s: number[];
  /**
   * How long the final frame is assumed to be displayed, since the container does not record it. Taken as the last observed interval, which is the same assumption `VideoTiming.duration_s` makes. None when the clip has fewer than two frames and there is no interval to observe.
   */
  last_interval_s?: number | null;
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
