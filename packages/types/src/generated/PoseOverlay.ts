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
 * The model's own landmark index; see `Landmark`.
 */
export type Landmark =
  | 0
  | 1
  | 2
  | 3
  | 4
  | 5
  | 6
  | 7
  | 8
  | 9
  | 10
  | 11
  | 12
  | 13
  | 14
  | 15
  | 16
  | 17
  | 18
  | 19
  | 20
  | 21
  | 22
  | 23
  | 24
  | 25
  | 26
  | 27
  | 28
  | 29
  | 30
  | 31
  | 32;
/**
 * How much the system actually knows about where a landmark is.
 *
 * Three values rather than a nullable position, because the two ways of not
 * knowing are different and only one of them is a problem with the recording.
 *
 * OBSERVED  - the estimator reported this landmark on this frame and the
 *             confidence gate kept it. The position is supported by an
 *             observation of this frame.
 * FILLED    - the position is supported, but not by an observation of this
 *             frame: either the gap policy bridged a short absence, or the
 *             local fit was carried by neighbouring frames. Drawn, and drawn
 *             differently, because a reader checking a metric against a frame
 *             should be able to see that this particular frame contributed
 *             nothing to it.
 * BLOCKED   - no position is supported here at all, so `x` and `y` are null.
 *             This is the state Phase 4 found on the down-the-line clip, where
 *             motion blur smeared the wrists for 1.92 s exactly when they were
 *             moving fastest. An overlay that interpolated across it would
 *             hide the one thing worth seeing.
 */
export type OverlayState = "observed" | "filled" | "blocked";

/**
 * Filtered landmarks over a range of frames, in the frame they are drawn in.
 *
 * A range rather than a clip, because the caller is a canvas: a 60 s clip at
 * 240 fps is 14,400 frames of 33 landmarks, and serialising all of it to draw
 * one of them is half a million points to move so that thirty-three can be
 * used. The range is also why `frames` carries its own indices rather than
 * being positional -- a consumer holding two ranges should not have to
 * remember where each one started.
 */
export interface PoseOverlay {
  schema_version?: number;
  video_path: string;
  content_key: ContentKey;
  geometry: FrameGeometry;
  start_frame: number;
  /**
   * Exclusive, as every frame range in this project is.
   */
  end_frame: number;
  frames: OverlayFrame[];
  /**
   * Which landmarks each frame carries, in the order they appear.
   */
  landmarks: Landmark[];
  /**
   * Skeleton edges to draw, from `POSE_CONNECTIONS`. Sent rather than left to the consumer so that the app cannot grow a second opinion about human anatomy; Phase 9's bone-length checks read the same tuple. Edges whose endpoints were not both requested are omitted.
   */
  connections: [Landmark, Landmark][];
  /**
   * The factor the timestamps were divided by. 1.0 for an ordinary recording.
   */
  slow_motion_factor: number;
  /**
   * Whether a lens correction was applied to these positions. When true they will not sit exactly on the raw pixels of the frame underneath, by up to the tens of pixels Phase 8 measured near the frame edge -- which is a real disagreement between the picture and the measurement, and belongs on screen rather than hidden.
   */
  undistorted: boolean;
  /**
   * Whether club tracking was asked for. False means every `shaft` is null because nothing looked, which is not the same as nothing being found.
   */
  club_tracked: boolean;
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
 * The displayed pixel size the coordinates are fractions of. Carried so a canvas can size itself, and so a caller can tell that a clip is portrait without probing the file again.
 */
export interface FrameGeometry {
  /**
   * Displayed frame width in pixels, after rotation.
   */
  width: number;
  /**
   * Displayed frame height in pixels, after rotation.
   */
  height: number;
}
/**
 * Everything drawn over one frame of the clip.
 */
export interface OverlayFrame {
  frame_index: number;
  /**
   * Real-clock time from the first frame, after any slow-motion factor has been divided out. On a slowed clip this is **not** where the frame sits in the video file, which is what the player seeks by -- see `SeekIndex`.
   */
  timestamp_s: number;
  /**
   * One entry per landmark asked for, in ascending landmark order.
   */
  points: OverlayPoint[];
  /**
   * The club, when it was tracked and found on this frame. Null otherwise.
   */
  shaft?: OverlayShaft | null;
}
/**
 * One landmark on one frame, ready to be drawn.
 *
 * `x` and `y` are null exactly when `state` is `blocked`. They are not NaN:
 * JSON has no NaN, and a serialiser that emitted one would produce a document
 * no strict parser accepts -- so the absence is carried in the type where a
 * TypeScript consumer has to handle it.
 */
export interface OverlayPoint {
  landmark: Landmark;
  /**
   * Fraction of the displayed width. Null when blocked.
   */
  x: number | null;
  /**
   * Fraction of the displayed height, measured downward. Null when blocked.
   */
  y: number | null;
  state: OverlayState;
  /**
   * What the estimator reported for this landmark on this frame, carried through unchanged -- including on frames the gate rejected. It is the estimator's own opinion of itself and this project has measured it being confidently wrong: see the Post-Phase 6 note, where both shoulders scored 1.00 across frames whose implied turn varied by 27 degrees. Shown, never trusted.
   */
  visibility: number;
}
/**
 * The club shaft on one frame, as a segment in drawing coordinates.
 *
 * The grip end is the hand anchor the search started from rather than
 * something the detector found, exactly as `ShaftObservation` records -- so a
 * reader watching the segment track the hands is watching a constraint that
 * was applied, not a measurement that was obtained.
 *
 * `reaches_head` is the field to draw differently on. False means the evidence
 * stopped before the end of the club, so the tip is where the image stopped
 * drawing a line and **not** the club head. Drawing the two the same way would
 * put a club head on screen in every frame where the club was blurred or
 * pointing at the camera, which is precisely where a reader would most want to
 * know that nothing was found.
 */
export interface OverlayShaft {
  grip_x: number;
  grip_y: number;
  tip_x: number;
  tip_y: number;
  /**
   * Whether `tip` is the club head or merely the end of the supported evidence.
   */
  reaches_head: boolean;
  confidence: number;
}
