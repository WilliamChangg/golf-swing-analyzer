/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * Every quantity this engine knows how to measure.
 *
 * Named per *quantity*, not per quantity-and-instant: `SPINE_TILT` at the top
 * and at impact are the same measurement made twice, and they carry the same
 * unit, basis and methodology. Which instant a particular value belongs to is
 * on the metric's `event` field, so the set of names does not grow by
 * multiplication every time another anchor becomes interesting.
 */
export type MetricName =
  | "spine_tilt"
  | "left_knee_flex"
  | "right_knee_flex"
  | "hip_sway"
  | "head_sway"
  | "head_lift"
  | "shoulder_turn"
  | "pelvis_turn"
  | "x_factor"
  | "shoulder_tilt"
  | "pelvis_tilt"
  | "hand_depth"
  | "hand_path_length"
  | "peak_hand_speed"
  | "lead_arm_angle"
  | "trail_arm_angle"
  | "backswing_duration"
  | "downswing_duration"
  | "follow_through_duration"
  | "takeaway_to_impact"
  | "tempo_ratio";
/**
 * Which family a metric belongs to. Presentation, not semantics.
 */
export type MetricGroup = "posture" | "rotation" | "arms" | "timing";
/**
 * What a metric's value is in.
 *
 * `TORSO_LENGTHS` is the length unit throughout, rather than frame widths or
 * pixels. A distance in frame widths halves when the camera is moved twice as
 * far away; the same distance in the subject's own torso lengths does not. It
 * is not a metric unit and does not pretend to be one -- it is a ratio of two
 * measured image distances, which is exactly what survives an unknown camera.
 */
export type MetricUnit = "degrees" | "seconds" | "ratio" | "torso_lengths" | "torso_lengths_per_s";
/**
 * What kind of claim a metric's value is.
 *
 * TEMPORAL
 *     A duration, or a ratio of durations. Depends on the clock and nothing
 *     else, so it is unaffected by where the camera stood. The only basis in
 *     this phase that measures the body rather than a picture of it.
 *
 * IMAGE_PLANE
 *     A distance or a speed between two points as they appear in the frame,
 *     divided by the subject's torso length. Exact as a statement about the
 *     image; it under-reports any motion that ran towards or away from the
 *     camera, which it cannot see at all.
 *
 * PROJECTED_ANGLE
 *     The angle between two segments as they appear in the frame. Equal to the
 *     real joint angle only when both segments lie in the image plane, and
 *     smaller than it otherwise. Never a 3D joint angle.
 *
 * FORESHORTENED_ANGLE
 *     A rotation about the vertical inferred from how much a body segment
 *     shortened in the image: a shoulder line seen at 60% of its full width
 *     has turned about 53 degrees away from the camera. It assumes the segment
 *     was square to the camera at its reference frame, and it is blind to
 *     direction -- a turn and its mirror image shorten identically -- so the
 *     value is a magnitude.
 */
export type MetricBasis = "temporal" | "image_plane" | "projected_angle" | "foreshortened_angle";
/**
 * The instants that divide a swing.
 *
 * TAKEAWAY - the hands begin moving away from the ball.
 * TOP      - the backswing reverses; the hands are momentarily near rest.
 * IMPACT   - estimated from hand kinematics, not observed. See module docstring.
 * FINISH   - the hands come back to rest after the follow-through.
 */
export type SwingEvent = "takeaway" | "top" | "impact" | "finish";
/**
 * The intervals between events.
 */
export type SwingPhase = "address" | "backswing" | "downswing" | "follow_through";
/**
 * The camera view this was measured in. Carried on every metric because a projected quantity means different things from different places, and two clips of the same swing must never be compared across views as though the numbers described the same thing.
 */
export type CameraView = "face_on" | "down_the_line" | "unknown";
export type BodySide = "left" | "right";

/**
 * Everything the biomechanics engine concluded about one clip.
 *
 * `computed` is false when no swing was detected, and then `metrics` is empty
 * rather than populated from arbitrary frames. Metrics are anchored to swing
 * events; without events there is nothing to anchor them to, and measuring a
 * "spine tilt at the top" on a clip with no top would be inventing the instant
 * and then measuring it carefully.
 */
export interface MetricSet {
  schema_version?: number;
  computed: boolean;
  metrics?: Metric[];
  refused?: RefusedMetric[];
  view?: ViewEstimate | null;
  lead_side?: LeadSide | null;
  references?: RotationReference[];
  /**
   * Median shoulder-midpoint to hip-midpoint distance, in frame widths.
   */
  torso_length: number;
  geometry: FrameGeometry;
  frames: number;
  config: MetricConfig;
  warnings?: string[];
}
/**
 * One measured quantity, with everything needed to interpret it.
 */
export interface Metric {
  name: MetricName;
  group: MetricGroup;
  /**
   * Human-readable name, including the instant it was measured at.
   */
  label: string;
  value: number;
  unit: MetricUnit;
  basis: MetricBasis;
  /**
   * The instant this was measured at, for metrics taken at one.
   */
  event?: SwingEvent | null;
  /**
   * The interval this was measured over, for metrics spanning one.
   */
  phase?: SwingPhase | null;
  /**
   * Every frame whose landmarks entered this value, so it can be checked against the video rather than taken on trust.
   */
  source_frames: number[];
  view: CameraView;
  /**
   * What the number means anatomically from this view, in words. Separate from `methodology`, which says how it was computed: the arithmetic is the same from every camera position and the meaning is not.
   */
  interpretation: string;
  confidence: MetricConfidence;
  /**
   * How this number was produced, in words. Never omitted.
   */
  methodology: string;
}
/**
 * Why a metric is or is not trustworthy, decomposed.
 *
 * Three factors and their product, matching how Phase 4 reports an event's
 * confidence -- for the same reason. A single number says a metric is worth
 * 0.3 without saying whether the fix is a better camera angle, a brighter
 * room, or a faster shutter.
 */
export interface MetricConfidence {
  /**
   * Product of the three factors below.
   */
  overall: number;
  /**
   * Mean reported visibility of the landmarks this metric used, over the frames it used them on.
   */
  observation: number;
  /**
   * Confidence of the swing event or phase this metric is measured at, propagated from phase detection. A perfectly measured angle at an instant that is not really the top of the backswing is not a measurement of anything.
   */
  anchor: number;
  /**
   * How well the method itself pins the quantity down, measured rather than assumed. For a duration, how finely the frame rate divides it. For a projected angle, the fraction of the segment lying in the image plane -- a segment seen nearly end-on subtends a noisy angle. For a foreshortened rotation, the sine of the angle found, which is how sharply the arccos converts a length into an angle there.
   */
  method: number;
}
/**
 * A metric that was asked for and deliberately not produced.
 *
 * Reported rather than silently omitted. An absent metric and a metric that
 * could not honestly be computed look identical in a list of results, and they
 * call for different responses: one is a gap in the engine, the other is a fact
 * about the recording that the person holding the camera can fix.
 */
export interface RefusedMetric {
  name: MetricName;
  event?: SwingEvent | null;
  reason: string;
}
/**
 * Which view a clip was shot from, and the measurements behind the verdict.
 *
 * Decided from the **projected width of the shoulder line at address**, in
 * torso lengths. Seen face-on the shoulders lie broadside and span most of a
 * torso length or more; seen down the line they point at the camera and
 * collapse to almost nothing. On the reference clips the two are 0.83 and 0.10,
 * which is a margin of eight times rather than a close call.
 *
 * `openness` corroborates it from an independent direction, the way Phase 4
 * corroborates impact: the address span divided by the widest the line was ever
 * seen in the clip is the cosine of how far off broadside it was at address. A
 * swing turns the shoulders through about a right angle, so both views contain
 * a frame where the line is nearly square, and the two signals should agree.
 * They are kept separate because the second depends on the clip containing that
 * frame and the first does not.
 */
export interface ViewEstimate {
  view: CameraView;
  /**
   * How far clear of ambiguity the measurement sits, as a fraction of the band between the two thresholds. 1.0 is a full band clear.
   */
  confidence: number;
  /**
   * Projected shoulder width at address, in torso lengths. The verdict.
   */
  shoulder_span_ratio: number;
  /**
   * Projected hip width at address, in torso lengths. Reported, not used.
   */
  hip_span_ratio: number;
  /**
   * Address shoulder span divided by the widest in the clip: the cosine of how far off broadside the shoulders were at address. Corroboration.
   */
  openness: number;
  /**
   * The address frames the spans were measured over.
   */
  frames?: number[];
  methodology: string;
}
/**
 * Which side of the body leads the swing, and how that was decided.
 *
 * Needed because "lead arm" and "trail arm" are not "left arm" and "right
 * arm": they swap between a right- and a left-handed player, and nothing in a
 * pose sequence declares which is which.
 *
 * It is inferred from a measurement rather than configured: at the top of the
 * backswing the hands sit over the **trail** shoulder, for either handedness
 * and from either side of the player. Projecting the hand position onto the
 * shoulder line therefore names the trail side directly, and the size of that
 * projection says how confidently.
 */
export interface LeadSide {
  /**
   * The leading side, or None when the evidence was too weak to name one.
   */
  side: BodySide | null;
  /**
   * How far the hands sat towards one shoulder at the top, as a fraction of half the shoulder span. 1.0 means directly over a shoulder, 0.0 means over the middle of the chest and therefore undecidable.
   */
  margin: number;
  /**
   * Projected shoulder span at the top, in torso lengths. A down-the-line camera sees the shoulders nearly end-on, which makes this small and the side undecidable -- correctly, because that view genuinely does not show it.
   */
  shoulder_span_ratio: number;
  methodology: string;
}
/**
 * The baseline a foreshortening rotation is measured against, and its check.
 *
 * A rotation here is a comparison of a body line's projected length against its
 * length when square to the camera. That baseline is taken **at address**,
 * because that is where the method's premise says the player is square: they
 * are set up to the ball with the camera in front of them.
 *
 * Taking it instead as the widest view anywhere in the clip was tried and is
 * worse, for a reason the reference footage demonstrated. A maximum over a
 * whole clip is a maximum over that clip's landmark noise too, so it selects
 * the single frame where the estimator most overstated the span -- on the
 * face-on reference clip, a frame just past impact where the shoulders read 12%
 * wider than the player can physically be. Every rotation in the clip is then
 * measured against an error.
 *
 * The premise is still checked, by the fields below: if some frame mid-swing
 * projects substantially wider than address did, the player was not square
 * there, and this is not a face-on recording.
 */
export interface RotationReference {
  /**
   * Which segment: 'shoulders' or 'hips'.
   */
  landmarks: string;
  /**
   * Median projected length over the address phase, in torso lengths.
   */
  span: number;
  /**
   * The address frames the baseline was taken over.
   */
  frames: number[];
  /**
   * Largest projected length anywhere in the clip, in torso lengths.
   */
  widest_span: number;
  widest_frame: number;
  /**
   * widest_span / span. At most slightly above 1 on a face-on recording, where address is the squarest the line ever is. Far above it when the line is seen end-on at address and opens up through the swing, which is what a down-the-line camera sees.
   */
  excess: number;
  /**
   * Whether `excess` is within the configured tolerance.
   */
  square_at_address: boolean;
}
/**
 * The displayed pixel dimensions IMAGE landmarks were normalised against.
 *
 * Carried with the landmarks because IMAGE space is **anisotropic** and
 * nothing downstream can discover that on its own. x is divided by the frame
 * width and y by the frame height, so on a 1080x1920 clip one pixel of
 * vertical travel becomes 1/1920 while one pixel of horizontal travel becomes
 * 1/1080: the same displacement in pixels counts for 0.5625 as much going down
 * as going across. Any Euclidean quantity that mixes the two -- a distance, a
 * speed, an angle -- is wrong by an amount that depends only on the shape of
 * the frame, and nothing about the result looks wrong.
 *
 * One number fixes it, and it is not recoverable from the landmarks: the
 * aspect ratio. It is recorded here, at the point where it is still known,
 * rather than re-derived later by probing a video file that may have moved.
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
 * Policy for the biomechanics engine.
 *
 * As in Phase 4, these are **structural bounds, not golf norms.** Nothing here
 * states what a good swing looks like. They exist to decide when a measurement
 * is too ill-conditioned to report at all, and the thresholds are stated policy
 * rather than measured optima, because no labelled set exists until Phase 12.
 */
export interface MetricConfig {
  /**
   * Projected shoulder span at the top, in torso lengths, below which the shoulder line is too foreshortened to read a direction from -- which is what naming the lead side depends on. Measured at the top rather than at address, and so a different question from the one the view thresholds answer: a down-the-line clip has the shoulders nearly square at the top even though they were end-on at address.
   */
  min_shoulder_span_ratio?: number;
  /**
   * How far towards one shoulder the hands must sit at the top before the lead side is named. Below it, no side is reported and the metrics that need one are refused rather than assigned to a coin flip.
   */
  min_lead_side_margin?: number;
  /**
   * Projected shoulder width at address, in torso lengths, at or above which the camera is treated as face-on. The basis is anatomical rather than measured here: an adult's shoulder width is a fairly stable multiple of the distance from their shoulders to their hips, so a shoulder line spanning more than half of it cannot be pointing at the camera. The reference face-on clip measures 0.83.
   */
  face_on_span_ratio?: number;
  /**
   * Projected shoulder width at address, in torso lengths, at or below which the camera is treated as down-the-line. The reference down-the-line clip measures 0.10. Between this and `face_on_span_ratio` the view is reported as unknown rather than rounded to the nearer label -- an oblique camera is a real thing to have recorded, and it supports some measurements and not others.
   */
  down_the_line_span_ratio?: number;
  /**
   * How much wider than its address baseline a body line may project somewhere in the clip before the recording is judged not to be face-on and its rotations refused. Above 1 because landmark noise alone moves the maximum a little past address -- it reaches 1.12 on the face-on reference clip. A down-the-line recording, where the shoulders start end-on and open through the backswing, exceeds it by a wide margin rather than a marginal one.
   */
  max_reference_excess?: number;
  /**
   * Torso length, in frame widths, below which the subject is too small or too poorly tracked to divide by. A guard against producing an enormous ratio from a torso that collapsed to a point.
   */
  min_torso_length?: number;
}
