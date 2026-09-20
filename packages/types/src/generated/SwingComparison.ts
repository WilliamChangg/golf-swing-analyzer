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
 * Where the camera stood relative to the player.
 *
 * The single fact that decides what a projected measurement *means*. The same
 * spine tilt is lateral side bend seen face-on and forward posture angle seen
 * down the line; the same hip displacement is a slide towards the target in one
 * and a move towards the ball in the other. Nothing in the arithmetic
 * distinguishes them, so the view is measured and carried on every metric, and
 * the anatomical reading is stated per view rather than assumed.
 *
 * FACE_ON
 *     Perpendicular to the target line, looking at the player. The shoulder
 *     line lies across the frame, so rotation is measurable by foreshortening
 *     and left and right are distinguishable.
 * DOWN_THE_LINE
 *     Along the target line. The shoulder line points towards the camera and
 *     collapses, so rotation about the spine is not recoverable, and the
 *     frame's horizontal axis runs towards and away from the ball.
 *
 *     **Which end of the target line is not determined.** A camera behind the
 *     player and one in front of them foreshorten the shoulder line
 *     identically, and nothing else here separates them. The measurable
 *     consequences are the same either way, which is why one label covers
 *     both; what it costs is the sign of anything measured along the frame's
 *     horizontal axis, so those quantities are reported as image directions
 *     rather than as "towards the player" or "away from them". The reference
 *     clip `data/amateur/dtl/iron_dtl.mp4` is filmed from in front, despite its name.
 * UNKNOWN
 *     Oblique, or too little of the body tracked at address to tell. Not a
 *     failure: an oblique camera genuinely supports some measurements and not
 *     others, and saying so beats picking the nearer label.
 */
export type CameraView = "face_on" | "down_the_line" | "unknown";
export type BodySide = "left" | "right";
/**
 * What this build may claim about the geometry of a recording.
 *
 * The honesty gate of Phases 8 and 9, as a type. It is ordered, and
 * `at_least` is how a consumer asks the question it actually has -- "may I
 * triangulate" -- rather than enumerating the values that permit it and
 * forgetting one when a fourth is added.
 *
 * NONE
 *     No calibration. Every measurement is a statement about the image plane,
 *     which is what Phases 5 and 6 produce and label as such. The lens's
 *     distortion is present in every landmark and unmeasured.
 * INTRINSICS
 *     One camera's focal lengths, optical centre and distortion are measured.
 *     Landmarks can be undistorted, so projected measurements improve, and a
 *     pixel becomes a known direction. **Still no depth, so still no
 *     metric-scale 3D claim.**
 * STEREO
 *     Both cameras are calibrated and their relative pose is measured, so two
 *     views of one instant give an intersection rather than two directions.
 *     This is what Phase 9 triangulates with.
 */
export type CalibrationStatus = "none" | "intrinsics" | "stereo";
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
  | "shoulder_turn_3d"
  | "pelvis_turn_3d"
  | "x_factor_3d"
  | "lead_arm_angle_3d"
  | "trail_arm_angle_3d"
  | "peak_hand_speed_3d"
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
export type MetricUnit = "degrees" | "seconds" | "ratio" | "torso_lengths" | "torso_lengths_per_s" | "metres_per_s";
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
 *
 * SPATIAL
 *     A length, angle or speed between points **reconstructed in three
 *     dimensions** from two calibrated views of the same instant. The only
 *     basis here that is a statement about the body rather than about a
 *     picture of it, and the reason every other one is named the way it is.
 *
 *     It is also the only basis whose meaning does not depend on the camera
 *     view: a distance between two 3D points is the same distance from
 *     anywhere, so a `SPATIAL` metric carries one anatomical reading instead
 *     of one per view. What it does depend on is the *reconstruction*, whose
 *     own error is reported per point and propagated into the metric's
 *     `uncertainty` -- a 3D number is not automatically a better number, it is
 *     a differently-conditioned one, and the conditioning is the ray
 *     convergence angle rather than the camera position.
 */
export type MetricBasis = "temporal" | "image_plane" | "projected_angle" | "foreshortened_angle" | "spatial";
/**
 * Which way a difference went. Two members, and both are deliberate.
 *
 * HIGHER  - the target's value is larger than the reference's.
 * LOWER   - the target's value is smaller.
 *
 * There is no third member. "The same" is not a verdict this system can reach:
 * two values closer together than the pair can resolve produce
 * `DifferenceRefusal.UNRESOLVED`, which says the recordings could not tell them
 * apart rather than that the swings agreed. And there is no "better": see the
 * module docstring.
 */
export type Direction = "higher" | "lower";
/**
 * Why a quantity present in both clips was not compared.
 *
 * Each is a different thing to do about it, which is why they are a typed set
 * rather than a sentence -- the same argument `FindingRefusal` makes one layer
 * across.
 *
 * NO_SWING
 *     One of the clips contains no detected swing, so it has no metrics and no
 *     clock to normalise onto.
 * MISSING
 *     The quantity was produced for one clip and not the other. Usually a fact
 *     about the recording that did not produce it, and its own `MetricSet`
 *     says which.
 * VIEW_MISMATCH
 *     The two clips were filmed from different camera positions. A projected
 *     angle is a fact about a camera position as much as about a body, so the
 *     two numbers are not measurements of the same quantity. Durations survive
 *     this; a stopwatch does not care where the camera stood.
 * CAMERA_MOVED
 *     Both clips were filmed from the same *class* of position and not from the
 *     same position. Measured from the address shoulder span, which is what the
 *     view detector reads and what every foreshortening rotation is referred
 *     to. See `CameraAgreement`.
 * CALIBRATION_MISMATCH
 *     One clip's landmarks had a lens correction applied and the other's did
 *     not. Phase 8 measured that correction moving peak hand speed by 3.0% and
 *     a rotation by up to 1.2 degrees on real footage, which is a systematic
 *     difference between the two recordings and not between the two swings.
 * SUPPLIED_TIMEBASE
 *     At least one clip carries a slow-motion factor, which is supplied rather
 *     than measured. A duration from such a clip is a measured number
 *     multiplied by a guess, and a difference of two of them is a difference of
 *     two guesses. A **ratio** of two durations from one clip divides it out
 *     and survives, which is the same gate Phase 13 applies.
 * LOW_CONFIDENCE
 *     One of the two measurements does not support a conclusion on its own, so
 *     a difference of the two supports less.
 * NO_BRACKET
 *     Neither clip's measurement carries an uncertainty, and the quantity is
 *     not one whose bracket can be derived from the clock. There is then no
 *     distance a difference could be required to clear, and one invented here
 *     would decide the answer.
 * UNRESOLVED
 *     The difference is real arithmetic and smaller than what the two
 *     recordings can resolve. The most common outcome, and the one worth
 *     reading: it names a difference that a faster camera or a steadier
 *     landmark would settle.
 */
export type DifferenceRefusal =
  | "no_swing"
  | "missing"
  | "view_mismatch"
  | "camera_moved"
  | "calibration_mismatch"
  | "supplied_timebase"
  | "low_confidence"
  | "no_bracket"
  | "unresolved";
/**
 * A scalar signal through the swing, comparable once both clips are normalised.
 *
 * These are the signals Phase 4 reads the swing's events off, carried up rather
 * than recomputed, so that a curve in this report and the timeline beside it
 * cannot come from two different readings of the same clip.
 *
 * HAND_SPEED and HAND_HEIGHT are divided by each subject's own torso length, so
 * they survive a change of framing and are dimensionless ratios rather than
 * speeds and distances. SHOULDER_ANGLE and HIP_ANGLE are projected into the
 * image plane and are comparable only between clips filmed from the same place.
 */
export type TrajectoryChannel = "hand_speed" | "hand_height" | "shoulder_angle" | "hip_angle";

/**
 * Everything this system concluded about two recordings, side by side.
 *
 * `computed` is false when either clip produced no swing, and then every list
 * is empty rather than populated from whichever half worked.
 *
 * **Two recordings, and not necessarily two swings by one player.** Nothing
 * here can tell whether the same person made both, and nothing here asks: the
 * lengths are in each subject's own torso lengths and the durations are on each
 * clip's own clock, which is what makes a comparison arithmetically meaningful
 * across subjects and framings. What it does not make it is a comparison of
 * like with like -- two people are not two attempts -- and that judgement
 * belongs to whoever chose the two files.
 */
export interface SwingComparison {
  schema_version?: number;
  computed: boolean;
  reference: ClipSummary;
  target: ClipSummary;
  camera: CameraAgreement;
  differences?: MetricDifference[];
  refused?: RefusedDifference[];
  trajectories?: TrajectoryComparison[];
  hand_path?: HandPathOverlay | null;
  /**
   * Quantities examined, whatever came out of them.
   */
  metrics_considered?: number;
  config?: ComparisonConfig;
  warnings?: string[];
}
/**
 * What one side of the comparison is, and what it is entitled to claim.
 *
 * Every field here is something that can differ between two recordings without
 * either swing differing, which is why they are on the report rather than left
 * to the caller to remember: a reader looking at a difference needs to be able
 * to see that one clip was undistorted and the other was not.
 */
export interface ClipSummary {
  path: string;
  /**
   * Identifies the recording by content. Two comparisons of the same pair are the same comparison; two clips with the same filename are not necessarily the same clip.
   */
  content_key: ContentKey;
  frames: number;
  view: CameraView;
  lead_side?: BodySide | null;
  calibration: CalibrationStatus;
  slow_motion_factor: number;
  /**
   * Median shoulder-midpoint to hip-midpoint distance, in frame widths. Null when the clip never tracked a torso, which is what a clip with no detected swing looks like. Every length and speed in this report is divided by it, so it is reported rather than hidden -- it is the one number that decides whether two clips of differently-framed subjects are comparable at all.
   */
  torso_length?: number | null;
  clock: PhaseClock;
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
 * The map from one clip's clock onto swing position.
 *
 * Piecewise linear, with a knot at each of the four events. Linear *in time*
 * within each phase rather than in frame number, because the footage may be
 * variable-rate and because a slow-motion factor has already been divided out
 * of the timestamps.
 *
 * A clip missing any of the four events has no clock: three knots cannot
 * normalise the phase they do not bound, and extrapolating one would invent the
 * instant that decides where every sample lands.
 */
export interface PhaseClock {
  /**
   * Four, in swing order, or empty when the clip could not be normalised.
   */
  knots: ClockKnot[];
  /**
   * Whether all four events were located.
   */
  usable: boolean;
  /**
   * The clip's clock resolution in real seconds, as its own events imply it. The floor under every ambiguity above.
   */
  frame_interval_s?: number | null;
  /**
   * The factor the timestamps were divided by, as supplied. Carried because a duration read off this clock is a measured number multiplied by it.
   */
  slow_motion_factor?: number;
  /**
   * How the map was built, in words. Never omitted.
   */
  methodology: string;
}
/**
 * One event, and how well the clip places it.
 *
 * `ambiguity_s` is the half-width of where the true instant could be, in real
 * seconds. It starts at one frame interval -- the clip's own clock resolution,
 * taken from the events rather than from a declared frame rate, for the reasons
 * `coaching.bracket` gives -- and is widened where an independent estimate of
 * the same instant disagrees by more. Phase 4 corroborates impact with the low
 * point of the hand arc; where two estimates of one instant sit 40 ms apart,
 * 40 ms is the honest ambiguity and one frame is not.
 */
export interface ClockKnot {
  event: SwingEvent;
  /**
   * Where this event sits on the normalised axis.
   */
  position: number;
  frame_index: number;
  /**
   * Real-clock instant, after any slow-motion factor has been divided out. On a slowed clip this does not index the video file.
   */
  timestamp_s: number;
  /**
   * Phase 4's own, carried through.
   */
  confidence: number;
  /**
   * Half-width of where this instant could be, in real seconds.
   */
  ambiguity_s: number;
  /**
   * What set the ambiguity, in words.
   */
  ambiguity_source: string;
}
/**
 * Whether the two clips were filmed from the same place, and how closely.
 *
 * **The measurement this phase exists for.** Every projected quantity is a fact
 * about a camera position as much as about a body, and the view detector's
 * three labels are far too coarse to decide whether two clips share one: a
 * camera can move twenty degrees round a player and stay comfortably inside
 * `face_on`.
 *
 * The evidence is the projected shoulder span at address, in torso lengths --
 * the same number `ViewEstimate` decides the view from, and the same number
 * every foreshortening rotation is referred to. `openness` divides it by the
 * widest the line was ever seen in that clip, which makes it the cosine of how
 * far off broadside the shoulders were at address; the arccosine of that is an
 * estimate of the camera's azimuth.
 *
 * **It is an estimate with two real weaknesses, stated rather than buried.** It
 * assumes both clips show the same body, because a wider-shouldered player
 * projects a wider line from the same place; and it depends on each clip
 * containing a frame where the shoulders come square, which Phase 5 recorded
 * landmark noise overstating by 12% on the reference footage. It is used to
 * refuse and to widen a bracket, never to correct a value.
 */
export interface CameraAgreement {
  /**
   * Projected shoulder span at address, in torso lengths. Null when that clip produced no view estimate, which is what a clip with no detected swing does -- there is then no address phase to measure it over.
   */
  reference_span?: number | null;
  target_span?: number | null;
  /**
   * Absolute difference of the two spans, divided by their mean. Null when either span is, and **not** a large number standing in for unknown: the gate below reads this, and an infinity here would refuse with a reason that sounded like a measurement.
   */
  span_disagreement?: number | null;
  reference_openness?: number | null;
  target_openness?: number | null;
  /**
   * How far off broadside this camera stood, in degrees, from the arccosine of `openness`. Null when the clip supplied no usable openness -- which is not a claim that the camera was square.
   *
   * **Reported and bracketed with, never gated on.** The arccosine is flat near broadside, so it turns a one-percent error in `openness` into eight degrees of azimuth; a gate on this number would refuse every pair ever filmed. The gate is `span_disagreement`, which is the measurement rather than a function of it.
   */
  reference_azimuth_deg?: number | null;
  target_azimuth_deg?: number | null;
  /**
   * The furthest apart the two cameras can have been, in degrees: the **sum** of the two azimuths, not their difference. Both are unsigned, because foreshortening is identical from either side of broadside and nothing here separates them -- so two cameras each five degrees off may have been in the same place or ten degrees apart, and a bracket has to assume the second.
   */
  azimuth_separation_deg?: number | null;
  /**
   * Whether the two cameras are close enough together for a projected quantity to be compared at all. Decided by the view class and by `span_disagreement`; false refuses every projected, image-plane and foreshortened difference by name.
   */
  consistent: boolean;
  methodology: string;
}
/**
 * One quantity measured in both clips, and how far apart the two came out.
 *
 * Carries no severity, no score and no judgement of which value is preferable.
 * See the module docstring; a test asserts the absence of those field names.
 */
export interface MetricDifference {
  name: MetricName;
  /**
   * The metric's own display label, including its anchor.
   */
  label: string;
  group: MetricGroup;
  unit: MetricUnit;
  basis: MetricBasis;
  event?: SwingEvent | null;
  reference_value: number;
  target_value: number;
  /**
   * target minus reference, in the metric's own unit.
   */
  difference: number;
  direction: Direction;
  /**
   * What the difference had to clear, in the metric's unit. The **sum** of the terms below rather than their quadrature, which is where this parts company with `coaching.combined_bracket`: two anchors in one clip share a systematic error that largely cancels in their difference, and two separate recordings share nothing. A sum of bounds is a bound; a quadrature of bounds is not.
   */
  bracket: number;
  /**
   * Where the bracket came from, itemised.
   */
  terms: BracketTerm[];
  /**
   * How far the difference sits outside the bracket, in the metric's unit. Always positive: a difference that does not clear its bracket is refused as `unresolved` rather than reported with a margin of zero.
   */
  margin: number;
  /**
   * The weaker of the two measurements' confidences.
   */
  confidence: number;
  /**
   * Frames the reference clip's value was measured on. Indices into its video.
   */
  reference_frames: number[];
  /**
   * The same, for the target clip's video.
   */
  target_frames: number[];
  /**
   * What the difference means anatomically from this view, in words.
   */
  interpretation: string;
  /**
   * How both numbers were produced. Never omitted.
   */
  methodology: string;
}
/**
 * One named contribution to what a difference had to clear.
 *
 * Itemised rather than summed into a single number, because the three sources
 * call for completely different responses: a clock term is fixed by a faster
 * camera, a measurement term by a steadier landmark, and a camera term by
 * putting the tripod back where it was.
 */
export interface BracketTerm {
  /**
   * What produced this term, in a few words.
   */
  source: string;
  /**
   * Its size, in the metric's own unit.
   */
  value: number;
  reason: string;
}
/**
 * A quantity that could have been compared and deliberately was not.
 *
 * Reported rather than omitted, for the reason `RefusedMetric` and
 * `RefusedFinding` are: a quantity that did not differ and a quantity that
 * could not be compared look identical in a list of differences, and they mean
 * opposite things.
 */
export interface RefusedDifference {
  name: MetricName;
  label: string;
  event?: SwingEvent | null;
  refusal: DifferenceRefusal;
  reason: string;
  /**
   * The arithmetic, where both values exist. Present on an `unresolved` refusal precisely because the numbers are not secret -- what is refused is the claim that they differ, not the values themselves.
   */
  reference_value?: number | null;
  target_value?: number | null;
  bracket?: number | null;
}
/**
 * One signal from both clips, on one axis, with the difference bracketed.
 */
export interface TrajectoryComparison {
  channel: TrajectoryChannel;
  label: string;
  unit: MetricUnit;
  basis: MetricBasis;
  samples?: ChannelSample[];
  /**
   * Fraction of the sampled positions where the two curves differ by more than the pair can resolve. A count of where they differ, and deliberately not a distance between them: a single number summarising how far apart two swings are is the score this report does not have.
   */
  resolved_fraction: number;
  /**
   * Biggest resolved difference, in the channel's unit.
   */
  largest_difference?: number | null;
  /**
   * Where on the axis that sat.
   */
  largest_at?: number | null;
  /**
   * Set when this channel was not comparable at all.
   */
  refusal?: DifferenceRefusal | null;
  /**
   * Why, when `refusal` is set.
   */
  reason?: string;
}
/**
 * Both clips' value at one position on the normalised axis, and their difference.
 *
 * One object rather than parallel arrays, so that "null exactly where nothing
 * is supported" survives into TypeScript instead of becoming a check every
 * consumer writes separately. Same argument Phase 15 made for a scene point.
 */
export interface ChannelSample {
  position: number;
  /**
   * Null where the reference clip supports no value.
   */
  reference: number | null;
  target?: number | null;
  /**
   * target minus reference. Null unless both sides have a value.
   */
  difference?: number | null;
  /**
   * What a difference here would have to clear: each clip's event ambiguity, carried onto this axis and multiplied by the local slope of its own curve, then summed. Large where the curve is steep, which is exactly where two curves most easily appear to differ.
   */
  bracket?: number | null;
  /**
   * Whether `difference` clears `bracket`. False where either is null.
   */
  resolved?: boolean;
}
/**
 * Both clips' hand arcs, resampled onto the shared axis.
 *
 * The one part of this report that is a picture rather than a number, and it
 * carries no bracket: a bracket on a two-dimensional path would be an ellipse
 * per sample, and nothing here has measured the second axis of one. It is drawn
 * as evidence to look at, alongside numbers that are bracketed, and the
 * contract says so rather than leaving a reader to assume the arcs are as well
 * determined as the curves beside them.
 */
export interface HandPathOverlay {
  samples?: PathSample[];
  unit?: MetricUnit & string;
  /**
   * False when the two clips' views differ, which makes the arcs two shapes.
   */
  comparable: boolean;
  reason?: string;
}
/**
 * Where each clip's hands were at one position on the normalised axis.
 *
 * In torso lengths from that clip's own address hand position, so two subjects
 * of different sizes filmed from different distances are on one scale. `x` is
 * along the image's horizontal and `y` is upward -- an image direction, not an
 * anatomical one, because a down-the-line camera may be in front of the player
 * or behind them and nothing here separates those.
 */
export interface PathSample {
  position: number;
  reference_x?: number | null;
  reference_y?: number | null;
  target_x?: number | null;
  target_y?: number | null;
}
/**
 * Policy for the comparison layer.
 *
 * **Structural bounds, not golf norms**, as everywhere else here. Nothing in
 * this file says what a good swing looks like, and the two camera tolerances
 * are the only numbers with a measurement behind them rather than a statement
 * of policy.
 */
export interface ComparisonConfig {
  /**
   * Points on the normalised axis, spanning takeaway to finish. 121 puts one every fortieth of a phase, which is finer than the frame rate of any clip this project has and coarse enough that the payload stays small. Nothing is interpolated beyond the clip's own samples: a position whose neighbours are unsupported comes back null.
   */
  samples?: number;
  /**
   * Confidence below which a measurement does not support a comparison. Stated policy, and deliberately the same number `CoachingConfig` uses: a measurement good enough to compare against a published band is good enough to compare against another measurement.
   */
  min_confidence?: number;
  /**
   * How far apart the two clips' address shoulder spans may sit, as a fraction of their mean, before the cameras are judged to have moved and every projected difference is refused.
   *
   * **The one threshold here with a measurement behind it.** `scripts/benchmark_compare.py --sweep camera` films one unchanged swing from a series of azimuths at the landmark noise Phase 3 measured on real footage, and reports both what the span does and what the metrics do. The gate is on the span rather than on the azimuth derived from it because the span is the measurement: the arccosine that turns it into an angle is flat near broadside and amplifies a percent into eight degrees.
   */
  max_span_disagreement?: number;
}
