/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * Where a measured value sat relative to a published band.
 *
 * `WITHIN` is a finding, not a silence. "This sits inside the range the source
 * reports" is an observation with evidence behind it, and a coaching engine that
 * only spoke when something was wrong would be reporting a selection rather than
 * a measurement.
 */
export type Comparison = "below" | "within" | "above";
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
 * How the source of a threshold measured the number it published.
 *
 * The field that decides which of this engine's measurements a borrowed number
 * may be compared against, because it decides what the number is *about*.
 *
 * THREE_D_MOTION_CAPTURE
 *     Optical or electromagnetic capture of markers in three dimensions. The
 *     quantity is an angle or a distance in space, so it may only be compared
 *     against a `SPATIAL` measurement -- which in this build means a stereo
 *     calibration and a triangulated pair.
 *
 * INSTRUMENTED_SENSOR
 *     A device worn on the body: a spine-mounted goniometer, an inertial unit,
 *     a pressure mat. Also a measurement of the body rather than of a picture
 *     of it, and so also `SPATIAL`.
 *
 * TWO_D_VIDEO
 *     Measured off video, which is what this engine has. A duration measured
 *     this way transfers directly, because a clock does not care where the
 *     camera stood. An *angle* measured this way transfers only to footage from
 *     the same camera position, which is why `filmed_from` exists and why a
 *     source that does not state it cannot supply an angular threshold.
 *
 * CONVENTION
 *     Published without a measurement protocol: instruction books, coaching
 *     received wisdom, the numbers on the back of a training aid. Permits no
 *     basis at all. See the module docstring.
 *
 * SAME_CLIP
 *     No borrowed number at all. The comparison is between two measurements of
 *     the same quantity, made the same way, by the same camera, on the same
 *     swing, and what has to be cleared is their own combined uncertainty. The
 *     only kind of threshold here that carries no population -- and therefore
 *     the only one that says nothing about anybody else's swing.
 *
 *     What survives a projection is the **direction** of the change and not its
 *     size: a foreshortening map is monotone, so a line that turned further
 *     really did turn further, and by how much is a statement about the picture.
 *     Rules using this say so in their own words.
 */
export type ThresholdMethod =
  "three_d_motion_capture" | "instrumented_sensor" | "two_d_video" | "convention" | "same_clip";
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
 * Why a rule that was considered produced nothing.
 *
 * Every one of these is a different thing to do about it, which is why they are
 * a typed set and not a sentence. `BASIS_NOT_PERMITTED` is fixed by a second
 * camera and a calibration; `NO_THRESHOLD` is fixed by somebody publishing one;
 * `UNRESOLVED` is fixed by a faster camera; `NO_METRIC` is usually fixed by
 * moving the one that filmed it.
 *
 * `SUPPLIED_TIMEBASE` is the one that cannot be fixed by anybody holding the
 * camera afterwards. A slow-motion clip's playback factor is not recorded in
 * the file; it is supplied by whoever ran the analysis, and every duration
 * measured from that clip is that supplied number multiplied by something
 * measured. Comparing one against a published band in seconds compares a
 * stopwatch against a guess. A **ratio** of two durations from the same clip
 * survives it, because the factor divides out -- the same shape of argument
 * that makes a length in torso lengths survive an unknown camera, one dimension
 * over.
 */
export type FindingRefusal =
  | "no_swing"
  | "no_metric"
  | "no_threshold"
  | "basis_not_permitted"
  | "view_mismatch"
  | "supplied_timebase"
  | "low_confidence"
  | "no_uncertainty"
  | "unresolved";
/**
 * Where a finding's words come from.
 *
 * OFF
 *     The rule's own sentence, assembled from the evidence. The default, and
 *     the only mode that needs nothing installed.
 * LOCAL
 *     A language model running on this machine, handed the findings as
 *     structured data and nothing else. Never the video, never the landmarks,
 *     never a file path.
 */
export type PhrasingMode = "off" | "local";

/**
 * Everything the coaching engine concluded about one swing.
 *
 * `computed` is false when there was no swing to reason about, and then both
 * lists are empty rather than populated from whatever the metrics layer managed
 * to produce anyway.
 */
export interface CoachingReport {
  schema_version?: number;
  computed: boolean;
  findings?: Finding[];
  refused?: RefusedFinding[];
  /**
   * How many rules were evaluated, whatever came out of them.
   */
  rules_considered?: number;
  /**
   * The camera position the metrics were measured from. Decides several rules.
   */
  view?: CameraView & string;
  /**
   * The clip's own clock resolution, in real seconds, as the swing events imply it. Every duration comparison's bracket comes from this, so a clip that did not supply one has its timing rules refused rather than compared against a threshold it cannot resolve.
   */
  frame_interval_s?: number | null;
  phrasing?: PhrasingReport;
  config?: CoachingConfig;
  warnings?: string[];
}
/**
 * One conclusion, with everything needed to check it.
 *
 * There is no severity and no score. A number that ranked findings against each
 * other would be the single most quoted output of this system and the least
 * defensible one: it would need a scale relating degrees of shoulder turn to
 * seconds of tempo, and nothing has measured one.
 */
export interface Finding {
  /**
   * Stable identifier, so a UI can link to one finding.
   */
  rule_id: string;
  title: string;
  /**
   * What was measured and how it compares, in words. Contains no number that is not in this finding's own evidence -- enforced by the same guard the phrasing layer is held to.
   */
  observation: string;
  comparison: Comparison;
  value: number;
  unit: MetricUnit;
  /**
   * The uncertainty the comparison had to clear, in the metric's unit. A value inside `bracket` of a band edge is refused as `UNRESOLVED` rather than reported as being on one side of it.
   */
  bracket: number;
  band_low?: number | null;
  band_high?: number | null;
  /**
   * How far outside the nearer band edge the value sat, in its own unit. Zero for a `WITHIN` finding.
   */
  margin: number;
  source: ThresholdSource;
  /**
   * At least one, always. A finding with no evidence cannot be produced by this engine -- the contract refuses it rather than the review.
   *
   * @minItems 1
   */
  evidence: [Evidence, ...Evidence[]];
  /**
   * The weakest confidence among the metrics this finding rests on.
   */
  confidence: number;
  /**
   * The phrasing layer's rewording of `observation`, present only when a model produced one **and** it passed the guard. `observation` is always there and is always what was measured; this is presentation.
   */
  phrased?: string | null;
}
/**
 * Where a threshold came from, and what it is therefore a threshold on.
 */
export interface ThresholdSource {
  /**
   * Author, work and year, as it would be written out.
   */
  citation: string;
  /**
   * Publication year, where there is one.
   */
  year?: number | null;
  /**
   * Who the number was measured on, in words. A threshold set on tour professionals says nothing about whether an amateur's swing is working; it says how far it sits from theirs.
   */
  population: string;
  /**
   * How many people, where the source states it. None means it does not.
   */
  sample_size?: number | null;
  method: ThresholdMethod;
  /**
   * What the source actually measured, in its own terms -- not what this engine measures. The two being different is the normal case and the reason `permitted_bases` is not derived from the metric.
   */
  measures: string;
  /**
   * The bases of this engine's measurements that this number may be compared against. Empty is a valid and common answer: it means the source measured something nothing here produces, and every rule resting on it is refused rather than approximated.
   */
  permitted_bases?: MetricBasis[];
  /**
   * For a `TWO_D_VIDEO` source, the camera position it measured from. Required before any angular or image-plane threshold from it may be used, because a projected angle is a fact about a camera position as much as about a body. None on a source that states only durations, which no camera position changes.
   */
  filmed_from?: CameraView | null;
  /**
   * Anything a reader needs in order not to over-read the number.
   */
  note?: string;
}
/**
 * One measured quantity a finding rests on, with the frames to check it in.
 *
 * The link the exit criterion is about: metric -> frames -> overlay. `frames`
 * are frame indices into **this clip's video file**, so a player can be shown
 * the picture the number came from; `timestamps_s` are the matching instants on
 * a real clock, which on a slow-motion clip is a different thing entirely and
 * does not index the file.
 */
export interface Evidence {
  metric: MetricName;
  /**
   * The metric's own display label, including its anchor.
   */
  label: string;
  value: number;
  unit: MetricUnit;
  basis: MetricBasis;
  event?: SwingEvent | null;
  phase?: SwingPhase | null;
  /**
   * As the metric reported it, in the metric's own unit.
   */
  uncertainty?: number | null;
  confidence: number;
  /**
   * Every frame whose landmarks entered this value. Indices into the video.
   */
  frames: number[];
  /**
   * Real-clock instants for `frames`, where the detection supplied them. Empty rather than interpolated when it did not.
   */
  timestamps_s?: number[];
  /**
   * How the number was produced, carried from the metric.
   */
  methodology: string;
}
/**
 * A rule that was considered and deliberately produced nothing.
 *
 * Reported rather than omitted, for the reason `RefusedMetric` is: a rule that
 * did not fire because the swing was fine and a rule that could not be applied
 * at all look identical in a list of findings, and they mean opposite things.
 */
export interface RefusedFinding {
  rule_id: string;
  title: string;
  refusal: FindingRefusal;
  reason: string;
  source: ThresholdSource;
}
/**
 * What the optional language layer did, if anything.
 *
 * `attempted` is zero in the default configuration and that is the expected
 * state. Every count here is present so that "the model rephrased nothing" and
 * "the model was never asked" cannot be confused, which is the same distinction
 * `RefusedMetric` exists for one layer down.
 */
export interface PhrasingReport {
  mode?: PhrasingMode & string;
  /**
   * What produced the candidates. None when nothing did.
   */
  provider?: string | null;
  attempted?: number;
  accepted?: number;
  rejected?: number;
  rejections?: GuardRejection[];
  /**
   * Why no model was reached, when one was asked for. A phrasing layer that cannot be reached falls back to the rules' own sentences and says so; it never silently produces the default and calls it a model's work.
   */
  unavailable?: string | null;
}
/**
 * One reason a candidate phrasing was thrown away.
 *
 * Kept in the report rather than logged, because the rejection rate is the only
 * evidence anybody has about whether the phrasing layer is safe to turn on. A
 * model that is rewritten into silence nine times in ten is not a phrasing
 * layer, and the number that says so has to survive to the reader.
 */
export interface GuardRejection {
  rule_id: string;
  /**
   * Which check failed, in a few words.
   */
  offence: string;
  /**
   * The text that failed it.
   */
  token: string;
  /**
   * The full candidate, kept so it can be inspected.
   */
  candidate: string;
}
/**
 * Policy for the coaching engine.
 *
 * As everywhere else in this system, these are **structural bounds, not golf
 * norms**. Nothing here says what a good swing looks like; the thresholds that
 * do live on the rules, each with the citation it came from.
 */
export interface CoachingConfig {
  /**
   * Confidence below which a metric does not support a conclusion, even one the arithmetic would reach comfortably. Stated policy: it is roughly the level at which the anchor factor alone -- how sure Phase 4 is that this frame is the top at all -- stops supporting a statement about the top.
   */
  min_confidence?: number;
  /**
   * Off by default, and off is a complete configuration. Every finding already has a sentence; the model layer rewords, and a reworded finding is not a better-founded one.
   */
  phrasing?: PhrasingMode & string;
  /**
   * Base URL of a language model running on this machine. Only used when `phrasing` is `local`. There is no remote default and no hosted fallback: a finding is a measurement of a person's body, and the decision to send one anywhere is not a default.
   */
  phrasing_endpoint?: string | null;
  /**
   * Which local model to ask for, where the endpoint hosts several.
   */
  phrasing_model?: string | null;
  /**
   * How long to wait for the whole phrasing pass before giving up on it.
   */
  phrasing_timeout_s?: number;
}
