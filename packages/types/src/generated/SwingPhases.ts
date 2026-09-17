/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

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
 * Which landmark(s) the hand trajectory was taken from.
 *
 * Chosen once for a clip rather than per frame. Switching between a two-wrist
 * midpoint and a single wrist partway through would move the tracked point by
 * half the distance between the hands, and the differentiator would read that
 * step as a velocity spike -- in the middle of a swing, plausibly at the very
 * instant being measured.
 */
export type HandSource = "midpoint" | "left_wrist" | "right_wrist";

/**
 * Everything detection concluded about one clip.
 *
 * `detected` is false whenever no swing was found, and then `events` and
 * `phases` are empty rather than populated with guesses. A caller that checks
 * only for events therefore cannot mistake an undetected swing for one at
 * frame zero.
 *
 * Durations are here; ratios are not. Tempo -- the backswing-to-downswing
 * ratio -- is a biomechanics metric and belongs to Phase 5, which is also
 * where it acquires a unit, a confidence and a methodology. Computing it here
 * would put the same number in two places with two different provenances.
 */
export interface SwingPhases {
  schema_version?: number;
  detected: boolean;
  events?: DetectedEvent[];
  phases?: DetectedPhase[];
  hand: HandSignalInfo;
  frames: number;
  /**
   * How many times slower than real time the clip plays, as supplied by the caller. 1.0 is an ordinary recording. Timestamps here are **real seconds**, already divided by it, so they no longer index into the video file -- frame numbers do. Nothing in a conformed slow-motion clip records this, so it cannot be measured and is not guessed.
   */
  slow_motion_factor?: number;
  config: PhaseConfig;
  warnings?: string[];
}
/**
 * One located instant, with how it was found and what corroborates it.
 */
export interface DetectedEvent {
  event: SwingEvent;
  frame_index: number;
  timestamp_s: number;
  confidence: EventConfidence;
  /**
   * The rule that produced this frame, in words. Never omitted.
   */
  methodology: string;
  /**
   * Frame an independent signal put this event at, where a second signal exists. None means no corroboration was available -- which is not the same as the two agreeing.
   */
  corroboration_frame?: number | null;
  /**
   * Seconds between this event and its corroborating estimate.
   */
  corroboration_delta_s?: number | null;
}
/**
 * Why an event is or is not trustworthy, decomposed.
 *
 * `overall` is the product of the three factors: an event needs all of them,
 * and a product says so where an average would let a strong factor cover for a
 * fatal one. The factors are reported individually because "0.3" is not
 * actionable while "the landmark was barely visible" is.
 */
export interface EventConfidence {
  /**
   * Product of the three factors below.
   */
  overall: number;
  /**
   * How clearly the signal singles this instant out -- peak prominence for IMPACT, depth of the speed minimum for TOP, crossing sharpness for TAKEAWAY and FINISH.
   */
  margin: number;
  /**
   * Mean reported visibility of the hand landmark around this event.
   */
  visibility: number;
  /**
   * Whether the capture and smoothing can resolve an event of this duration at all. Falls towards zero as the smoothing window approaches the length of the phase being measured.
   */
  resolution: number;
}
/**
 * One interval between events.
 *
 * `start_frame` is inclusive and `end_frame` exclusive, so consecutive phases
 * tile the clip without overlapping and `end_frame - start_frame` is the frame
 * count. `duration_s` is measured from timestamps rather than derived from the
 * frame count, because on variable-rate footage those disagree.
 */
export interface DetectedPhase {
  phase: SwingPhase;
  start_frame: number;
  /**
   * Exclusive.
   */
  end_frame: number;
  start_s: number;
  end_s: number;
  duration_s: number;
  /**
   * The weaker of the confidences of the two events bounding this phase.
   */
  confidence: number;
}
/**
 * What the hand trajectory was built from, and how well it stands out.
 *
 * `travel_ratio` is the measurement that decides whether there is a swing here
 * at all: how far the hands ranged, measured in the subject's own torso
 * lengths. Judging against the body rather than against the frame makes it
 * independent of where the camera was put — the same swing filmed twice as far
 * away halves every distance in frame, torso included, and leaves the ratio
 * unchanged.
 */
export interface HandSignalInfo {
  source: HandSource;
  valid_frames: number;
  total_frames: number;
  /**
   * In the filtered signal's velocity unit.
   */
  peak_speed: number;
  /**
   * Diagonal of the box the hands stayed inside, in frame widths.
   */
  travel: number;
  /**
   * Median shoulder-midpoint to hip-midpoint distance, in frame widths.
   */
  torso_length: number;
  /**
   * travel / torso_length.
   */
  travel_ratio: number;
}
/**
 * Policy for phase detection.
 *
 * These are **structural sanity bounds, not golf norms.** Nothing here claims
 * that a backswing lasts a particular time; the durations exist to reject
 * motion that cannot be a swing at all, and are set loose enough that a slow
 * practice swing and a fast one both pass. Numbers tight enough to encode what
 * a swing "should" look like would need a labelled set, which is Phase 12.
 */
export interface PhaseConfig {
  /**
   * Fraction of peak hand speed above which the hands count as moving. Expressed as a fraction because absolute speed has no fixed meaning in normalised frame coordinates.
   */
  moving_fraction?: number;
  /**
   * The hands must range at least this many torso lengths before a swing is reported. Half a torso is a long way below a full swing, which sweeps them through more than one; the bound is there to exclude a subject standing still, not to describe what a swing looks like.
   */
  min_travel_ratio?: number;
  /**
   * The backswing and the downswing must each move the hands at least this many torso lengths. Without it a clip that is still except for one brief twitch yields a top on whichever still frame noise made highest, and a long backswing containing no movement at all.
   */
  min_phase_travel_ratio?: number;
  /**
   * Below this the motion is not treated as a backswing.
   */
  min_backswing_s?: number;
  /**
   * Below this the motion is not treated as a downswing.
   */
  min_downswing_s?: number;
  /**
   * Above this the motion is not treated as a downswing. A club falling under gravity alone covers the distance far quicker than a second, so a slower descent is someone lowering the club rather than swinging it.
   */
  max_downswing_s?: number;
  /**
   * How far either side of the highest hand position to look for the speed minimum that marks the top.
   */
  transition_search_s?: number;
  /**
   * Span around an event over which visibility and margin are measured.
   */
  confidence_window_s?: number;
  /**
   * How long the hands must stay below the moving threshold before that counts as being still. Without a minimum, the momentary pause at the top of the backswing reads as the hands having stopped, and the takeaway is found there instead of at address.
   */
  min_still_s?: number;
  /**
   * A stretch in which the estimator saw no hand is reported once it lasts this long, because by then it could be concealing the fastest part of a swing. Shorter stretches are ordinary and reporting them would be noise.
   */
  min_tracking_gap_s?: number;
}
