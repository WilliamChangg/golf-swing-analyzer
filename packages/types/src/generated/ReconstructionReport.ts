/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * Which reference frame a set of landmarks is expressed in.
 *
 * The full vocabulary, including the frames this build cannot yet produce, so
 * that a later phase adds the capability rather than the concept and nothing
 * in between can claim a frame it does not have. `analyzer/coordinates.py`
 * holds the conversions and `docs/coordinate-systems.md` the conventions.
 *
 * IMAGE
 *     Normalised to the displayed frame: x divided by the width and y by the
 *     height, both in [0, 1], y increasing downward. **Anisotropic** -- see
 *     `FrameGeometry`. As stored by the pose estimator, and the frame a
 *     landmark is drawn in.
 * FRAME_WIDTHS
 *     Both axes divided by the frame *width*, y increasing upward, origin at
 *     the bottom-left of the displayed frame. Isotropic, so a distance means
 *     the same whichever way it points, and an angle is the angle in the
 *     picture. **The frame every measurement is taken in.** Derived from
 *     IMAGE and `FrameGeometry`; never stored.
 * HIP_LOCAL
 *     Approximate metres, centred on the hip midpoint and oriented to the
 *     body. MediaPipe calls these "world landmarks"; they are not calibrated
 *     world coordinates and carry no camera geometry.
 * CAMERA
 *     Metres in three dimensions, centred on the reference camera. Produced by
 *     triangulating two calibrated views of the same instant, which is Phase 9
 *     -- so it is **not readable from a stored pose sequence**, which is one
 *     clip and therefore one projection. `analyzer.reconstruction` is where it
 *     comes from.
 * WORLD
 *     Metres in three dimensions, in a frame fixed to the scene. **Not
 *     produced by this build**, and not for want of arithmetic: a scene-fixed
 *     frame needs a gravity direction and a target line, and a stereo pair
 *     supplies neither. See `docs/coordinate-systems.md` for what would.
 */
export type LandmarkSpace = "image" | "frame_widths" | "hip_local" | "camera" | "world";
/**
 * Where a camera was put, as the user declared it.
 *
 * Deliberately the *declared* role, not the measured one. Phase 6's
 * `CameraView` is measured from the shoulder line at address and is the
 * authority on what the footage contains; this is the authority on what the
 * user intended, and the interesting case is the two disagreeing.
 *
 * OTHER exists so a third camera, or a phone propped at an angle nobody would
 * call either name, can still be part of a project instead of forcing a
 * dishonest label.
 */
export type CameraRole = "face_on" | "down_the_line" | "other";
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
 * Everything the engine concluded about reconstructing one pair of clips.
 *
 * The landmarks themselves are **not** here, for the reason
 * `PoseExtractionResult` leaves them out: a swing at 240 fps is tens of
 * thousands of frames of 33 points, which belongs in an array in the engine or
 * a columnar file on disk rather than in a JSON-RPC response. What crosses the
 * boundary is what a reader needs in order to decide whether to believe it.
 *
 * `reconstructed` is false whenever no point could be produced, and then the
 * quality fields are None rather than zero -- zero coverage and "no
 * reconstruction attempted" are different facts, and a consumer reading a zero
 * as a measurement would be reading a refusal as a result.
 */
export interface ReconstructionReport {
  schema_version?: number;
  reconstructed: boolean;
  /**
   * The frame the points are in: metres, centred on the reference camera. **Not WORLD.** A scene-fixed frame needs a gravity direction and a target line, and a stereo pair supplies neither; see the module docstring for what would.
   */
  space?: LandmarkSpace & string;
  reference_role: CameraRole;
  target_role: CameraRole;
  /**
   * Basename of the reference clip, for display.
   */
  reference_name: string;
  target_name: string;
  /**
   * Reference-clip frames the reconstruction spans.
   */
  frames: number;
  /**
   * Frames carrying at least one reconstructed landmark.
   */
  reconstructed_frames: number;
  /**
   * The reference clip's factor. Timestamps here are real seconds, as everywhere else above Phase 3.
   */
  slow_motion_factor?: number;
  /**
   * What was known about the two cameras' geometry. Always `stereo` for a reconstruction that happened; carried so a stored result cannot be read without it.
   */
  calibration: CalibrationStatus;
  /**
   * Distance between the two optical centres, from the rig.
   */
  baseline_m: number;
  /**
   * Angle between the two optical axes, from the rig. Distinct from the per-point ray convergence in `ReconstructionQuality`: this is where the cameras pointed, that is how well two particular rays met.
   */
  convergence_deg: number;
  quality?: ReconstructionQuality | null;
  pairing?: PairingSummary | null;
  landmarks?: LandmarkReconstruction[];
  reconstructed_at?: string | null;
  config: ReconstructionConfig;
  /**
   * Why nothing was produced, in words. Set exactly when `reconstructed` is false.
   */
  refusal?: string | null;
  warnings?: string[];
}
/**
 * What the reconstruction is worth, in the three senses that differ.
 *
 * Deliberately shaped like `CalibrationQuality`, because the lesson is the
 * same one a layer up. The residual is the fit, the convergence angle is the
 * capture, and the bones are the independent check -- and a tool that reported
 * only the first would be reporting the number that a badly-conditioned capture
 * makes *better*, since two nearly-parallel rays can be made to agree by moving
 * a point a long way in depth.
 */
export interface ReconstructionQuality {
  /**
   * Landmark-frames both cameras were asked about.
   */
  points_attempted: number;
  points_reconstructed: number;
  coverage: number;
  /**
   * Median distance, over both views, between a reconstructed point's projection and where that camera saw the landmark. **How well the two views agree in the direction they can disagree in** -- the component of correspondence error across the epipolar line. Blind to the component along it, which is the one that moves the point in depth.
   */
  median_reprojection_px?: number | null;
  max_reprojection_px?: number | null;
  /**
   * Median angle between the two rays at their meeting point. **The gate.** Depth error scales as 1/sin of it, so this is what decides whether the capture could determine a position, independently of how well the fit agrees with itself.
   */
  median_convergence_deg?: number | null;
  min_convergence_deg?: number | null;
  /**
   * Median propagated positional uncertainty in the worst direction, in metres. Propagated from a per-view pixel sigma through the triangulation's own Jacobian, so it carries the convergence angle automatically: the same pixel noise at 10 degrees of convergence gives six times the uncertainty it gives at 90.
   */
  median_uncertainty_m?: number | null;
  p95_uncertainty_m?: number | null;
  bones?: BoneConsistency[];
  symmetry?: SymmetryCheck[];
  /**
   * The least stable segment's coefficient of variation. The headline of the check the residual cannot make.
   */
  worst_bone_variation?: number | null;
  /**
   * The per-view landmark uncertainty the propagation used, in pixels. **Measured from this clip**, as the filter's own residual RMS -- how far the raw detections sat from the fitted trajectory -- rather than assumed. It is an upper bound on the fitted position's own error, because smoothing averages several samples, and is used as one.
   */
  pixel_sigma_px: number;
  methodology: string;
}
/**
 * One skeletal segment's reconstructed length across the clip.
 *
 * **The check the reprojection residual cannot perform.** A landmark displaced
 * along its epipolar line reprojects perfectly and sits at the wrong depth; the
 * bones ending at it are then the wrong length, and they are a different wrong
 * length in every frame, because the displacement varies. So the *variation* is
 * the error signal, and it needs no ground truth -- a humerus does not change
 * length during a swing.
 *
 * `median_length_m` is reported too, and it is this engine's **first metric
 * statement about a body**. It is not an anatomical bone length: MediaPipe's
 * landmarks are the model's estimate of a joint's image position, not a joint
 * centre, and the distance between two of them is not a distance between two
 * bones. What it is good for is a sanity check a person can make against a
 * tape measure, and that is how it should be read.
 */
export interface BoneConsistency {
  /**
   * The segment, as 'left_shoulder-left_elbow'.
   */
  name: string;
  /**
   * Frames in which both ends were reconstructed.
   */
  frames: number;
  median_length_m: number;
  /**
   * Robust coefficient of variation: the interquartile range over the median. Robust rather than a standard deviation because a handful of badly reconstructed frames should not be allowed to describe the clip, and because a clip with any of them still has a usable middle.
   */
  variation: number;
  /**
   * Interquartile range of the length, in metres. The same fact, unscaled.
   */
  spread_m: number;
  /**
   * Whether `variation` sits inside `ReconstructionConfig.max_bone_variation`.
   */
  stable: boolean;
}
/**
 * How far a left segment's reconstructed length sits from its right twin.
 *
 * A second check that needs no ground truth, for a second reason: a person's
 * left and right upper arms are the same length to within a per cent or two,
 * and the two are reconstructed from completely different landmarks with
 * completely different occlusion histories. A large disagreement is evidence
 * about the reconstruction rather than about the body.
 *
 * **It is corroboration, not a stronger instrument, and the design here
 * expected otherwise.** The argument for it was that a landmark with a
 * *consistent* depth bias -- one camera placing a hip a few centimetres deep
 * throughout, which occlusion does routinely -- would give a stable wrong
 * length in every frame and sail through the variation check. Measured, it does
 * not: a swing rotates the body, so a displacement that is constant in the
 * camera's frame is not constant relative to the bone, and the length moves.
 * Displacing one elbow 5 cm along the optical axis on the synthetic body makes
 * the forearm's variation 11% -- past the bound -- while the left-right
 * disagreement is 2.6%, well inside it. Variation gets there first.
 *
 * So what this adds is **localisation**, which is worth its two lines: it
 * compares two chains reconstructed from different landmarks, so a large
 * disagreement says one *side* is systematically worse, which a per-segment
 * variation does not say. The test asserting this is
 * `test_a_depth_bias_reaches_variation_before_symmetry`, and it is written to
 * fail if anyone restores the stronger claim.
 */
export interface SymmetryCheck {
  /**
   * The pair, as 'shoulder-elbow'.
   */
  segment: string;
  left_m: number;
  right_m: number;
  /**
   * |left - right| divided by their mean.
   */
  disagreement: number;
  /**
   * Whether the disagreement sits inside `ReconstructionConfig.max_asymmetry`.
   */
  agrees: boolean;
}
/**
 * How the two cameras' instants were brought together, and what it cost.
 *
 * The Phase 8 thread, continued into a scene where its capture instruction is
 * unavailable. See the module docstring: a board can be held still and a swing
 * cannot, so the target clip is resampled onto the reference clock rather than
 * paired frame-to-frame, and what remains is the time map's own uncertainty.
 *
 * That is reported here in **pixels**, by multiplying it by the landmark's
 * measured image speed, for the same reason `PairingReport` does: it is then
 * the same unit as the reprojection error it contaminates, and the two can be
 * compared instead of one being mistaken for the other.
 */
export interface PairingSummary {
  /**
   * How the target clip's positions were obtained at reference instants: 'resampled' (Hermite interpolation of the filter's own fitted position and velocity) or 'nearest_frame'.
   */
  method: string;
  /**
   * Reference frames a target position was produced at.
   */
  resampled_frames: number;
  /**
   * Reference frames whose mapped instant fell outside the target clip.
   */
  outside_overlap: number;
  /**
   * The target clip's own frame interval. What nearest-frame pairing would round to.
   */
  median_interval_ms: number;
  /**
   * Median image speed of the tracked landmarks in the target view, over the reconstructed frames. The factor that turns a timing error into a displacement -- and the one a still board sets to zero and a swing cannot.
   */
  median_landmark_speed_px_s: number;
  /**
   * The fastest any tracked landmark moved in the target image. Usually a hand.
   */
  max_landmark_speed_px_s: number;
  /**
   * The time map's own uncertainty over the reconstructed span, in milliseconds, as Phase 7 reports it. None when the map carried none.
   */
  sync_uncertainty_ms?: number | null;
  /**
   * Sync uncertainty times landmark image speed, at the median. The displacement attributable to not knowing exactly when each frame was taken. Comparable with `median_reprojection_px` because it is the same unit and it adds to it.
   */
  median_pairing_error_px?: number | null;
  /**
   * The same quantity at the fastest landmark. This is the number that makes nearest-frame pairing unusable on a swing: it is tens of pixels at consumer frame rates, against sub-pixel calibration residuals.
   */
  max_pairing_error_px?: number | null;
  /**
   * What pairing to the nearest target frame *would* have cost at the **fastest** landmark: a quarter of a frame interval -- the mean magnitude of a rounding uniform over half of one -- times that speed. Quoted at the fastest rather than the median because the median is dominated by the parts of a body that barely move, and every metric worth computing is anchored to the parts that do. Reported rather than assumed, because it is the measurement that decided this layer resamples.
   */
  nearest_frame_error_px?: number | null;
}
/**
 * One landmark's reconstruction across the clip, and where it failed.
 *
 * Per landmark rather than clip-wide, because the failures are per landmark:
 * down-the-line footage hides one wrist behind the other for half a swing, and
 * a clip-wide coverage figure would report that as a uniform 80% rather than as
 * one landmark at 40% and the rest at 95%. The first reading suggests a
 * marginal capture; the second names the landmark to distrust.
 */
export interface LandmarkReconstruction {
  /**
   * The `Landmark` enum value.
   */
  landmark: number;
  name: string;
  /**
   * Reference frames attempted.
   */
  frames: number;
  reconstructed: number;
  coverage: number;
  /**
   * Median angle between the two rays. The conditioning of the intersection.
   */
  median_convergence_deg?: number | null;
  median_reprojection_px?: number | null;
  /**
   * Median propagated positional uncertainty, in metres, in the worst direction -- which is essentially always the depth direction. See `ReconstructionQuality.methodology`.
   */
  median_uncertainty_m?: number | null;
  /**
   * Frames refused, counted by reason.
   */
  refused?: {
    [k: string]: number;
  };
}
/**
 * Policy for reconstruction.
 *
 * As everywhere else in this engine, **stated policy informed by measurement**
 * rather than measured optima. `scripts/benchmark_reconstruct.py` establishes
 * the shape of each relationship -- what error a given convergence angle
 * produces, how a residual responds to an epipolar displacement and how it
 * fails to -- and where on that curve to refuse is a judgement about what a
 * wrong answer costs. Where a number came from a sweep, the field says so.
 */
export interface ReconstructionConfig {
  /**
   * Angle between the two rays below which a point is refused rather than reported. **The gate**, and the only bound here that describes the capture rather than the fit.
   *
   * Measured (`--sweep convergence`): depth error scales as 1/sin, so 15 degrees costs 3.9x the error of a right-angled pair and 5 degrees costs 11.5x. Fifteen is where a centimetre-scale reconstruction becomes a several-centimetre one, which is roughly where a joint angle stops being worth reporting. Two cameras placed as the capture protocol asks -- one face-on, one down-the-line -- converge at nearly 90 degrees and are nowhere near this.
   */
  min_convergence_deg?: number;
  /**
   * How far a reconstructed point may land from where a camera saw it before that point is refused. Loose on purpose, and loose for a different reason from the calibration bound it resembles: a calibration residual is a static board fitted by a rigid model, and this is a pose estimator's guess at a joint on a moving body seen from two sides. Several pixels of honest disagreement is ordinary.
   *
   * What it catches is gross failure -- a mis-paired instant, one camera tracking the wrong wrist. It is **not** a quality measure, and the module docstring is about why not.
   */
  max_reprojection_px?: number;
  /**
   * Propagated positional uncertainty above which a point is refused. Fifteen centimetres is most of the width of a pelvis, which is roughly the point past which a reconstructed joint stops distinguishing the poses it exists to distinguish. None disables the bound and reports the uncertainty without acting on it.
   */
  max_uncertainty_m?: number | null;
  /**
   * Reported visibility, in **either** view, below which the landmark is not triangulated at that instant. Either rather than both: a point reconstructed from one confident view and one guess is a point located by the guess, since it is the disagreement between the two that fixes the depth.
   */
  min_visibility?: number;
  /**
   * Robust coefficient of variation above which a reconstructed segment is reported as unstable. Ten per cent of a forearm is about three centimetres, which is the scale at which the reconstruction stops supporting the joint angles it exists to produce.
   *
   * **Not tighter, because not every segment here is a bone.** `POSE_CONNECTIONS` includes shoulder-to-hip on each side, and a torso genuinely changes that distance as it twists -- the synthetic body, whose limb lengths are exact by construction, varies those two by 3% across a swing purely from the shoulders and pelvis turning different amounts. A bound tight enough for a humerus would report that anatomy as an error in every clip.
   */
  max_bone_variation?: number;
  /**
   * How far a left segment's median length may sit from its right twin's, as a fraction of their mean, before the pair is reported as disagreeing. Wider than `max_bone_variation` because real people are slightly asymmetric and because one side is systematically the more occluded in any single capture.
   */
  max_asymmetry?: number;
  /**
   * Per-view landmark uncertainty used to propagate a positional uncertainty. None measures it from the clip, as the filter's own residual RMS -- which is what the detections actually did, rather than a number chosen to make the output look good. Supplying one is for a sensitivity check.
   */
  pixel_sigma_px?: number | null;
  /**
   * Whether to resample the target clip onto the reference clock rather than pairing each reference frame with the target frame nearest it in time.
   *
   * **Measured, and it is not a close call.** Phase 8 pairs nearest frames and gets away with it because a calibration board can be held still, which drives the `time_error x image_speed` product to zero. A swing cannot be held still: at 30 fps a quarter-frame pairing error against hands moving several thousand pixels per second is tens of pixels of displacement, two orders of magnitude past the sub-pixel residuals the calibration itself was gated on. `scripts/benchmark_reconstruct.py --sweep pairing` has both. False is there so that comparison can be run, not as an option worth choosing.
   */
  resample?: boolean;
  /**
   * Whether to refine the linear triangulation by minimising reprojection error in both views. The linear solution minimises an algebraic quantity with no geometric meaning and is biased when the two views are at very different distances; the refinement is a few Gauss-Newton steps from it and is what makes the residual a reprojection error rather than a proxy for one. Off is for measuring what it bought.
   */
  refine?: boolean;
  /**
   * Gauss-Newton steps. From a linear seed this converges in two or three; the bound exists so a pathological point cannot spend the clip's budget rather than because ten are expected to be used.
   */
  max_refine_iterations?: number;
}
