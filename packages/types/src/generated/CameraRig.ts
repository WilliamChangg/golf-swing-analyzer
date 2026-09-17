/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

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
 * Which lens-distortion coefficients were fitted.
 *
 * Named and chosen rather than defaulted, because a coefficient that is not
 * determined by the data does not sit harmlessly near zero -- it takes a value
 * that cancels some of the residual in the region the board happened to cover
 * and then diverges outside it, which is the region a swing is filmed in.
 *
 * PINHOLE
 *     No distortion. Correct only for a lens that has none, which no phone
 *     wide-angle has; useful as a null model and as the thing a test can
 *     recover exactly.
 * RADIAL_TANGENTIAL_4
 *     k1, k2, p1, p2. Two radial terms and the tangential pair that models a
 *     lens not quite parallel to the sensor. The default here, on the
 *     measurement in `scripts/benchmark_calibration.py`.
 * RADIAL_TANGENTIAL_5
 *     The above plus k3. OpenCV's own default, and a third radial term is
 *     genuinely needed by a fisheye or an action camera -- but it is the term
 *     that most readily absorbs noise when the board never reaches the corners,
 *     and it does its damage exactly there.
 */
export type DistortionModel = "pinhole" | "radial_tangential_4" | "radial_tangential_5";
/**
 * Which ArUco dictionary the board's markers are drawn from.
 *
 * Carried rather than fixed because the board is a physical object that the
 * user printed, and a detector looking for the wrong dictionary finds nothing
 * at all -- a failure that looks exactly like bad lighting.
 *
 * The default is `DICT_5X5_100`: 5x5 markers stay legible at the size a
 * Charuco square allows on an A4 sheet, and 100 of them is far more than a
 * board this size consumes, so ids never wrap.
 */
export type BoardFamily = "DICT_4X4_50" | "DICT_4X4_100" | "DICT_5X5_100" | "DICT_5X5_250" | "DICT_6X6_250";

/**
 * Everything known about the geometry of one project's cameras.
 *
 * Stored whole, one per project, for the reason `ProjectSync` is stored whole:
 * it is a nested document read entire and never queried by part, and the
 * Pydantic model is already the authoritative definition of its shape.
 *
 * Keeping both cameras and their relation in one document rather than three
 * rows is what makes `status` computable without a join, and `status` is the
 * thing every consumer actually asks for.
 */
export interface CameraRig {
  schema_version?: number;
  cameras?: {
    [k: string]: CameraCalibration;
  };
  stereo?: StereoCalibration | null;
  config?: CalibrationConfig;
}
/**
 * One camera's calibration: what was measured, from what, and how well.
 *
 * `usable` and `refusal` are the gate. A calibration that exists is not a
 * calibration that may be used, and the difference is decided here rather than
 * by each consumer, so a consumer that forgets to check the coverage cannot
 * silently use a set of views that never determined a focal length.
 */
export interface CameraCalibration {
  schema_version?: number;
  role: CameraRole;
  intrinsics: CameraIntrinsics;
  quality: CalibrationQuality;
  detection: DetectionReport;
  calibrated_at: string;
  /**
   * What the board views came from, for provenance.
   */
  source: string;
  /**
   * Free text for the capture setting a video file cannot record: which lens, what zoom, whether stabilisation was off. The failure this field exists for is a correct calibration applied to footage shot at a different zoom, which nothing can detect.
   */
  notes?: string;
  /**
   * Whether this calibration passed the bounds in `CalibrationConfig`.
   */
  usable: boolean;
  /**
   * Why it did not, in words. Set exactly when `usable` is false.
   */
  refusal?: string | null;
  warnings?: string[];
}
/**
 * One camera's interior geometry, measured.
 *
 * Stated as four scalars rather than a 3x3 matrix because that is what was
 * fitted: the skew term of a general K is zero for any sensor manufactured in
 * the last forty years, and carrying a full matrix would invite a reader to
 * believe this build had measured it. `matrix()` assembles the conventional
 * form for the arithmetic that wants one.
 *
 * Pixel coordinates here are **displayed** pixels, after Phase 1's rotation has
 * been applied -- the same frame the landmarks are normalised against. A
 * calibration measured on the stored orientation and applied to the displayed
 * one has fx and fy interchanged, which is a plausible-looking calibration that
 * is wrong everywhere.
 */
export interface CameraIntrinsics {
  /**
   * Focal length in pixels, horizontal.
   */
  fx: number;
  /**
   * Focal length in pixels, vertical.
   */
  fy: number;
  /**
   * Optical centre, horizontal, in pixels from the left.
   */
  cx: number;
  /**
   * Optical centre, vertical, in pixels from the top.
   */
  cy: number;
  /**
   * Distortion coefficients in OpenCV's order (k1, k2, p1, p2[, k3]), as many as `model` declares.
   */
  distortion: number[];
  model: DistortionModel;
  /**
   * Displayed frame width this was measured at.
   */
  image_width: number;
  /**
   * Displayed frame height this was measured at.
   */
  image_height: number;
  /**
   * Standard deviation of `fx`, in pixels, propagated from the fit's residuals through its Jacobian.
   *
   * **It does not catch a degenerate capture, which is what the design here expected of it.** Measured, it is smallest exactly where the answer is worst -- 0.012% on a capture whose focal length is wrong by 6%, against 0.31% on one that is right to 0.05% -- because the distortion coefficients absorb the degeneracy and leave a tightly determined wrong answer. A covariance computed from one set of views cannot see outside them. Reported because it is a real statement about the fit's conditioning, and not what `usable` rests on; `CoverageReport` is. None where the fit did not report one.
   */
  fx_uncertainty?: number | null;
  fy_uncertainty?: number | null;
  cx_uncertainty?: number | null;
  cy_uncertainty?: number | null;
}
/**
 * How good a calibration is, in the three senses that differ.
 *
 * See the module docstring. The short version: `rms_reprojection_px` is the
 * fit, the parameter uncertainties on `CameraIntrinsics` are the answer, and
 * `coverage` is the cause. A tool that reports only the first is reporting the
 * one of the three that a degenerate capture makes *better*.
 */
export interface CalibrationQuality {
  /**
   * Root-mean-square distance between each detected corner and where the fitted model puts it. How well the model fits the data it was given -- necessary, and on its own not evidence that the data determined the model.
   */
  rms_reprojection_px: number;
  /**
   * Worst single corner residual. Separates a uniformly mediocre fit from a good one with a misdetected view in it, which call for different responses.
   */
  max_reprojection_px: number;
  /**
   * Residual per board view, in the order the views were used. Drawn by 8.4.
   */
  per_view_rms_px?: number[];
  coverage: CoverageReport;
  /**
   * Corner observations times two, minus the parameters fitted. The same guard `SyncQuality.degrees_of_freedom` provides one layer down: a residual from a fit with nothing spare is not evidence.
   */
  degrees_of_freedom: number;
}
/**
 * What the board views actually sampled, and therefore what they could fix.
 *
 * The diagnostic half of a calibration. Each field detects a different
 * degeneracy, and every one of them is invisible in the reprojection residual
 * -- a degenerate set fits *better*, because the model has less to disagree
 * with. See the module docstring for why each matters.
 */
export interface CoverageReport {
  /**
   * Board views that entered the fit.
   */
  views: number;
  /**
   * Chessboard corners summed over those views.
   */
  corners: number;
  /**
   * Fraction of the frame, on a coarse grid, containing at least one detected corner. The blunt measure of whether the lens was sampled where it bends.
   */
  image_fraction: number;
  /**
   * Fraction of all detected corners lying in the outer fifth of the frame radius. Distortion is a function of radius and is nearly nothing at the centre, so this is the share of the evidence that carries any information about it at all.
   */
  edge_fraction: number;
  /**
   * Spread between the least and most tilted board view, in degrees off square to the camera. **The field that catches the worst case**: a board always held parallel to the sensor cannot separate focal length from distance, and produces an excellent residual around a focal length that is simply not determined.
   */
  tilt_range_deg: number;
  /**
   * Largest apparent board size over smallest, across views. 1.0 means every view was taken at one distance.
   */
  scale_range: number;
  methodology: string;
}
/**
 * What board detection found across a source, and what it discarded.
 *
 * Every count is here rather than only the survivors, because the ratios are
 * the diagnostic. Frames scanned against frames with a board found says
 * whether the board was visible; found against used says whether the views
 * were good enough; and a large gap in either has a different fix -- reshoot
 * with the board in frame, or reshoot with it closer and steadier.
 */
export interface DetectionReport {
  frames_scanned: number;
  frames_with_board: number;
  views_used: number;
  corners_total: number;
  board: BoardSpec;
  observations?: BoardObservation[];
  warnings?: string[];
}
/**
 * The physical board, as printed.
 *
 * Charuco rather than a plain chessboard, for two reasons that both matter on
 * footage taken by one person with no assistant. A chessboard must be seen
 * **whole** or it yields nothing, so every view where a corner leaves the
 * frame is wasted -- which is exactly the view that carries the distortion
 * information. And its corners are interchangeable, so a rotationally
 * symmetric view is ambiguous. Charuco's markers identify each corner
 * individually, so a partial view contributes the corners it does show and the
 * correspondence is never in doubt.
 *
 * `square_length_m` is what puts a metre into the system. Every metric-scale
 * claim Phase 9 makes descends from this one measured number, so it is the
 * length to measure on the printed sheet with a ruler rather than to take from
 * the file that was sent to the printer: printers scale to fit, and a page
 * scaled to 96% makes every future distance wrong by 4% with nothing
 * anywhere looking amiss.
 */
export interface BoardSpec {
  /**
   * Chessboard squares across.
   */
  squares_x: number;
  /**
   * Chessboard squares down.
   */
  squares_y: number;
  /**
   * Side of one chessboard square, in metres, **measured on the printed sheet**. The scale of every metric claim descends from this number.
   */
  square_length_m: number;
  /**
   * Side of one ArUco marker, in metres. Must be smaller than the square it sits in; around 0.75 of it leaves a white margin that the detector needs to find the marker's border.
   */
  marker_length_m: number;
  family?: BoardFamily;
  /**
   * Whether the board was generated by OpenCV before 4.6, which laid the markers out differently. A board printed from an old generator and detected as a new one produces corners in the wrong places rather than no corners at all, which is the worse failure. False for any board this build generated.
   */
  legacy_pattern?: boolean;
}
/**
 * One frame in which the board was found.
 *
 * Kept per frame rather than summarised because 8.4 draws them: the review UI
 * shows where in the image each view landed, which is how a person sees that
 * their board never reached a corner -- a fact no single number communicates
 * as quickly as the picture of it does.
 */
export interface BoardObservation {
  frame: number;
  /**
   * Chessboard corners identified in this view.
   */
  corners: number;
  /**
   * This view's own RMS residual after the fit. None before a fit, and for a view the fit excluded.
   */
  reprojection_rms_px?: number | null;
  /**
   * How far off square to the camera this view sat.
   */
  tilt_deg?: number | null;
  /**
   * Board distance along the optical axis. Metric, because the board's own square size supplies the scale -- the first genuinely metric length in this engine.
   */
  distance_m?: number | null;
  /**
   * Mean corner position, in pixels.
   */
  centroid_x: number;
  centroid_y: number;
  /**
   * Whether this view entered the fit, or was dropped.
   */
  used?: boolean;
  dropped_reason?: string | null;
}
/**
 * Where the second camera stands relative to the first, measured.
 *
 * Stated as the rotation and translation taking a point from the reference
 * camera's frame into the target's. The baseline is reported separately
 * because it is the one number in here a person can check with a tape measure,
 * and a stereo calibration whose baseline is wrong by a factor of two is
 * usually a board whose `square_length_m` was taken from the file rather than
 * the print.
 */
export interface StereoCalibration {
  schema_version?: number;
  reference_role: CameraRole;
  target_role: CameraRole;
  /**
   * 3x3 rotation, reference frame to target.
   */
  rotation: number[][];
  /**
   * 3-vector, in metres, in the target's frame.
   */
  translation_m: number[];
  /**
   * Distance between the two optical centres. Checkable with a tape.
   */
  baseline_m: number;
  /**
   * Angle between the two optical axes. Near zero is a parallel rig; a face-on and a down-the-line camera are near 90, which is the best possible geometry for triangulation and the worst for finding anything visible in both.
   */
  convergence_deg: number;
  quality: CalibrationQuality;
  pairing: PairingReport;
  calibrated_at: string;
  usable: boolean;
  refusal?: string | null;
  warnings?: string[];
}
/**
 * How the two cameras' board views were paired, and what that cost.
 *
 * The measurement that lets an unsynchronised pair be stereo-calibrated at
 * all. `worst_pairing_error_px` is the time map's uncertainty converted into
 * the unit it contaminates, by multiplying it by the board's observed image
 * speed: a still board makes it nearly zero however badly the clocks are
 * known, and a moving one makes it large however well they are. See the module
 * docstring.
 */
export interface PairingReport {
  /**
   * Frame pairs that entered the stereo fit.
   */
  pairs: number;
  /**
   * Pairs considered before the bounds below dropped any.
   */
  candidates: number;
  /**
   * How frames were matched: 'time_map' or 'explicit'.
   */
  method: string;
  /**
   * Median distance between the two frames of a pair once mapped onto one clock. None for explicit pairs, where the caller asserted simultaneity and nothing here can check it.
   */
  median_time_error_ms?: number | null;
  /**
   * Largest board displacement, in pixels, attributable to the pairing being imperfect: the sync uncertainty there times the board's image speed there. Directly comparable with the reprojection error, because it is the same unit and it adds to it.
   */
  worst_pairing_error_px?: number | null;
  /**
   * How fast the board was moving in the image across the used pairs. The number the capture instruction acts on: hold it still and this goes to zero, and with it the whole cost of not having a genlock.
   */
  median_board_speed_px_s?: number | null;
  /**
   * Why each rejected candidate was rejected.
   */
  dropped?: string[];
}
/**
 * Policy for calibration.
 *
 * The bounds are **stated policy informed by measurement**, not measured
 * optima: `scripts/benchmark_calibration.py` establishes the shape of each
 * relationship -- that focal uncertainty explodes as tilt range collapses,
 * for instance -- and where on that curve to refuse is a judgement about what
 * a wrong answer costs downstream. Where a number came from a sweep, the field
 * says so.
 */
export interface CalibrationConfig {
  distortion_model?: DistortionModel;
  /**
   * Board views required before a calibration is attempted. Three is the algebraic minimum for a planar target and is nowhere near enough in practice; the usual advice of 15-20 is about coverage rather than count, which `CoverageReport` measures directly. This is a floor below which the coverage numbers themselves stop meaning much.
   */
  min_views?: number;
  /**
   * Corners a view must contribute to be used. Four is the minimum for a pose; six leaves something over, and a view showing fewer is usually the board leaving the frame rather than a useful oblique.
   */
  min_corners_per_view?: number;
  /**
   * Fraction of the frame the board must have visited. Below it the lens is unmeasured over most of its area, and the distortion coefficients are extrapolating into the region the subject is filmed in.
   */
  min_image_fraction?: number;
  /**
   * Spread of board tilts required. **The bound that catches the failure neither the residual nor the reported uncertainty can see.** With every view square to the sensor, focal length and distance are very nearly interchangeable, and the fit is free to pick badly while fitting beautifully.
   *
   * Measured (`--sweep tilt`): at 0-2 degrees of spread the focal length comes out 6-10% wrong at an RMS of 0.21 px; at 20 degrees it is 0.10% wrong at an RMS of 0.25 px. Twenty is where the error collapses, and it is far more tilt than anyone holding a board produces by accident, which is the point.
   */
  min_tilt_range_deg?: number;
  /**
   * Standard deviation of the fitted focal length, as a fraction of it, above which the calibration is refused. Two percent of focal length is roughly two percent of every triangulated depth, which is a centimetre at half a metre and compounds through Phase 9.
   *
   * A backstop rather than the main instrument: it catches a fit that is visibly ill-conditioned and, as `fx_uncertainty` documents, misses the degenerate capture entirely. The coverage bounds are what catch that, and they are checked first.
   */
  max_focal_uncertainty_ratio?: number;
  /**
   * Residual above which the fit is refused outright. A pixel is a loose bound -- a good calibration lands well under half of one -- and it is deliberately loose, because this bound catches gross failure (a mis-specified board, a wrong dictionary) rather than mediocrity, which the uncertainty bound above is the instrument for.
   */
  max_rms_reprojection_px?: number;
  /**
   * How far fx and fy may differ, as a fraction of their mean. Consumer sensors have square pixels, so a real disagreement is evidence of a bad fit rather than of an unusual camera.
   */
  max_focal_disagreement?: number;
  /**
   * Board displacement attributable to imperfect pairing, above which a stereo pair is dropped. One pixel, so that the sync error a pair contributes stays below the reprojection error it would otherwise be mistaken for. Holding the board still is what keeps pairs inside this; nothing else has to.
   */
  max_pairing_error_px?: number;
  /**
   * How far apart, on one clock, two frames of a pair may sit. A backstop for the bound above rather than the main instrument: a perfectly still board would satisfy the pixel test at any time error, and at some separation the assumption that nothing else in the scene moved stops being reasonable.
   */
  max_pair_time_error_ms?: number;
  /**
   * Paired board views required for stereo extrinsics. Fewer resolves the pose but leaves nothing over to check it with, and a stereo calibration with no spare degrees of freedom has a residual of zero by construction.
   */
  min_stereo_pairs?: number;
}
