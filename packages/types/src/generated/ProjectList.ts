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
 * What produced a time map.
 *
 * MANUAL
 *     Anchors supplied by a person, who looked at both clips and said which
 *     frames show the same instant. The engine does the arithmetic and the
 *     error accounting; it does not second-guess the instants.
 *
 * EVENTS
 *     Anchors from Phase 4's detected swing events, paired by name. The only
 *     method that can estimate a rate, because it is the only one that
 *     produces anchors spread across the swing.
 *
 * CORRELATION
 *     The lag that maximises the normalised correlation of the two hand-speed
 *     signals. Produces one number -- an offset -- and no rate.
 *
 * COMBINED
 *     The default where both are available, and each is used for what it can
 *     actually determine: the **offset from the correlation**, the anchors and
 *     the residual from the events. See `SyncConfig.prefer_correlated_offset`
 *     for the measurement that decided this.
 */
export type SyncMethod = "manual" | "events" | "correlation" | "combined";
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
 * Where one anchor's pair of frames came from.
 *
 * Kept distinct from `SyncMethod` because a single fit may mix them: a person
 * correcting one badly-detected event leaves the other three as detections,
 * and the result is neither wholly manual nor wholly automatic.
 */
export type AnchorSource = "detected" | "manual";
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
 * Every project, with its clips but without its syncs.
 *
 * Syncs are omitted here on purpose: a listing is for choosing which project to
 * open, and a stored `SyncModel` carries anchors, residuals and a correlation
 * report, which would make the list of a dozen projects an order of magnitude
 * larger than the thing being chosen from.
 */
export interface ProjectList {
  schema_version?: number;
  projects?: Project[];
  /**
   * Where these were read from. Useful when one is missing.
   */
  database_path: string;
}
/**
 * A session: the clips of one swing, and how their clocks relate.
 *
 * `clips` is a list rather than a fixed pair. Two is what Phases 7 to 9 need
 * and what the capture protocol asks for, but nothing here is made harder by
 * allowing three, and a fixed pair would have to be widened later by changing
 * the stored shape rather than by adding a row.
 */
export interface Project {
  schema_version?: number;
  id: number;
  name: string;
  notes?: string;
  created_at: string;
  clips?: ProjectClip[];
  syncs?: ProjectSync[];
  /**
   * What is known about this project's cameras: their intrinsics, and their relative pose where it has been measured. None until a calibration is stored, which is the ordinary state -- an uncalibrated project is fully supported and simply makes no metric-scale claims.
   *
   * One rig per project rather than one per clip, because extrinsics relate two cameras and belong to neither. It is read whole and never queried by part, which is why it is stored as a document.
   */
  rig?: CameraRig | null;
  /**
   * Facts about reading this project back. A stored sync written under a schema version this build does not understand is reported here and left out of `syncs`, rather than making the whole project unopenable over one stale row.
   */
  warnings?: string[];
}
/**
 * One video belonging to a project.
 */
export interface ProjectClip {
  id: number;
  role: CameraRole;
  path: string;
  /**
   * Basename, for display.
   */
  name: string;
  content_key: ContentKey;
  /**
   * How many times slower than real time this clip plays, as supplied when it was added. Stored per clip because two cameras in one session routinely differ -- one phone at 240 fps and one at 30.
   */
  slow_motion_factor: number;
  added_at: string;
  /**
   * Whether a file is still at `path`. False does not invalidate the project: the content key still identifies the clip and its cached extraction, and the fix is to point the project at the moved file.
   */
  exists: boolean;
  /**
   * Free text the user attached. Never interpreted.
   */
  label?: string;
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
 * A stored alignment between two of a project's clips.
 *
 * The clip ids are here as well as inside the model because the model
 * identifies its clips by path, and a project's clips can be re-pointed at
 * moved files. The ids are what survive that.
 */
export interface ProjectSync {
  reference_clip_id: number;
  target_clip_id: number;
  model: SyncModel;
  updated_at: string;
}
/**
 * Everything the engine concluded about aligning two clips.
 *
 * `aligned` is false whenever no map could be produced or the map produced was
 * too poor to report, and then `time_map` is None rather than populated with a
 * guess. A caller that reads the map without checking cannot therefore mistake
 * a refusal for an alignment at zero offset, which is the failure mode this
 * shape exists to prevent -- and zero offset is exactly the wrong value to
 * default to, because it is also a perfectly plausible answer.
 */
export interface SyncModel {
  schema_version?: number;
  aligned: boolean;
  /**
   * What produced the map. None when nothing did.
   */
  method?: SyncMethod | null;
  reference: ClipRef;
  target: ClipRef;
  time_map?: TimeMap | null;
  anchors?: SyncAnchor[];
  residuals?: AnchorResidual[];
  quality?: SyncQuality | null;
  confidence?: SyncConfidence | null;
  /**
   * The independent estimate, when one could be computed. Never merged into the map.
   */
  correlation?: CorrelationReport | null;
  overlap?: OverlapReport | null;
  config: SyncConfig;
  /**
   * Why no map was produced, in words. Set exactly when `aligned` is false.
   */
  refusal?: string | null;
  warnings?: string[];
}
/**
 * One clip of a synchronised pair, as the sync layer needs to see it.
 *
 * Timings here are **real seconds**, already divided by the slow-motion factor,
 * matching every other duration in the engine. `slow_motion_factor` is carried
 * alongside so a reader can recover the clip's own playback clock, and because
 * a fitted rate far from 1.0 usually means one of these two numbers is wrong.
 */
export interface ClipRef {
  path: string;
  /**
   * Basename, for display.
   */
  name: string;
  frames: number;
  /**
   * Timestamp of the first frame, in real seconds.
   */
  start_s: number;
  /**
   * Span from the first to the last frame, in real seconds.
   */
  duration_s: number;
  /**
   * Median gap between frames, in real seconds. The quantisation of every instant located in this clip, and half of what bounds how well the pair can be aligned.
   */
  median_interval_s: number;
  /**
   * The factor the caller supplied for this clip. 1.0 is real time.
   */
  slow_motion_factor: number;
}
/**
 * The affine map from reference-clip time to target-clip time.
 *
 * Both directions are provided because both are needed: rendering the target
 * clip alongside the reference needs `to_target`, and reporting an instant
 * found in the target in the reference's frame numbering needs `to_reference`.
 */
export interface TimeMap {
  /**
   * How far ahead of the reference clock the target clock runs, at `pivot_s`. With `rate` at 1.0 this is the whole map.
   */
  offset_s: number;
  /**
   * Target seconds per reference second. 1.0 means the two clocks run at the same speed, which is the default and is not a measurement unless `rate_estimated` is true.
   */
  rate: number;
  /**
   * True when the rate was fitted from anchors spanning enough time to determine it. False means it was held at 1.0, which is an assumption about the hardware rather than something measured here.
   */
  rate_estimated: boolean;
  /**
   * The reference-clip instant `offset_s` is stated at: the centroid of the anchors. Chosen so the offset and rate uncertainties are uncorrelated and can be combined by `uncertainty_at`.
   */
  pivot_s: number;
  /**
   * Standard error of `offset_s`, from the fit's residual scatter where there are spare degrees of freedom and from the frame quantisation otherwise. None where neither is available.
   */
  offset_uncertainty_s?: number | null;
  /**
   * Standard error of `rate`. None whenever the rate was not estimated.
   */
  rate_uncertainty?: number | null;
  /**
   * Earliest reference time an anchor was placed at.
   */
  support_start_s: number;
  /**
   * Latest reference time an anchor was placed at.
   */
  support_end_s: number;
}
/**
 * One instant identified in both clips.
 *
 * Frames and seconds are both carried. Frames are what a person picks and what
 * a UI scrubs to; seconds are what the map is fitted in, and on
 * variable-rate footage the two are not interconvertible without the clip's own
 * timestamps -- which live in the engine, not in the caller.
 */
export interface SyncAnchor {
  /**
   * What this instant is, in words. Shown to a reader.
   */
  label: string;
  /**
   * The swing event this anchor is, when it is one. None for a manual pick.
   */
  event?: SwingEvent | null;
  source: AnchorSource;
  reference_frame: number;
  target_frame: number;
  reference_s: number;
  target_s: number;
  /**
   * How well this instant was located, in the weaker of the two clips. A detected anchor inherits Phase 4's event confidence; a manual anchor is the assertion being fitted rather than an estimate of it, and scores 1.0 -- what checks it is the residual, not this.
   */
  confidence: number;
}
/**
 * How far one anchor sits from the fitted map.
 *
 * Signed, and in milliseconds, because the sign is the informative part: four
 * residuals that alternate are noise, and four that trend are a rate the fit
 * was not allowed to take up.
 */
export interface AnchorResidual {
  label: string;
  reference_s: number;
  observed_target_s: number;
  predicted_target_s: number;
  /**
   * observed - predicted, in milliseconds.
   */
  residual_ms: number;
}
/**
 * How well the alignment is determined, in milliseconds.
 *
 * `quantisation_floor_ms` is the bound nothing about this pair can beat. An
 * instant located to the nearest frame carries a uniform error of a frame
 * interval, whose standard deviation is that interval over sqrt(12); two clips
 * contribute one each, in quadrature. A residual at the floor means the anchors
 * agree as well as their frame rates allow, and the coarser clip is what sets
 * it -- pairing a 240 fps camera with a 30 fps one buys almost nothing.
 */
export interface SyncQuality {
  /**
   * RMS of the anchor residuals. **None when the fit has no spare degrees of freedom**, where the residual is zero by construction and reporting it would look like a perfect alignment.
   */
  residual_rms_ms?: number | null;
  /**
   * Largest absolute anchor residual, on the same condition.
   */
  residual_max_ms?: number | null;
  /**
   * Anchors minus fitted parameters. Zero or less means the residual is not evidence.
   */
  degrees_of_freedom: number;
  /**
   * Best residual the two clips' frame rates permit. See the class docstring.
   */
  quantisation_floor_ms: number;
  /**
   * Distance between the fitted offset and the cross-correlation's, at the pivot. None when only one estimate exists, which is itself worth knowing: nothing independent checked the map.
   *
   * **Comparable only when the map is offset-only.** A correlation produces one lag for the whole clip; a map with a fitted rate has a different offset at every instant, and the two are then estimating different quantities. A rate of 0.91 moves the map by 90 ms per second away from the pivot, so a correlation landing 9 ms from the pivot value is close agreement rather than poor. It is reported in both cases and drives `SyncConfidence.agreement` in neither, once a rate is in play -- there the residual is the better instrument, being the only one that is rate-aware.
   */
  method_disagreement_ms?: number | null;
}
/**
 * Why an alignment is or is not trustworthy, decomposed.
 *
 * Three measured factors and their product, following Phase 4 and Phase 5. The
 * product is the headline because an alignment needs all three; the factors are
 * what make a low score actionable, and they fail for different reasons that
 * call for different fixes -- re-film, re-pick the anchors, or supply the right
 * slow-motion factor.
 */
export interface SyncConfidence {
  /**
   * Product of the three factors below.
   */
  overall: number;
  /**
   * Evidence about whichever estimator actually produced the offset, which is a different quantity for the two kinds of map.
   *
   * Where the anchors produced it (`events`, `manual`): how closely they agree with the map, measured against the quantisation floor rather than against zero, taken at its worst alongside the cross-correlation's independent estimate.
   *
   * Where the correlation produced it (`combined`, `correlation`): how far the peak sat above its nearest rival. Under realistic landmark noise the anchor residual cannot do this job -- four events scatter tens of milliseconds on a pair that is correctly aligned to under one, and scatter no more than that on two clips of genuinely different swings. The correlation margin separates those cases and the residual does not; `scripts/benchmark_sync.py` has the numbers.
   *
   * **Zero when nothing checked the map at all** -- an exactly determined fit with no correlation estimate is an assertion, not a measurement, and this is where that shows.
   */
  agreement: number;
  /**
   * Mean confidence of the instants the map was fitted to.
   */
  anchors: number;
  /**
   * How much of the map's accuracy survives to the far edges of the overlap, where its evidence is not. The uncertainty at the pivot over the worst uncertainty anywhere in the overlap, so it measures *only* the cost of extrapolation and does not restate what `agreement` already said about how well the anchors fit.
   *
   * An offset-only map scores 1.0 by construction: a constant offset is exactly as good a thousand frames away as at its anchors. A fitted rate pays here for the extrapolation it enables, and that payment is the whole reason `min_rate_span_s` and `rate_significance` exist.
   */
  stability: number;
}
/**
 * The independent estimate, and what it had to work with.
 *
 * Cross-correlating two *projected* speed signals assumes their temporal shapes
 * match, which they do not exactly: a face-on camera sees the full sweep of the
 * hands and a down-the-line camera sees much of it foreshortened, so the same
 * swing produces two differently-shaped curves. The instants survive the
 * projection -- the hands are slowest at the top and fastest near impact from
 * every angle -- which is why this works at all, and `peak_correlation` is how
 * well it worked on this pair rather than an assumption that it did.
 */
export interface CorrelationReport {
  /**
   * Pearson correlation at the best lag.
   */
  peak_correlation: number;
  /**
   * The lag that maximised it, as a target-minus-reference offset.
   */
  peak_offset_s: number;
  /**
   * The best correlation at any lag far enough from the peak to be a different alignment. None when no such lag scored at all.
   */
  rival_correlation?: number | null;
  /**
   * Where that rival sat. A swing has two speed humps, so a rival is expected.
   */
  rival_offset_s?: number | null;
  /**
   * Resampling interval both signals were put on: the finer of the two clips'.
   */
  grid_interval_s: number;
  /**
   * Seconds of signal both clips supplied at the best lag.
   */
  overlap_s: number;
  /**
   * Grid samples that contributed to the peak correlation.
   */
  samples: number;
  /**
   * How far the parabolic refinement moved the peak off the grid. Bounded by half a grid step by construction; a value at that bound means the refinement saturated and the peak is not well shaped.
   */
  sub_grid_shift_s: number;
}
/**
 * What the two clips have in common once the map is applied.
 *
 * Stated in reference-clip time, because that is the frame the map is
 * parameterised in and a reader comparing it against `ClipRef.duration_s` needs
 * the two in the same clock.
 */
export interface OverlapReport {
  start_s: number;
  end_s: number;
  duration_s: number;
  /**
   * Fraction of the reference clip inside the overlap.
   */
  reference_fraction: number;
  /**
   * Fraction of the target clip inside the overlap.
   */
  target_fraction: number;
}
/**
 * Policy for synchronisation.
 *
 * Every bound here is **stated policy except where it says otherwise**, in the
 * same sense as `GapPolicy`: none of it is tuned against a ground-truth pair,
 * because no simultaneous two-camera recording exists in this project yet.
 * `scripts/benchmark_sync.py` measures what the method achieves against known
 * offsets, which is a different and weaker claim, and says so.
 */
export interface SyncConfig {
  /**
   * Force a method. None picks events when both clips yielded a swing and correlation otherwise, and cross-checks whichever it picked against the other.
   */
  method?: SyncMethod | null;
  /**
   * Anchors must span at least this long before a rate is fitted. Below it, the rate's own error does more damage than assuming the clocks agree: a frame of anchor error over half a second is a 6% rate at 30 fps, which is 60 ms of drift a second later.
   */
  min_rate_span_s?: number;
  /**
   * Take the offset from the cross-correlation when no clock rate was needed, keeping the events for the anchors and the residual.
   *
   * **Measured, and it overturned the obvious default.** Four swing events look like four clocks and are not: `scripts/benchmark_sync.py` perturbs the landmarks at the noise level Phase 3 measured from real footage and watches where each event lands. Over eight seeds the top moves 8 ms, impact 117 ms, the finish 350 ms and the takeaway 542 ms -- the last two are threshold crossings on a signal that is barely moving there, and one seed puts the takeaway on frame zero. A correlation over the same pair averages several hundred samples and recovers the offset to under a millisecond, against 8-31 ms for the anchored fit.
   *
   * So the offset comes from the estimator with the evidence behind it. This is not averaging two estimates -- which this layer refuses to do -- it is using each for the quantity it can determine: a correlation cannot produce a rate or a residual, and four instants cannot out-measure three hundred samples of the same offset.
   *
   * It does not apply when a rate *was* fitted. A correlation yields one lag for the whole clip and a rate-bearing map has a different offset at every instant, so substituting one into the other would introduce exactly the bias `SyncQuality.method_disagreement_ms` documents. There the events fit stands whole.
   */
  prefer_correlated_offset?: boolean;
  /**
   * How many of its own standard errors a fitted rate must sit away from 1.0 to be kept. Below it the anchors are consistent with two clocks running in step, and fitting the difference anyway makes the map *worse*: it spends a degree of freedom on noise and buys an uncertainty that grows with distance from the anchors, where a constant offset has none. Two sigma, so a rate is kept when the evidence for it would not be dismissed as chance.
   */
  rate_significance?: number;
  /**
   * A fitted rate further than this from 1.0 is refused and the map falls back to offset-only. Half again is already absurd for two clocks; the bound is there to stop a bad anchor pair producing a map that is wrong everywhere rather than wrong in one place.
   */
  max_rate_deviation?: number;
  /**
   * A fitted rate further than this from 1.0 earns a warning. Consumer camera clocks agree to far better than 2%, so a rate outside it is evidence about the supplied slow-motion factors, not about the hardware.
   */
  clock_drift_tolerance?: number;
  /**
   * A candidate lag is not scored unless both clips supply this much signal there. Without it the extreme lags, where a handful of samples overlap, produce spuriously perfect correlations.
   */
  min_overlap_s?: number;
  /**
   * Largest offset the correlation will consider. None searches every lag that satisfies `min_overlap_s`, which is the honest default: two cameras started by hand can be any distance apart.
   */
  max_lag_s?: number | null;
  /**
   * Interval both hand-speed signals are resampled onto before correlating. None uses the finer of the two clips', which is what lets the lag search resolve below the coarser clip's frame interval -- the only thing the faster camera of a mismatched pair contributes to the alignment.
   */
  grid_interval_s?: number | null;
  /**
   * How far from the peak a lag must be to count as a different alignment rather than part of the same peak.
   */
  rival_separation_s?: number;
  /**
   * Above this residual the alignment is refused rather than reported with a low score. A tenth of a second is most of a downswing, so a map that wrong is not a worse answer but a different swing. Policy, not a measured optimum.
   */
  max_residual_ms?: number;
  /**
   * An event pair whose offset differs from the median pair's by more than this is dropped before fitting, and said so. One misdetected event should cost its own anchor, not the whole alignment.
   *
   * **Measured, not assumed.** The bound has to admit the spread a genuine clock rate produces while catching an event located in the wrong place, and those two overlap on paper: a rate of 0.7 over a 1.6 s swing moves the anchors 0.4 s apart. In practice they do not, because the median sits among the clustered anchors rather than at one end. `scripts/benchmark_sync.py` sweeps the supplied slow-motion factor from 2% to 40% wrong and keeps all four anchors at this bound, recovering the rate to within 0.3%; the same bound drops the spurious takeaway a clip that starts mid-backswing produces, which sits 0.47 s out.
   */
  max_anchor_disagreement_s?: number;
}
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
  family?: BoardFamily & string;
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
  /**
   * Which coefficients to fit. Four rather than OpenCV's five, on the measurement: with the board reaching the frame edge the fifth term buys almost nothing, and without it the fifth term is fitted to noise and does its damage at the corners where the swing is. `scripts/benchmark_calibration.py --sweep distortion` has the numbers, and a fisheye lens is the case for overriding this.
   */
  distortion_model?: DistortionModel & string;
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
