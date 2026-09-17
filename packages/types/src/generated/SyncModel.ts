/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

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
