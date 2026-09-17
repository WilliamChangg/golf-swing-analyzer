"""Typed contracts for two-camera synchronisation.

Two cameras filming one swing keep two clocks, and neither clock knows about the
other. Each was started by hand at an arbitrary moment; each runs at its own
nominal rate, which the container reports and the hardware only approximates;
and either may be slow motion, whose factor nothing in the file records. Before a
single quantity can be computed from both views at once, those two clocks have to
be related to each other -- and the relation has to carry how well it is known,
because every later phase inherits the error in it.

## The map is affine, and its second parameter is refused by default

    target_s = reference_s + offset_s + (rate - 1) * (reference_s - pivot_s)

`offset_s` says how far ahead the target clip's clock runs at one named instant.
`rate` says how fast it runs relative to the reference: 1.0 is two clocks in
step, and the map collapses to a constant offset.

Estimating both from a swing is tempting and usually wrong. A swing's events span
a second or two, and each is located to about a frame; a rate fitted across that
short a baseline carries a fractional error of roughly a frame divided by the
span, which at 30 fps over 1.5 s is 2%. Applied ten seconds from the anchors that
is 200 ms of sync error -- far worse than assuming the two clocks agree, which
for real hardware they do to a small fraction of a percent. So `rate` stays at
1.0 unless the anchors span `SyncConfig.min_rate_span_s`, and `rate_estimated`
says which happened. Nothing has to infer it from the value.

**The pivot is not decoration.** Stating the offset at the centroid of the
anchors rather than at time zero makes the two fitted parameters uncorrelated, so
`offset_uncertainty_s` and `rate_uncertainty` can be reported as two independent
numbers and combined by `TimeMap.uncertainty_at`. Quoted at time zero they would
be strongly correlated, and reporting them separately would overstate the error
near the anchors and understate it far from them.

## The residual is doing two jobs

With four swing events located in both clips there are four anchor pairs and two
free parameters, so the fit has two degrees of freedom and its residual is a real
measurement. It answers the obvious question -- how well the map fits -- and a
less obvious one: **whether the two clips show the same swing at all.**

Nothing in this layer can see that a face-on camera and a down-the-line camera
were pointed at the same event. It aligns swing-shaped signals, and it will align
two *different* swings just as willingly. What it cannot do is make their phase
durations match: two swings of different tempo cannot be brought into agreement
by any offset and any rate, so a mismatch shows up as residual. That is the only
evidence available, so it is reported rather than collapsed into the headline.

The corollary is the trap to avoid. A fit with as many anchors as parameters has
a residual of zero **by construction**, and a zero there means nothing was
checked -- the same trap `SmoothingConfig.min_observations` documents one layer
down. `SyncQuality.degrees_of_freedom` is how a reader tells the two cases apart,
and an exactly-determined fit is corroborated against the cross-correlation
estimate instead, or scores zero on agreement if there is nothing to check it
with.

## Two estimates, deliberately not averaged

Cross-correlation of the hand-speed signals and a fit to the detected events are
independent: one uses the whole trajectory and no golf knowledge beyond which
landmark to track, the other uses four identified instants and nothing between
them. Where they agree to within the frame quantisation, both are probably right.
Where they disagree by a tenth of a second, one of them has locked onto the wrong
feature -- a swing's speed signal has two humps, and a correlation lag that puts
one clip's backswing over the other's downswing is a real local maximum.

They are therefore kept apart and their difference is reported. Averaging two
estimates that disagree produces a third number that matches neither and hides
the fact that something is wrong.

## What a second camera makes measurable

Phase 6 recorded that a slow-motion factor "is supplied, not measured, and
nothing in the video can recover it". That is true of one clip and false of two:
if both factors are supplied and the fitted `rate` comes out at 1.14 with a small
residual, the *ratio* of the two supplied factors is wrong by 14%. Real camera
clocks do not differ by anything approaching that, so a rate far from 1.0 is
reported as a slow-motion problem rather than as a clock difference.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.phases import SwingEvent

# Bump on any change that alters what a stored or reported sync means. The
# projects store keys on this: a row written under a version it does not
# understand is refused rather than parsed hopefully.
SYNC_SCHEMA_VERSION = 1


class SyncMethod(StrEnum):
    """What produced a time map.

    MANUAL
        Anchors supplied by a person, who looked at both clips and said which
        frames show the same instant. The engine does the arithmetic and the
        error accounting; it does not second-guess the instants.

    EVENTS
        Anchors from Phase 4's detected swing events, paired by name. The only
        method that can estimate a rate, because it is the only one that
        produces anchors spread across the swing.

    CORRELATION
        The lag that maximises the normalised correlation of the two hand-speed
        signals. Produces one number -- an offset -- and no rate.

    COMBINED
        The default where both are available, and each is used for what it can
        actually determine: the **offset from the correlation**, the anchors and
        the residual from the events. See `SyncConfig.prefer_correlated_offset`
        for the measurement that decided this.
    """

    MANUAL = "manual"
    EVENTS = "events"
    CORRELATION = "correlation"
    COMBINED = "combined"


class AnchorSource(StrEnum):
    """Where one anchor's pair of frames came from.

    Kept distinct from `SyncMethod` because a single fit may mix them: a person
    correcting one badly-detected event leaves the other three as detections,
    and the result is neither wholly manual nor wholly automatic.
    """

    DETECTED = "detected"
    MANUAL = "manual"


class ClipRef(BaseModel):
    """One clip of a synchronised pair, as the sync layer needs to see it.

    Timings here are **real seconds**, already divided by the slow-motion factor,
    matching every other duration in the engine. `slow_motion_factor` is carried
    alongside so a reader can recover the clip's own playback clock, and because
    a fitted rate far from 1.0 usually means one of these two numbers is wrong.
    """

    path: str
    name: str = Field(description="Basename, for display.")
    frames: int
    start_s: float = Field(description="Timestamp of the first frame, in real seconds.")
    duration_s: float = Field(description="Span from the first to the last frame, in real seconds.")
    median_interval_s: float = Field(
        description=(
            "Median gap between frames, in real seconds. The quantisation of "
            "every instant located in this clip, and half of what bounds how "
            "well the pair can be aligned."
        )
    )
    slow_motion_factor: float = Field(
        gt=0.0, description="The factor the caller supplied for this clip. 1.0 is real time."
    )


class TimeMap(BaseModel):
    """The affine map from reference-clip time to target-clip time.

    Both directions are provided because both are needed: rendering the target
    clip alongside the reference needs `to_target`, and reporting an instant
    found in the target in the reference's frame numbering needs `to_reference`.
    """

    offset_s: float = Field(
        description=(
            "How far ahead of the reference clock the target clock runs, at "
            "`pivot_s`. With `rate` at 1.0 this is the whole map."
        )
    )
    rate: float = Field(
        gt=0.0,
        description=(
            "Target seconds per reference second. 1.0 means the two clocks run "
            "at the same speed, which is the default and is not a measurement "
            "unless `rate_estimated` is true."
        ),
    )
    rate_estimated: bool = Field(
        description=(
            "True when the rate was fitted from anchors spanning enough time to "
            "determine it. False means it was held at 1.0, which is an "
            "assumption about the hardware rather than something measured here."
        )
    )
    pivot_s: float = Field(
        description=(
            "The reference-clip instant `offset_s` is stated at: the centroid of "
            "the anchors. Chosen so the offset and rate uncertainties are "
            "uncorrelated and can be combined by `uncertainty_at`."
        )
    )
    offset_uncertainty_s: float | None = Field(
        default=None,
        description=(
            "Standard error of `offset_s`, from the fit's residual scatter where "
            "there are spare degrees of freedom and from the frame quantisation "
            "otherwise. None where neither is available."
        ),
    )
    rate_uncertainty: float | None = Field(
        default=None,
        description="Standard error of `rate`. None whenever the rate was not estimated.",
    )
    support_start_s: float = Field(description="Earliest reference time an anchor was placed at.")
    support_end_s: float = Field(description="Latest reference time an anchor was placed at.")

    def to_target(self, reference_s: float) -> float:
        """The target-clip time corresponding to a reference-clip time."""
        return reference_s + self.offset_s + (self.rate - 1.0) * (reference_s - self.pivot_s)

    def to_reference(self, target_s: float) -> float:
        """The reference-clip time corresponding to a target-clip time."""
        return self.pivot_s + (target_s - self.pivot_s - self.offset_s) / self.rate

    def uncertainty_at(self, reference_s: float) -> float | None:
        """How well the map is known at one instant, in seconds.

        Grows linearly with distance from the pivot whenever a rate was
        estimated, which is the quantitative form of "a rate fitted over a short
        baseline should not be extrapolated far". Constant otherwise, because a
        pure offset is equally good everywhere.

        The two terms are added in quadrature, which is only valid because the
        pivot is the anchor centroid; at any other pivot they would be
        correlated and this would understate the total.
        """
        if self.offset_uncertainty_s is None:
            return None
        lever = abs(reference_s - self.pivot_s) * (self.rate_uncertainty or 0.0)
        return math.hypot(self.offset_uncertainty_s, lever)

    def inverse(self) -> TimeMap:
        """The same relation, stated from the target clip's point of view.

        Needed because "which clip is the reference" is a free choice above this
        layer and a recorded one below it. Phase 9 reconstructs into whichever
        camera's frame the caller names, and a project stores one alignment per
        ordered pair; without this, asking for the other order would mean
        re-fitting a relation that is already known.

        The pivot moves to `to_target(pivot_s)`, which is the same instant on the
        other clock. That is not decoration either: the pivot is the anchor
        centroid precisely so the two uncertainties are uncorrelated, and a pivot
        left at the original value would re-correlate them and make
        `uncertainty_at` understate the total away from the anchors.
        """
        return TimeMap(
            offset_s=-self.offset_s,
            rate=1.0 / self.rate,
            rate_estimated=self.rate_estimated,
            pivot_s=self.pivot_s + self.offset_s,
            # An offset in the other clock's seconds, and a rate that is the
            # reciprocal: both propagated to first order, which is exact enough
            # for a rate this close to 1 and honest about being a propagation.
            offset_uncertainty_s=(
                None if self.offset_uncertainty_s is None else self.offset_uncertainty_s / self.rate
            ),
            rate_uncertainty=(
                None if self.rate_uncertainty is None else self.rate_uncertainty / (self.rate**2)
            ),
            support_start_s=self.to_target(self.support_start_s),
            support_end_s=self.to_target(self.support_end_s),
        )

    @property
    def extrapolates_beyond(self) -> float:
        """Seconds of anchored span. Zero when every anchor sits at one instant."""
        return self.support_end_s - self.support_start_s


class SyncAnchor(BaseModel):
    """One instant identified in both clips.

    Frames and seconds are both carried. Frames are what a person picks and what
    a UI scrubs to; seconds are what the map is fitted in, and on
    variable-rate footage the two are not interconvertible without the clip's own
    timestamps -- which live in the engine, not in the caller.
    """

    label: str = Field(description="What this instant is, in words. Shown to a reader.")
    event: SwingEvent | None = Field(
        default=None,
        description="The swing event this anchor is, when it is one. None for a manual pick.",
    )
    source: AnchorSource
    reference_frame: int
    target_frame: int
    reference_s: float
    target_s: float
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well this instant was located, in the weaker of the two clips. "
            "A detected anchor inherits Phase 4's event confidence; a manual "
            "anchor is the assertion being fitted rather than an estimate of it, "
            "and scores 1.0 -- what checks it is the residual, not this."
        ),
    )


class AnchorResidual(BaseModel):
    """How far one anchor sits from the fitted map.

    Signed, and in milliseconds, because the sign is the informative part: four
    residuals that alternate are noise, and four that trend are a rate the fit
    was not allowed to take up.
    """

    label: str
    reference_s: float
    observed_target_s: float
    predicted_target_s: float
    residual_ms: float = Field(description="observed - predicted, in milliseconds.")


class CorrelationReport(BaseModel):
    """The independent estimate, and what it had to work with.

    Cross-correlating two *projected* speed signals assumes their temporal shapes
    match, which they do not exactly: a face-on camera sees the full sweep of the
    hands and a down-the-line camera sees much of it foreshortened, so the same
    swing produces two differently-shaped curves. The instants survive the
    projection -- the hands are slowest at the top and fastest near impact from
    every angle -- which is why this works at all, and `peak_correlation` is how
    well it worked on this pair rather than an assumption that it did.
    """

    peak_correlation: float = Field(
        ge=-1.0, le=1.0, description="Pearson correlation at the best lag."
    )
    peak_offset_s: float = Field(
        description="The lag that maximised it, as a target-minus-reference offset."
    )
    rival_correlation: float | None = Field(
        default=None,
        description=(
            "The best correlation at any lag far enough from the peak to be a "
            "different alignment. None when no such lag scored at all."
        ),
    )
    rival_offset_s: float | None = Field(
        default=None,
        description="Where that rival sat. A swing has two speed humps, so a rival is expected.",
    )
    grid_interval_s: float = Field(
        description="Resampling interval both signals were put on: the finer of the two clips'."
    )
    overlap_s: float = Field(description="Seconds of signal both clips supplied at the best lag.")
    samples: int = Field(description="Grid samples that contributed to the peak correlation.")
    sub_grid_shift_s: float = Field(
        description=(
            "How far the parabolic refinement moved the peak off the grid. "
            "Bounded by half a grid step by construction; a value at that bound "
            "means the refinement saturated and the peak is not well shaped."
        )
    )


class OverlapReport(BaseModel):
    """What the two clips have in common once the map is applied.

    Stated in reference-clip time, because that is the frame the map is
    parameterised in and a reader comparing it against `ClipRef.duration_s` needs
    the two in the same clock.
    """

    start_s: float
    end_s: float
    duration_s: float
    reference_fraction: float = Field(
        ge=0.0, le=1.0, description="Fraction of the reference clip inside the overlap."
    )
    target_fraction: float = Field(
        ge=0.0, le=1.0, description="Fraction of the target clip inside the overlap."
    )


class SyncQuality(BaseModel):
    """How well the alignment is determined, in milliseconds.

    `quantisation_floor_ms` is the bound nothing about this pair can beat. An
    instant located to the nearest frame carries a uniform error of a frame
    interval, whose standard deviation is that interval over sqrt(12); two clips
    contribute one each, in quadrature. A residual at the floor means the anchors
    agree as well as their frame rates allow, and the coarser clip is what sets
    it -- pairing a 240 fps camera with a 30 fps one buys almost nothing.
    """

    residual_rms_ms: float | None = Field(
        default=None,
        description=(
            "RMS of the anchor residuals. **None when the fit has no spare "
            "degrees of freedom**, where the residual is zero by construction "
            "and reporting it would look like a perfect alignment."
        ),
    )
    residual_max_ms: float | None = Field(
        default=None, description="Largest absolute anchor residual, on the same condition."
    )
    degrees_of_freedom: int = Field(
        description="Anchors minus fitted parameters. Zero or less means the residual is not evidence."
    )
    quantisation_floor_ms: float = Field(
        description="Best residual the two clips' frame rates permit. See the class docstring."
    )
    method_disagreement_ms: float | None = Field(
        default=None,
        description=(
            "Distance between the fitted offset and the cross-correlation's, at "
            "the pivot. None when only one estimate exists, which is itself worth "
            "knowing: nothing independent checked the map.\n\n"
            "**Comparable only when the map is offset-only.** A correlation "
            "produces one lag for the whole clip; a map with a fitted rate has a "
            "different offset at every instant, and the two are then estimating "
            "different quantities. A rate of 0.91 moves the map by 90 ms per "
            "second away from the pivot, so a correlation landing 9 ms from the "
            "pivot value is close agreement rather than poor. It is reported in "
            "both cases and drives `SyncConfidence.agreement` in neither, once a "
            "rate is in play -- there the residual is the better instrument, "
            "being the only one that is rate-aware."
        ),
    )


class SyncConfidence(BaseModel):
    """Why an alignment is or is not trustworthy, decomposed.

    Three measured factors and their product, following Phase 4 and Phase 5. The
    product is the headline because an alignment needs all three; the factors are
    what make a low score actionable, and they fail for different reasons that
    call for different fixes -- re-film, re-pick the anchors, or supply the right
    slow-motion factor.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    agreement: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Evidence about whichever estimator actually produced the offset, "
            "which is a different quantity for the two kinds of map.\n\n"
            "Where the anchors produced it (`events`, `manual`): how closely "
            "they agree with the map, measured against the quantisation floor "
            "rather than against zero, taken at its worst alongside the "
            "cross-correlation's independent estimate.\n\n"
            "Where the correlation produced it (`combined`, `correlation`): how "
            "far the peak sat above its nearest rival. Under realistic landmark "
            "noise the anchor residual cannot do this job -- four events scatter "
            "tens of milliseconds on a pair that is correctly aligned to under "
            "one, and scatter no more than that on two clips of genuinely "
            "different swings. The correlation margin separates those cases and "
            "the residual does not; `scripts/benchmark_sync.py` has the numbers.\n\n"
            "**Zero when nothing checked the map at all** -- an exactly "
            "determined fit with no correlation estimate is an assertion, not a "
            "measurement, and this is where that shows."
        ),
    )
    anchors: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean confidence of the instants the map was fitted to.",
    )
    stability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How much of the map's accuracy survives to the far edges of the "
            "overlap, where its evidence is not. The uncertainty at the pivot "
            "over the worst uncertainty anywhere in the overlap, so it measures "
            "*only* the cost of extrapolation and does not restate what "
            "`agreement` already said about how well the anchors fit.\n\n"
            "An offset-only map scores 1.0 by construction: a constant offset is "
            "exactly as good a thousand frames away as at its anchors. A fitted "
            "rate pays here for the extrapolation it enables, and that payment "
            "is the whole reason `min_rate_span_s` and `rate_significance` exist."
        ),
    )


class SyncConfig(BaseModel, extra="forbid"):
    """Policy for synchronisation.

    Every bound here is **stated policy except where it says otherwise**, in the
    same sense as `GapPolicy`: none of it is tuned against a ground-truth pair,
    because no simultaneous two-camera recording exists in this project yet.
    `scripts/benchmark_sync.py` measures what the method achieves against known
    offsets, which is a different and weaker claim, and says so.
    """

    method: SyncMethod | None = Field(
        default=None,
        description=(
            "Force a method. None picks events when both clips yielded a swing "
            "and correlation otherwise, and cross-checks whichever it picked "
            "against the other."
        ),
    )
    min_rate_span_s: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Anchors must span at least this long before a rate is fitted. Below "
            "it, the rate's own error does more damage than assuming the clocks "
            "agree: a frame of anchor error over half a second is a 6% rate at "
            "30 fps, which is 60 ms of drift a second later."
        ),
    )
    prefer_correlated_offset: bool = Field(
        default=True,
        description=(
            "Take the offset from the cross-correlation when no clock rate was "
            "needed, keeping the events for the anchors and the residual.\n\n"
            "**Measured, and it overturned the obvious default.** Four swing "
            "events look like four clocks and are not: "
            "`scripts/benchmark_sync.py` perturbs the landmarks at the noise "
            "level Phase 3 measured from real footage and watches where each "
            "event lands. Over eight seeds the top moves 8 ms, impact 117 ms, "
            "the finish 350 ms and the takeaway 542 ms -- the last two are "
            "threshold crossings on a signal that is barely moving there, and "
            "one seed puts the takeaway on frame zero. A correlation over the "
            "same pair averages several hundred samples and recovers the offset "
            "to under a millisecond, against 8-31 ms for the anchored fit.\n\n"
            "So the offset comes from the estimator with the evidence behind "
            "it. This is not averaging two estimates -- which this layer "
            "refuses to do -- it is using each for the quantity it can "
            "determine: a correlation cannot produce a rate or a residual, and "
            "four instants cannot out-measure three hundred samples of the same "
            "offset.\n\n"
            "It does not apply when a rate *was* fitted. A correlation yields "
            "one lag for the whole clip and a rate-bearing map has a different "
            "offset at every instant, so substituting one into the other would "
            "introduce exactly the bias `SyncQuality.method_disagreement_ms` "
            "documents. There the events fit stands whole."
        ),
    )
    rate_significance: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "How many of its own standard errors a fitted rate must sit away "
            "from 1.0 to be kept. Below it the anchors are consistent with two "
            "clocks running in step, and fitting the difference anyway makes the "
            "map *worse*: it spends a degree of freedom on noise and buys an "
            "uncertainty that grows with distance from the anchors, where a "
            "constant offset has none. Two sigma, so a rate is kept when the "
            "evidence for it would not be dismissed as chance."
        ),
    )
    max_rate_deviation: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "A fitted rate further than this from 1.0 is refused and the map "
            "falls back to offset-only. Half again is already absurd for two "
            "clocks; the bound is there to stop a bad anchor pair producing a "
            "map that is wrong everywhere rather than wrong in one place."
        ),
    )
    clock_drift_tolerance: float = Field(
        default=0.02,
        gt=0.0,
        description=(
            "A fitted rate further than this from 1.0 earns a warning. Consumer "
            "camera clocks agree to far better than 2%, so a rate outside it is "
            "evidence about the supplied slow-motion factors, not about the "
            "hardware."
        ),
    )
    min_overlap_s: float = Field(
        default=0.30,
        gt=0.0,
        description=(
            "A candidate lag is not scored unless both clips supply this much "
            "signal there. Without it the extreme lags, where a handful of "
            "samples overlap, produce spuriously perfect correlations."
        ),
    )
    max_lag_s: float | None = Field(
        default=None,
        description=(
            "Largest offset the correlation will consider. None searches every "
            "lag that satisfies `min_overlap_s`, which is the honest default: "
            "two cameras started by hand can be any distance apart."
        ),
    )
    grid_interval_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Interval both hand-speed signals are resampled onto before "
            "correlating. None uses the finer of the two clips', which is what "
            "lets the lag search resolve below the coarser clip's frame "
            "interval -- the only thing the faster camera of a mismatched pair "
            "contributes to the alignment."
        ),
    )
    rival_separation_s: float = Field(
        default=0.20,
        gt=0.0,
        description=(
            "How far from the peak a lag must be to count as a different "
            "alignment rather than part of the same peak."
        ),
    )
    max_residual_ms: float = Field(
        default=150.0,
        gt=0.0,
        description=(
            "Above this residual the alignment is refused rather than reported "
            "with a low score. A tenth of a second is most of a downswing, so a "
            "map that wrong is not a worse answer but a different swing. Policy, "
            "not a measured optimum."
        ),
    )
    max_anchor_disagreement_s: float = Field(
        default=0.30,
        gt=0.0,
        description=(
            "An event pair whose offset differs from the median pair's by more "
            "than this is dropped before fitting, and said so. One misdetected "
            "event should cost its own anchor, not the whole alignment.\n\n"
            "**Measured, not assumed.** The bound has to admit the spread a "
            "genuine clock rate produces while catching an event located in the "
            "wrong place, and those two overlap on paper: a rate of 0.7 over a "
            "1.6 s swing moves the anchors 0.4 s apart. In practice they do not, "
            "because the median sits among the clustered anchors rather than at "
            "one end. `scripts/benchmark_sync.py` sweeps the supplied "
            "slow-motion factor from 2% to 40% wrong and keeps all four anchors "
            "at this bound, recovering the rate to within 0.3%; the same bound "
            "drops the spurious takeaway a clip that starts mid-backswing "
            "produces, which sits 0.47 s out."
        ),
    )


class SyncModel(BaseModel):
    """Everything the engine concluded about aligning two clips.

    `aligned` is false whenever no map could be produced or the map produced was
    too poor to report, and then `time_map` is None rather than populated with a
    guess. A caller that reads the map without checking cannot therefore mistake
    a refusal for an alignment at zero offset, which is the failure mode this
    shape exists to prevent -- and zero offset is exactly the wrong value to
    default to, because it is also a perfectly plausible answer.
    """

    schema_version: int = SYNC_SCHEMA_VERSION
    aligned: bool
    method: SyncMethod | None = Field(
        default=None, description="What produced the map. None when nothing did."
    )
    reference: ClipRef
    target: ClipRef
    time_map: TimeMap | None = None
    anchors: list[SyncAnchor] = Field(default_factory=list)
    residuals: list[AnchorResidual] = Field(default_factory=list)
    quality: SyncQuality | None = None
    confidence: SyncConfidence | None = None
    correlation: CorrelationReport | None = Field(
        default=None,
        description="The independent estimate, when one could be computed. Never merged into the map.",
    )
    overlap: OverlapReport | None = None
    config: SyncConfig
    refusal: str | None = Field(
        default=None,
        description="Why no map was produced, in words. Set exactly when `aligned` is false.",
    )
    warnings: list[str] = Field(default_factory=list)

    def anchor(self, event: SwingEvent) -> SyncAnchor | None:
        """The anchor for a named event, or None if that event was not paired."""
        return next((entry for entry in self.anchors if entry.event is event), None)
