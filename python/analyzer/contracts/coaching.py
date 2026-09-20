"""Typed contracts for the coaching engine.

A `Finding` is what this system says to a person. Everything below it measures;
this is the layer that draws a conclusion, and the shape of the type is chosen so
that a conclusion cannot travel without the thing that makes it checkable:

* **the measurement it is about**, by name, with the frames it came from;
* **the threshold it was compared against**, with who published it, on whom, and
  by what method;
* **the direction and the margin**, in the metric's own unit;
* **the bracket the comparison had to clear**; and
* **prose that contains no number the evidence does not.**

## Why a threshold needs a provenance

Golf coaching runs on numbers with no papers behind them. "Ninety degrees of
shoulder turn", "forty-five degrees of X-factor", "three to one tempo" -- these
are quoted everywhere, and the three have completely different standing: one is
an instruction-book convention, one is a book's summary of a measurement nobody
publishes the protocol for, and one is a frame count off broadcast video that can
be reproduced. A system that hard-codes all three as `THRESHOLD = 90.0` has
flattened that difference, and the number it prints afterwards looks equally
authoritative in each case.

So `ThresholdSource` is a required field on every rule, and the field that does
the most work is `permitted_bases`. A threshold is a number about a *particular
measured quantity*. A shoulder turn from an electromagnetic sensor on the spine
and a shoulder turn inferred from how much a line shortened in one photograph are
not the same quantity, they disagree by tens of degrees, and comparing the second
against a threshold set for the first is not a measurement with extra error -- it
is a category mistake that produces a confident sentence.

## Why a convention permits nothing

`ThresholdMethod.CONVENTION` sources declare `permitted_bases` empty, so a rule
resting on one can never fire. That is not an oversight to be fixed by finding a
citation later; it is the honest reading of what those numbers are. A number with
no stated measurement protocol has no quantity attached to it, and therefore
nothing this engine measures can be compared against it.

They are in the registry anyway, refused by name. A coaching engine that silently
omitted the eight numbers every golf app displays would look like it had not
thought of them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.metrics import CameraView, MetricBasis, MetricName, MetricUnit
from analyzer.contracts.phases import SwingEvent, SwingPhase

# Bump on any change that alters what a reported finding means.
COACHING_SCHEMA_VERSION = 1


class ThresholdMethod(StrEnum):
    """How the source of a threshold measured the number it published.

    The field that decides which of this engine's measurements a borrowed number
    may be compared against, because it decides what the number is *about*.

    THREE_D_MOTION_CAPTURE
        Optical or electromagnetic capture of markers in three dimensions. The
        quantity is an angle or a distance in space, so it may only be compared
        against a `SPATIAL` measurement -- which in this build means a stereo
        calibration and a triangulated pair.

    INSTRUMENTED_SENSOR
        A device worn on the body: a spine-mounted goniometer, an inertial unit,
        a pressure mat. Also a measurement of the body rather than of a picture
        of it, and so also `SPATIAL`.

    TWO_D_VIDEO
        Measured off video, which is what this engine has. A duration measured
        this way transfers directly, because a clock does not care where the
        camera stood. An *angle* measured this way transfers only to footage from
        the same camera position, which is why `filmed_from` exists and why a
        source that does not state it cannot supply an angular threshold.

    CONVENTION
        Published without a measurement protocol: instruction books, coaching
        received wisdom, the numbers on the back of a training aid. Permits no
        basis at all. See the module docstring.

    SAME_CLIP
        No borrowed number at all. The comparison is between two measurements of
        the same quantity, made the same way, by the same camera, on the same
        swing, and what has to be cleared is their own combined uncertainty. The
        only kind of threshold here that carries no population -- and therefore
        the only one that says nothing about anybody else's swing.

        What survives a projection is the **direction** of the change and not its
        size: a foreshortening map is monotone, so a line that turned further
        really did turn further, and by how much is a statement about the picture.
        Rules using this say so in their own words.
    """

    THREE_D_MOTION_CAPTURE = "three_d_motion_capture"
    INSTRUMENTED_SENSOR = "instrumented_sensor"
    TWO_D_VIDEO = "two_d_video"
    CONVENTION = "convention"
    SAME_CLIP = "same_clip"


class ThresholdSource(BaseModel):
    """Where a threshold came from, and what it is therefore a threshold on."""

    citation: str = Field(description="Author, work and year, as it would be written out.")
    year: int | None = Field(default=None, description="Publication year, where there is one.")
    population: str = Field(
        description=(
            "Who the number was measured on, in words. A threshold set on tour "
            "professionals says nothing about whether an amateur's swing is "
            "working; it says how far it sits from theirs."
        )
    )
    sample_size: int | None = Field(
        default=None,
        description="How many people, where the source states it. None means it does not.",
    )
    method: ThresholdMethod
    measures: str = Field(
        description=(
            "What the source actually measured, in its own terms -- not what this "
            "engine measures. The two being different is the normal case and the "
            "reason `permitted_bases` is not derived from the metric."
        )
    )
    permitted_bases: list[MetricBasis] = Field(
        default_factory=list,
        description=(
            "The bases of this engine's measurements that this number may be "
            "compared against. Empty is a valid and common answer: it means the "
            "source measured something nothing here produces, and every rule "
            "resting on it is refused rather than approximated."
        ),
    )
    filmed_from: CameraView | None = Field(
        default=None,
        description=(
            "For a `TWO_D_VIDEO` source, the camera position it measured from. "
            "Required before any angular or image-plane threshold from it may be "
            "used, because a projected angle is a fact about a camera position as "
            "much as about a body. None on a source that states only durations, "
            "which no camera position changes."
        ),
    )
    note: str = Field(
        default="",
        description="Anything a reader needs in order not to over-read the number.",
    )


class Comparison(StrEnum):
    """Where a measured value sat relative to a published band.

    `WITHIN` is a finding, not a silence. "This sits inside the range the source
    reports" is an observation with evidence behind it, and a coaching engine that
    only spoke when something was wrong would be reporting a selection rather than
    a measurement.
    """

    BELOW = "below"
    WITHIN = "within"
    ABOVE = "above"


class FindingRefusal(StrEnum):
    """Why a rule that was considered produced nothing.

    Every one of these is a different thing to do about it, which is why they are
    a typed set and not a sentence. `BASIS_NOT_PERMITTED` is fixed by a second
    camera and a calibration; `NO_THRESHOLD` is fixed by somebody publishing one;
    `UNRESOLVED` is fixed by a faster camera; `NO_METRIC` is usually fixed by
    moving the one that filmed it.

    `SUPPLIED_TIMEBASE` is the one that cannot be fixed by anybody holding the
    camera afterwards. A slow-motion clip's playback factor is not recorded in
    the file; it is supplied by whoever ran the analysis, and every duration
    measured from that clip is that supplied number multiplied by something
    measured. Comparing one against a published band in seconds compares a
    stopwatch against a guess. A **ratio** of two durations from the same clip
    survives it, because the factor divides out -- the same shape of argument
    that makes a length in torso lengths survive an unknown camera, one dimension
    over.
    """

    NO_SWING = "no_swing"
    NO_METRIC = "no_metric"
    NO_THRESHOLD = "no_threshold"
    BASIS_NOT_PERMITTED = "basis_not_permitted"
    VIEW_MISMATCH = "view_mismatch"
    SUPPLIED_TIMEBASE = "supplied_timebase"
    LOW_CONFIDENCE = "low_confidence"
    NO_UNCERTAINTY = "no_uncertainty"
    UNRESOLVED = "unresolved"


class Evidence(BaseModel):
    """One measured quantity a finding rests on, with the frames to check it in.

    The link the exit criterion is about: metric -> frames -> overlay. `frames`
    are frame indices into **this clip's video file**, so a player can be shown
    the picture the number came from; `timestamps_s` are the matching instants on
    a real clock, which on a slow-motion clip is a different thing entirely and
    does not index the file.
    """

    metric: MetricName
    label: str = Field(description="The metric's own display label, including its anchor.")
    value: float
    unit: MetricUnit
    basis: MetricBasis
    event: SwingEvent | None = None
    phase: SwingPhase | None = None
    uncertainty: float | None = Field(
        default=None, description="As the metric reported it, in the metric's own unit."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    frames: list[int] = Field(
        description="Every frame whose landmarks entered this value. Indices into the video."
    )
    timestamps_s: list[float] = Field(
        default_factory=list,
        description=(
            "Real-clock instants for `frames`, where the detection supplied them. "
            "Empty rather than interpolated when it did not."
        ),
    )
    methodology: str = Field(description="How the number was produced, carried from the metric.")


class Finding(BaseModel):
    """One conclusion, with everything needed to check it.

    There is no severity and no score. A number that ranked findings against each
    other would be the single most quoted output of this system and the least
    defensible one: it would need a scale relating degrees of shoulder turn to
    seconds of tempo, and nothing has measured one.
    """

    rule_id: str = Field(description="Stable identifier, so a UI can link to one finding.")
    title: str
    observation: str = Field(
        description=(
            "What was measured and how it compares, in words. Contains no number "
            "that is not in this finding's own evidence -- enforced by the same "
            "guard the phrasing layer is held to."
        )
    )
    comparison: Comparison
    value: float
    unit: MetricUnit
    bracket: float = Field(
        ge=0.0,
        description=(
            "The uncertainty the comparison had to clear, in the metric's unit. "
            "A value inside `bracket` of a band edge is refused as `UNRESOLVED` "
            "rather than reported as being on one side of it."
        ),
    )
    band_low: float | None = None
    band_high: float | None = None
    margin: float = Field(
        ge=0.0,
        description=(
            "How far outside the nearer band edge the value sat, in its own unit. "
            "Zero for a `WITHIN` finding."
        ),
    )
    source: ThresholdSource
    evidence: list[Evidence] = Field(
        min_length=1,
        description=(
            "At least one, always. A finding with no evidence cannot be produced "
            "by this engine -- the contract refuses it rather than the review."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The weakest confidence among the metrics this finding rests on.",
    )
    phrased: str | None = Field(
        default=None,
        description=(
            "The phrasing layer's rewording of `observation`, present only when a "
            "model produced one **and** it passed the guard. `observation` is "
            "always there and is always what was measured; this is presentation."
        ),
    )


class RefusedFinding(BaseModel):
    """A rule that was considered and deliberately produced nothing.

    Reported rather than omitted, for the reason `RefusedMetric` is: a rule that
    did not fire because the swing was fine and a rule that could not be applied
    at all look identical in a list of findings, and they mean opposite things.
    """

    rule_id: str
    title: str
    refusal: FindingRefusal
    reason: str
    source: ThresholdSource


class PhrasingMode(StrEnum):
    """Where a finding's words come from.

    OFF
        The rule's own sentence, assembled from the evidence. The default, and
        the only mode that needs nothing installed.
    LOCAL
        A language model running on this machine, handed the findings as
        structured data and nothing else. Never the video, never the landmarks,
        never a file path.
    """

    OFF = "off"
    LOCAL = "local"


class GuardRejection(BaseModel):
    """One reason a candidate phrasing was thrown away.

    Kept in the report rather than logged, because the rejection rate is the only
    evidence anybody has about whether the phrasing layer is safe to turn on. A
    model that is rewritten into silence nine times in ten is not a phrasing
    layer, and the number that says so has to survive to the reader.
    """

    rule_id: str
    offence: str = Field(description="Which check failed, in a few words.")
    token: str = Field(description="The text that failed it.")
    candidate: str = Field(description="The full candidate, kept so it can be inspected.")


class PhrasingReport(BaseModel):
    """What the optional language layer did, if anything.

    `attempted` is zero in the default configuration and that is the expected
    state. Every count here is present so that "the model rephrased nothing" and
    "the model was never asked" cannot be confused, which is the same distinction
    `RefusedMetric` exists for one layer down.
    """

    mode: PhrasingMode = PhrasingMode.OFF
    provider: str | None = Field(
        default=None, description="What produced the candidates. None when nothing did."
    )
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    rejections: list[GuardRejection] = Field(default_factory=list)
    unavailable: str | None = Field(
        default=None,
        description=(
            "Why no model was reached, when one was asked for. A phrasing layer "
            "that cannot be reached falls back to the rules' own sentences and "
            "says so; it never silently produces the default and calls it a "
            "model's work."
        ),
    )


class CoachingConfig(BaseModel, extra="forbid"):
    """Policy for the coaching engine.

    As everywhere else in this system, these are **structural bounds, not golf
    norms**. Nothing here says what a good swing looks like; the thresholds that
    do live on the rules, each with the citation it came from.
    """

    min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence below which a metric does not support a conclusion, even "
            "one the arithmetic would reach comfortably. Stated policy: it is "
            "roughly the level at which the anchor factor alone -- how sure Phase "
            "4 is that this frame is the top at all -- stops supporting a "
            "statement about the top."
        ),
    )
    phrasing: PhrasingMode = Field(
        default=PhrasingMode.OFF,
        description=(
            "Off by default, and off is a complete configuration. Every finding "
            "already has a sentence; the model layer rewords, and a reworded "
            "finding is not a better-founded one."
        ),
    )
    phrasing_endpoint: str | None = Field(
        default=None,
        description=(
            "Base URL of a language model running on this machine. Only used when "
            "`phrasing` is `local`. There is no remote default and no hosted "
            "fallback: a finding is a measurement of a person's body, and the "
            "decision to send one anywhere is not a default."
        ),
    )
    phrasing_model: str | None = Field(
        default=None, description="Which local model to ask for, where the endpoint hosts several."
    )
    phrasing_timeout_s: float = Field(
        default=20.0,
        gt=0.0,
        description="How long to wait for the whole phrasing pass before giving up on it.",
    )


class CoachingReport(BaseModel):
    """Everything the coaching engine concluded about one swing.

    `computed` is false when there was no swing to reason about, and then both
    lists are empty rather than populated from whatever the metrics layer managed
    to produce anyway.
    """

    schema_version: int = COACHING_SCHEMA_VERSION
    computed: bool
    findings: list[Finding] = Field(default_factory=list)
    refused: list[RefusedFinding] = Field(default_factory=list)
    rules_considered: int = Field(
        default=0, description="How many rules were evaluated, whatever came out of them."
    )
    view: CameraView = Field(
        default=CameraView.UNKNOWN,
        description="The camera position the metrics were measured from. Decides several rules.",
    )
    frame_interval_s: float | None = Field(
        default=None,
        description=(
            "The clip's own clock resolution, in real seconds, as the swing events "
            "imply it. Every duration comparison's bracket comes from this, so a "
            "clip that did not supply one has its timing rules refused rather than "
            "compared against a threshold it cannot resolve."
        ),
    )
    phrasing: PhrasingReport = Field(default_factory=PhrasingReport)
    config: CoachingConfig = Field(default_factory=CoachingConfig)
    warnings: list[str] = Field(default_factory=list)

    def finding(self, rule_id: str) -> Finding | None:
        """The named finding, or None if that rule did not produce one."""
        return next((entry for entry in self.findings if entry.rule_id == rule_id), None)

    def refusal(self, rule_id: str) -> RefusedFinding | None:
        """The named refusal, or None if that rule was not refused."""
        return next((entry for entry in self.refused if entry.rule_id == rule_id), None)
