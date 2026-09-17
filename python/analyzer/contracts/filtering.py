"""Typed contracts for temporal filtering.

Filtering is where a pose sequence stops being a list of detections and starts
being a signal with derivatives, and it is the first place in this pipeline
where a number can be invented rather than measured. Three policies decide
whether that happens, and all three are configuration rather than buried
constants, because each is a judgement that a later phase may need to revisit
with evidence this one does not have:

`ConfidenceGate` decides which detections count as observations at all.
`GapPolicy` decides how far the system is willing to interpolate across an
absence -- and, more importantly, when it must refuse and leave a hole.
`SmoothingConfig` decides the support of the local fit that produces position,
velocity and acceleration.

None of these defaults is tuned against ground-truth landmarks, because none
exist until Phase 12. The smoothing defaults come from a measured sweep against
an analytical trajectory (`scripts/benchmark_filter.py`); the gate and gap
defaults are stated policy. The distinction is kept explicit in the field
documentation so that a later reader does not mistake one for the other.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from analyzer.contracts.pose import LandmarkSpace

# Bump on any change that alters what a stored or reported filter result means.
FILTER_SCHEMA_VERSION = 1


class SignalUnit(StrEnum):
    """What a filtered value is actually in.

    Named rather than assumed, because the unit is the difference between a
    measurement and a number that looks like one. Nothing here is a calibrated
    metric quantity: IMAGE space is normalised to the frame, and HIP_LOCAL is
    MediaPipe's hip-centred output, which is only roughly metric and carries no
    camera geometry. The `APPROX_` prefix is there so that a call site
    formatting a value cannot spell "metres" without reading it.
    """

    NORMALIZED_FRAME = "normalized_frame"
    NORMALIZED_FRAME_PER_S = "normalized_frame_per_s"
    NORMALIZED_FRAME_PER_S2 = "normalized_frame_per_s2"
    FRAME_WIDTH = "frame_width"
    FRAME_WIDTH_PER_S = "frame_width_per_s"
    FRAME_WIDTH_PER_S2 = "frame_width_per_s2"
    APPROX_M = "approx_m"
    APPROX_M_PER_S = "approx_m_per_s"
    APPROX_M_PER_S2 = "approx_m_per_s2"


_UNITS: dict[LandmarkSpace, tuple[SignalUnit, SignalUnit, SignalUnit]] = {
    LandmarkSpace.IMAGE: (
        SignalUnit.NORMALIZED_FRAME,
        SignalUnit.NORMALIZED_FRAME_PER_S,
        SignalUnit.NORMALIZED_FRAME_PER_S2,
    ),
    # Distinct from NORMALIZED_FRAME rather than sharing it, because the two are
    # different units that happen to agree on x: a y of 0.5 is half the frame's
    # height in one and half its width in the other. Naming them apart is what
    # stops a residual measured in one being compared against a bound derived in
    # the other.
    LandmarkSpace.FRAME_WIDTHS: (
        SignalUnit.FRAME_WIDTH,
        SignalUnit.FRAME_WIDTH_PER_S,
        SignalUnit.FRAME_WIDTH_PER_S2,
    ),
    LandmarkSpace.HIP_LOCAL: (
        SignalUnit.APPROX_M,
        SignalUnit.APPROX_M_PER_S,
        SignalUnit.APPROX_M_PER_S2,
    ),
}


def unit_for(space: LandmarkSpace, derivative: int) -> SignalUnit:
    """The unit of the `derivative`-th time derivative of a signal in `space`.

    Time is always seconds -- it comes from the container's presentation
    timestamps -- so only the position unit varies with the space.
    """
    if derivative not in (0, 1, 2):
        raise ValueError(f"No unit is defined for derivative order {derivative}.")
    units = _UNITS.get(space)
    if units is None:
        raise ValueError(
            f"No unit is defined for {space.value} coordinates; this build cannot "
            "produce landmarks in that frame."
        )
    return units[derivative]


class ConfidenceGate(BaseModel, extra="forbid"):
    """Which detections are treated as observations.

    The two thresholds are separate because the model's two confidences mean
    different things: presence is whether the landmark is in frame, visibility
    is whether it is unoccluded. A landmark that is present but hidden behind
    the torso has a position the model guessed, and guessed positions are what
    this gate exists to remove.

    Both defaults are **policy, not a measured optimum.** 0.5 is MediaPipe's own
    default for its detection thresholds, so it is at least the model author's
    notion of an even split; nothing here has been checked against labelled
    landmark positions, because none exist until Phase 12.
    """

    min_visibility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this visibility a landmark is treated as not observed.",
    )
    min_presence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this presence a landmark is treated as not observed.",
    )


class GapPolicy(BaseModel, extra="forbid"):
    """How far the system will interpolate across an absence, and when it refuses.

    The refusal is the point. A local polynomial fit will happily produce a
    smooth value in the middle of a two-second hole if there are observations at
    the edges of its window, and that value would be indistinguishable in the
    output from one supported by data. `max_gap_s` draws the line, and anything
    beyond it stays NaN through every later stage.
    """

    max_gap_s: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "Longest absence, in seconds, that may be bridged by interpolation. "
            "Gaps longer than this stay NaN. The default is one dropped frame at "
            "30 fps and about a fifth of a typical downswing -- a stated policy "
            "bounded by how far a hand travels in that time, not a measured optimum."
        ),
    )
    filled_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight an interpolated sample carries in the smoothing fit. Zero by "
            "default: an interpolated value is a function of its neighbours and "
            "carries no independent information, so weighting it would count "
            "those neighbours twice. Raise it only to anchor a fit that would "
            "otherwise extrapolate, knowing that is what it does."
        ),
    )


class SmoothingConfig(BaseModel, extra="forbid"):
    """Support and order of the local polynomial fit.

    The window is in **seconds, not samples.** A fixed sample count means a
    different amount of time on every clip, and a different amount of time
    within one clip once the frame rate varies -- which is precisely the input
    this layer exists to handle correctly.

    **The defaults were measured, not chosen.** `scripts/benchmark_filter.py`
    sweeps window against order at 30/60/120/240 fps over three analytical
    trajectories, at a noise level measured from real footage, and compares the
    result against each trajectory's exact derivatives. Degree 4 over a 0.10 s
    window came out best or within a few percent of best on every quantity at
    the frame rates the capture protocol asks for:

    | order | window | pos RMS | vel RMS | acc RMS | peak speed err |
    | ----- | ------ | ------- | ------- | ------- | -------------- |
    | 2     | 0.10 s | 0.00194 | 0.6226  | 13.69   | -8.4%          |
    | 3     | 0.10 s | 0.00195 | 0.0435  | 13.68   | -1.1%          |
    | 4     | 0.10 s | 0.00077 | 0.0437  | 3.29    | -1.1%          |
    | 4     | 0.125 s| 0.00072 | 0.0609  | 2.33    | -0.7%          |

    (120 fps, sigma = 0.0014 normalized_frame, worst case over the three
    trajectories.) Degree 2 is not viable: it underestimates peak speed by 8%,
    which is an error Phase 4 would inherit when locating impact. Degree 4 beats
    degree 3 on acceleration by about 4x at equal velocity error. The 0.10 s
    window is preferred over 0.125 s because velocity is the signal phase
    detection keys on, and it is 28% better there.
    """

    window_s: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "Full width of the fitting window in seconds. Samples within "
            "±window_s/2 of the evaluation point are fitted. Widening it past "
            "~0.15 s flattens the velocity peak measurably: 0.30 s underestimates "
            "peak speed by 13%."
        ),
    )
    polyorder: int = Field(
        default=4,
        ge=0,
        le=5,
        description=(
            "Degree of the local polynomial. Acceleration needs at least 2; "
            "above 5 the fit follows noise faster than it follows motion."
        ),
    )
    min_observations: int | None = Field(
        default=None,
        description=(
            "Observations a window must contain before a value is emitted. "
            "None means polyorder + 1, the fewest that determine the fit. Note "
            "that a window holding exactly that many is *interpolated* rather "
            "than smoothed -- the fit passes through every point and its residual "
            "is zero by construction. Set this to polyorder + 2 or more to "
            "require that every emitted value actually averaged something."
        ),
    )

    @model_validator(mode="after")
    def _check_observations(self) -> SmoothingConfig:
        floor = self.polyorder + 1
        if self.min_observations is not None and self.min_observations < floor:
            raise ValueError(
                f"min_observations={self.min_observations} cannot determine a degree-"
                f"{self.polyorder} fit; it needs at least {floor}."
            )
        return self

    @property
    def required_observations(self) -> int:
        """Observations per window, with the default resolved."""
        return self.min_observations if self.min_observations is not None else self.polyorder + 1


class FilterConfig(BaseModel, extra="forbid"):
    """Everything that decides what the filtering layer produces.

    Carried alongside results rather than applied and forgotten: a filtered
    signal is only interpretable against the policy that produced it, and
    Phase 17's incremental re-analysis will key its cache on this.
    """

    schema_version: int = FILTER_SCHEMA_VERSION
    gate: ConfidenceGate = Field(default_factory=ConfidenceGate)
    gaps: GapPolicy = Field(default_factory=GapPolicy)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class StageReport(BaseModel):
    """What one pipeline stage did, measured.

    Counts are exact; measurements are derived quantities such as a residual.
    They are separate fields rather than one bag so that a consumer cannot read
    a count as a continuous quantity or vice versa.
    """

    stage: str
    counts: dict[str, int] = Field(default_factory=dict)
    measurements: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class LandmarkFilterReport(BaseModel):
    """What filtering did to one landmark's trajectory.

    `valid_samples` is the honest headline: the number of frames at which a
    position was actually produced. It is not the frame count, and the
    difference is made of gated detections, refused gaps and windows with too
    little support -- each counted separately below so the cause is visible
    rather than inferred.
    """

    landmark: int = Field(description="Landmark index, matching the Landmark enum.")
    space: LandmarkSpace
    samples: int = Field(description="Frames in the clip.")
    observed: int = Field(description="Frames with a detection surviving the confidence gate.")
    gated_out: int = Field(description="Detections rejected by the confidence gate.")
    never_detected: int = Field(description="Frames with no detection to gate in the first place.")
    filled: int = Field(description="Frames whose position was interpolated across a short gap.")
    blocked: int = Field(description="Frames inside a gap too long to bridge; left NaN.")
    longest_gap_s: float = Field(description="Longest absence in the trajectory, 0.0 if none.")
    valid_samples: int = Field(description="Frames with a smoothed position and derivatives.")
    unsupported: int = Field(
        description="Frames where the window held too few observations to fit, so nothing was emitted."
    )
    residual_rms: list[float] = Field(
        description=(
            "Per-axis RMS of (observation - fitted value) at observed frames. An "
            "estimate of the noise the fit removed, in the position unit."
        )
    )
    position_unit: SignalUnit
    velocity_unit: SignalUnit
    acceleration_unit: SignalUnit
    stages: list[StageReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        """Proportion of frames carrying a filtered position. 0.0 for an empty clip."""
        return self.valid_samples / self.samples if self.samples else 0.0


class SequenceFilterReport(BaseModel):
    """Filtering across every landmark of one clip."""

    schema_version: int = FILTER_SCHEMA_VERSION
    config: FilterConfig
    space: LandmarkSpace
    samples: int
    landmarks: list[LandmarkFilterReport]
    elapsed_s: float
    warnings: list[str] = Field(default_factory=list)

    @property
    def mean_valid_fraction(self) -> float:
        if not self.landmarks:
            return 0.0
        return sum(report.valid_fraction for report in self.landmarks) / len(self.landmarks)
