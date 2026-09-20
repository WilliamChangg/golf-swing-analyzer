"""Two recordings, side by side, with everything that is not a swing difference removed.

The entry point of the comparison layer. It reads two clips' filtered
trajectories, detections and metric sets -- never a pixel and never a file --
and produces one `SwingComparison`.

Two halves, answering two questions that the same report cannot answer with one
mechanism:

**The trajectories** answer *what shape was different*. Both clips are put on the
normalised clock `normalise` builds, which forces the four events to coincide,
and the curves are then laid over each other with a bracket that says how much of
any gap between them the event ambiguity alone can account for.

**The metric differences** answer *what was different by how much*, including
every timing difference the normalisation had to divide out in order to make the
first half possible. They go through the gates in `diff`, and on most real pairs
most of them refuse.

Neither half contains a score, and the report has no summary number. See
`contracts.comparison`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.comparison import normalise
from analyzer.comparison.diff import camera_agreement, compare_metrics
from analyzer.contracts.cache import ContentKey
from analyzer.contracts.comparison import (
    CameraAgreement,
    ChannelSample,
    ClipSummary,
    ComparisonConfig,
    DifferenceRefusal,
    HandPathOverlay,
    PathSample,
    PhaseClock,
    SwingComparison,
    TrajectoryChannel,
    TrajectoryComparison,
)
from analyzer.contracts.metrics import CameraView, MetricBasis, MetricSet, MetricUnit
from analyzer.contracts.phases import SwingEvent, SwingPhase, SwingPhases
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.phases.signals import SignalError, SwingSignals, swing_signals


class ComparisonError(ValueError):
    """Two clips could not be compared at all."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class SwingInput:
    """One side of a comparison: everything measured about one clip.

    Assembled by the caller from the same chain the metrics panel runs, rather
    than recomputed here, so that a difference between two clips can never be a
    difference between two filter configurations.
    """

    path: str
    content_key: ContentKey
    filtered: FilteredSequence
    phases: SwingPhases
    metrics: MetricSet


@dataclass(frozen=True)
class _Channel:
    """A signal, and what it is once divided by the subject's own torso length."""

    channel: TrajectoryChannel
    label: str
    unit: MetricUnit
    basis: MetricBasis


CHANNELS: tuple[_Channel, ...] = (
    _Channel(
        channel=TrajectoryChannel.HAND_SPEED,
        label="Hand speed",
        unit=MetricUnit.TORSO_LENGTHS_PER_S,
        basis=MetricBasis.IMAGE_PLANE,
    ),
    _Channel(
        channel=TrajectoryChannel.HAND_HEIGHT,
        label="Hand height above address",
        unit=MetricUnit.TORSO_LENGTHS,
        basis=MetricBasis.IMAGE_PLANE,
    ),
    _Channel(
        channel=TrajectoryChannel.SHOULDER_ANGLE,
        label="Shoulder line tilt",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
    ),
    _Channel(
        channel=TrajectoryChannel.HIP_ANGLE,
        label="Hip line tilt",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
    ),
)


def compare(
    reference: SwingInput, target: SwingInput, config: ComparisonConfig | None = None
) -> SwingComparison:
    """Everything these two recordings support saying about each other.

    Raises only when a clip's signals cannot be built at all. A clip with no
    swing, or one whose four events were not all located, comes back as a report
    with `computed=False` and a clock saying which knot is missing -- a result,
    not a failure, for the same reason `detected: false` is one.
    """
    settings = config or ComparisonConfig()

    reference_clock = normalise.build_clock(reference.phases)
    target_clock = normalise.build_clock(target.phases)
    camera = camera_agreement(reference.metrics, target.metrics, settings)

    summaries = (
        _summary(reference, reference_clock),
        _summary(target, target_clock),
    )

    if not (reference_clock.usable and target_clock.usable):
        return SwingComparison(
            computed=False,
            reference=summaries[0],
            target=summaries[1],
            camera=camera,
            config=settings,
            warnings=[
                warning
                for warning, clock in (
                    (f"{reference.path}: {reference_clock.methodology}", reference_clock),
                    (f"{target.path}: {target_clock.methodology}", target_clock),
                )
                if not clock.usable
            ],
        )

    differences, refused, considered = compare_metrics(
        reference.metrics,
        target.metrics,
        camera,
        settings,
        reference_interval_s=reference_clock.frame_interval_s,
        target_interval_s=target_clock.frame_interval_s,
    )

    try:
        reference_signals = swing_signals(reference.filtered)
        target_signals = swing_signals(target.filtered)
    except SignalError as exc:
        raise ComparisonError(
            str(exc), remediation="Re-run the extraction and filtering for both clips."
        ) from exc

    axis = normalise.positions(settings.samples)
    block = _channel_refusal(reference.metrics, target.metrics, camera)

    trajectories = [
        _channel_comparison(
            spec,
            axis,
            reference_clock,
            target_clock,
            reference_signals,
            target_signals,
            reference.phases,
            target.phases,
            block,
        )
        for spec in CHANNELS
    ]

    hand_path = _hand_path(
        axis,
        reference_clock,
        target_clock,
        reference_signals,
        target_signals,
        reference.phases,
        target.phases,
        block,
    )

    return SwingComparison(
        computed=True,
        reference=summaries[0],
        target=summaries[1],
        camera=camera,
        differences=differences,
        refused=refused,
        trajectories=trajectories,
        hand_path=hand_path,
        metrics_considered=considered,
        config=settings,
        warnings=[*reference.metrics.warnings, *target.metrics.warnings],
    )


def _summary(clip: SwingInput, clock: PhaseClock) -> ClipSummary:
    metrics = clip.metrics
    return ClipSummary(
        path=clip.path,
        content_key=clip.content_key,
        frames=metrics.frames,
        view=metrics.view.view if metrics.view is not None else CameraView.UNKNOWN,
        lead_side=metrics.lead_side.side if metrics.lead_side is not None else None,
        calibration=metrics.calibration,
        slow_motion_factor=metrics.slow_motion_factor,
        torso_length=_finite(metrics.torso_length),
        clock=clock,
    )


def _channel_refusal(
    reference: MetricSet, target: MetricSet, camera: CameraAgreement
) -> tuple[DifferenceRefusal, str] | None:
    """Why no trajectory may be overlaid at all, when that is the case.

    Every channel this layer carries is a measurement of the image plane, so all
    four stand or fall on the same two questions -- whether the cameras were in
    the same class of position, and whether they were in the same position. The
    check is made once and applied to all of them rather than per channel,
    because a reader offered three refused curves and one drawn one would read the
    drawn one as having passed a test the others failed, when in truth it was
    never asked a different question.
    """
    reference_view = reference.view.view if reference.view is not None else CameraView.UNKNOWN
    target_view = target.view.view if target.view is not None else CameraView.UNKNOWN
    if reference_view is not target_view or reference_view is CameraView.UNKNOWN:
        return DifferenceRefusal.VIEW_MISMATCH, (
            f"These clips were filmed from different positions "
            f"({reference_view.value} and {target_view.value}). Every signal here "
            "is a projection into the image plane, so two curves drawn over each "
            "other would be two different projections of two different swings and "
            "nothing could be attributed to either."
        )

    if reference.calibration is not target.calibration:
        return DifferenceRefusal.CALIBRATION_MISMATCH, (
            f"One clip's landmarks were corrected for the lens and the other's were "
            f"not ({reference.calibration.value} against {target.calibration.value}), "
            "which displaces every position these curves are built from."
        )

    if not camera.consistent:
        spans = (
            f"the shoulders span {camera.reference_span:.2f} torso lengths at "
            f"address in one and {camera.target_span:.2f} in the other"
            if camera.reference_span is not None and camera.target_span is not None
            else camera.methodology
        )
        return DifferenceRefusal.CAMERA_MOVED, (
            "These two recordings disagree about where the camera stood: "
            f"{spans}. Nothing here can separate a camera that moved from a player "
            "built differently, and either explains a difference between the "
            "curves without the swing having changed."
        )

    return None


def _channel_comparison(  # the two clips are symmetric; splitting them would hide that
    spec: _Channel,
    axis: NDArray[np.float64],
    reference_clock: PhaseClock,
    target_clock: PhaseClock,
    reference_signals: SwingSignals,
    target_signals: SwingSignals,
    reference_phases: SwingPhases,
    target_phases: SwingPhases,
    block: tuple[DifferenceRefusal, str] | None,
) -> TrajectoryComparison:
    """One signal from both clips, resampled onto the shared axis and differenced."""
    if block is not None:
        refusal, reason = block
        return TrajectoryComparison(
            channel=spec.channel,
            label=spec.label,
            unit=spec.unit,
            basis=spec.basis,
            samples=[],
            resolved_fraction=0.0,
            refusal=refusal,
            reason=reason,
        )

    left = normalise.resample(
        reference_clock,
        reference_signals.t,
        _values(spec.channel, reference_signals, reference_phases),
        axis,
    )
    right = normalise.resample(
        target_clock,
        target_signals.t,
        _values(spec.channel, target_signals, target_phases),
        axis,
    )

    samples: list[ChannelSample] = []
    resolved = 0
    largest, largest_at = None, None

    for index, position in enumerate(axis):
        reference_value = _finite(left.value[index])
        target_value = _finite(right.value[index])
        difference: float | None = None
        bracket: float | None = None
        is_resolved = False

        if reference_value is not None and target_value is not None:
            difference = target_value - reference_value
            terms = (left.bracket[index], right.bracket[index])
            if all(math.isfinite(term) for term in terms):
                # Added, not in quadrature. Two recordings share no systematic
                # error for a quadrature to be excused by; see `diff`.
                bracket = float(terms[0] + terms[1])
                is_resolved = abs(difference) > bracket
            if is_resolved:
                resolved += 1
                if largest is None or abs(difference) > abs(largest):
                    largest, largest_at = difference, float(position)

        samples.append(
            ChannelSample(
                position=float(position),
                reference=reference_value,
                target=target_value,
                difference=difference,
                bracket=bracket,
                resolved=is_resolved,
            )
        )

    return TrajectoryComparison(
        channel=spec.channel,
        label=spec.label,
        unit=spec.unit,
        basis=spec.basis,
        samples=samples,
        resolved_fraction=resolved / len(samples) if samples else 0.0,
        largest_difference=largest,
        largest_at=largest_at,
    )


def _values(
    channel: TrajectoryChannel, signals: SwingSignals, phases: SwingPhases
) -> NDArray[np.float64]:
    """One channel's series on a clip's own clock, on a scale another clip can meet.

    Lengths and speeds are divided by the subject's own torso length, which is
    what makes them survive a change of framing and a change of subject -- the
    same argument Phase 4 makes for judging hand travel in torso lengths rather
    than in frame widths. Heights are additionally taken from that clip's own
    address pose, because "how high the hands are in the frame" is a fact about
    where the player was standing.

    Angles are left alone. A projected tilt is already in degrees and a baseline
    subtracted from it would hide a difference in setup, which is a difference in
    the swing rather than in the recording.
    """
    scale = signals.torso_length
    if not math.isfinite(scale) or scale <= 0.0:
        return np.full(signals.t.shape, np.nan, dtype=np.float64)

    if channel is TrajectoryChannel.HAND_SPEED:
        return np.asarray(signals.speed, dtype=np.float64) / scale
    if channel is TrajectoryChannel.HAND_HEIGHT:
        anchor = _address_hand(signals, phases)
        return (np.asarray(signals.height, dtype=np.float64) - anchor[1]) / scale
    if channel is TrajectoryChannel.SHOULDER_ANGLE:
        return np.asarray(signals.shoulder_angle_deg, dtype=np.float64)
    return np.asarray(signals.hip_angle_deg, dtype=np.float64)


def _address_hand(signals: SwingSignals, phases: SwingPhases) -> tuple[float, float]:
    """Where the hands sat at address, in frame widths, as the origin for heights.

    The median over the address phase rather than one frame: the subject is still
    there, so a reduction over the interval is a better estimate of the pose than
    any single frame of it, which is the same reasoning the metric layer applies
    to its own address baselines.

    Falls back to the takeaway frame when no address phase was detected -- a clip
    that starts mid-swing has one -- and to zero when even that is untracked, in
    which case the heights are measured from the frame's own bottom edge and the
    channel is a comparison of framings. That last case cannot arise while the
    takeaway is a knot on the clock, which is checked before this is called.
    """
    address = next(
        (entry for entry in phases.phases if entry.phase is SwingPhase.ADDRESS),
        None,
    )
    position = signals.hand.position
    valid = signals.hand.valid

    if address is not None and address.end_frame > address.start_frame:
        window = slice(address.start_frame, address.end_frame)
        usable = valid[window]
        if np.any(usable):
            inside = position[window][usable]
            return float(np.median(inside[:, 0])), float(np.median(inside[:, 1]))

    takeaway = phases.event(SwingEvent.TAKEAWAY)
    if (
        takeaway is not None
        and 0 <= takeaway.frame_index < len(valid)
        and valid[takeaway.frame_index]
    ):
        point = position[takeaway.frame_index]
        return float(point[0]), float(point[1])

    return 0.0, 0.0


def _hand_path(  # symmetric in the two clips, as `_channel_comparison` is
    axis: NDArray[np.float64],
    reference_clock: PhaseClock,
    target_clock: PhaseClock,
    reference_signals: SwingSignals,
    target_signals: SwingSignals,
    reference_phases: SwingPhases,
    target_phases: SwingPhases,
    block: tuple[DifferenceRefusal, str] | None,
) -> HandPathOverlay:
    """Both hand arcs on one axis, in each subject's own torso lengths from address."""
    if block is not None:
        return HandPathOverlay(samples=[], comparable=False, reason=block[1])

    left = _path_series(axis, reference_clock, reference_signals, reference_phases)
    right = _path_series(axis, target_clock, target_signals, target_phases)

    samples = [
        PathSample(
            position=float(position),
            reference_x=_finite(left[0][index]),
            reference_y=_finite(left[1][index]),
            target_x=_finite(right[0][index]),
            target_y=_finite(right[1][index]),
        )
        for index, position in enumerate(axis)
    ]
    return HandPathOverlay(samples=samples, comparable=True)


def _path_series(
    axis: NDArray[np.float64],
    clock: PhaseClock,
    signals: SwingSignals,
    phases: SwingPhases,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scale = signals.torso_length
    anchor = _address_hand(signals, phases)
    if not math.isfinite(scale) or scale <= 0.0:
        empty = np.full(axis.shape, np.nan, dtype=np.float64)
        return empty, empty

    across = (np.where(signals.hand.valid, signals.hand.position[:, 0], np.nan) - anchor[0]) / scale
    upward = (np.where(signals.hand.valid, signals.hand.position[:, 1], np.nan) - anchor[1]) / scale
    return (
        normalise.resample(clock, signals.t, across, axis).value,
        normalise.resample(clock, signals.t, upward, axis).value,
    )


def _finite(value: float) -> float | None:
    """None rather than NaN, because JSON has no NaN and a null is the honest shape."""
    return float(value) if math.isfinite(value) else None
