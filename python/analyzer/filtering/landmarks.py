"""Filtering a pose sequence: from stored landmarks to trajectories with derivatives.

This is the layer Phase 4 consumes. It runs the pipeline over each landmark's
three coordinate channels and assembles the results into position, velocity and
acceleration arrays, together with the mask saying where those values exist.

The three channels of one landmark share a gate and a gap policy without any
code coordinating them, because both stages key off the same detection flags and
the same reported confidences -- x, y and z of a landmark are present or absent
together. That consistency is a property of the data rather than an arrangement
between the stages, which is worth knowing when adding a stage that does not
share it.

Nothing here is persisted. Filtering a full sequence is milliseconds against the
seconds pose extraction takes (see `scripts/benchmark_filter.py`), so caching it
would add a second schema to version and invalidate for no measurable gain. That
is a decision made on the measurement, and it changes if the measurement does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.calibration import CameraIntrinsics
from analyzer.contracts.filtering import (
    FilterConfig,
    LandmarkFilterReport,
    SequenceFilterReport,
    StageReport,
    unit_for,
)
from analyzer.contracts.pose import FrameGeometry, Landmark, LandmarkSpace, PoseSequence
from analyzer.filtering.gating import gated_observations
from analyzer.filtering.localpoly import narrowest_window_s, sampling_interval_s
from analyzer.filtering.pipeline import FilterPipeline, default_pipeline
from analyzer.filtering.signal import Signal, signal_from_arrays
from analyzer.pose.series import LandmarkSeries, landmark_series, require_slow_motion_factor
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker

TASK_NAME = "filter_poses"

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class FilteredLandmark:
    """One landmark's filtered trajectory in one coordinate space.

    `position`, `velocity` and `acceleration` are (frames, 3) arrays, NaN
    wherever no value is supported. `valid` is the mask of frames carrying a
    complete position -- the only frames a metric may be computed from.

    `visibility` is what the estimator reported per frame, carried through
    rather than collapsed into the report's averages: Phase 4 weighs an event's
    confidence by how well the landmark was seen *around that instant*, which a
    clip-wide mean cannot answer. It is the raw reported value, including on
    frames the gate rejected — averaging only the frames that survived the gate
    would raise the score precisely where the landmark was least visible.

    `observed` is the separate question of whether a frame carried a real
    observation that survived the gate. It differs from `valid` at the ends of
    every clip, where the estimator saw the landmark perfectly well but the fit
    had too little support to emit anything, and the difference matters: one is
    a capture problem and the other a window-width one.
    """

    landmark: Landmark
    space: LandmarkSpace
    t: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    valid: NDArray[np.bool_]
    visibility: NDArray[np.float64]
    observed: NDArray[np.bool_]
    report: LandmarkFilterReport

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def speed(self) -> NDArray[np.float64]:
        """Magnitude of velocity, NaN where the velocity is not defined.

        Note this is the speed of the landmark *in its coordinate space*: in
        IMAGE space that is frame-widths per second, which is not a physical
        speed and must not be reported as one.
        """
        return np.linalg.norm(self.velocity, axis=1)


@dataclass(frozen=True)
class FilteredSequence:
    """Every landmark of one clip, filtered.

    `geometry` is carried through untouched from the pose sequence. Nothing in
    this layer uses it -- filtering is per-axis and an anisotropic scaling
    commutes with every stage of it -- but the biomechanics layer above cannot
    compute a distance or an angle without it, and this is the only path by
    which it can arrive there.
    """

    space: LandmarkSpace
    t: NDArray[np.float64]
    landmarks: dict[Landmark, FilteredLandmark]
    geometry: FrameGeometry
    slow_motion_factor: float
    report: SequenceFilterReport
    intrinsics: CameraIntrinsics | None = None
    """The calibration the landmarks were undistorted with, or None.

    Carried for the same reason `geometry` is: the layer above cannot tell from
    the numbers whether a lens was removed from them, and the difference decides
    what the measurements are entitled to claim."""

    def __getitem__(self, landmark: Landmark) -> FilteredLandmark:
        return self.landmarks[landmark]


def signals_from_series(series: LandmarkSeries) -> dict[str, Signal]:
    """One signal per coordinate axis, all carrying the landmark's confidences."""
    channels = {"x": series.x, "y": series.y, "z": series.z}
    return {
        axis: signal_from_arrays(
            series.timestamps_s,
            values,
            visibility=series.visibility,
            presence=series.presence,
            label=f"{series.landmark.name.lower()}.{axis}[{series.space.value}]",
        )
        for axis, values in channels.items()
    }


def _landmark_report(
    series: LandmarkSeries,
    reports: dict[str, list[StageReport]],
    valid: NDArray[np.bool_],
) -> LandmarkFilterReport:
    """Assemble the per-landmark report from the x-channel's stage reports.

    The gate and gap counts are read from one channel because all three see
    identical detection flags and confidences, so all three produce identical
    counts. The residual is per-axis, because that is the one quantity that
    genuinely differs between them.
    """
    gate, gaps, fit = (stage for stage in reports["x"])

    return LandmarkFilterReport(
        landmark=int(series.landmark),
        space=series.space,
        samples=len(series),
        observed=gate.counts["observed"],
        gated_out=gate.counts["gated_out"],
        never_detected=gate.counts["never_detected"],
        filled=gaps.counts["samples_filled"],
        blocked=gaps.counts["samples_blocked"],
        longest_gap_s=gaps.measurements["longest_gap_s"],
        valid_samples=int(np.count_nonzero(valid)),
        unsupported=fit.counts["unsupported"],
        residual_rms=[reports[axis][2].measurements["residual_rms"] for axis in _AXES],
        position_unit=unit_for(series.space, 0),
        velocity_unit=unit_for(series.space, 1),
        acceleration_unit=unit_for(series.space, 2),
        stages=reports["x"],
        notes=list(fit.notes),
    )


def filter_landmark(
    series: LandmarkSeries,
    config: FilterConfig | None = None,
    *,
    pipeline: FilterPipeline | None = None,
) -> FilteredLandmark:
    """Run the filter pipeline over one landmark's three coordinate channels."""
    resolved = config or FilterConfig()
    stages = pipeline or default_pipeline(resolved)

    signals = signals_from_series(series)
    outputs: dict[str, Signal] = {}
    reports: dict[str, list[StageReport]] = {}
    for axis, signal in signals.items():
        outputs[axis], reports[axis] = stages.run(signal)

    def _stack(attribute: str) -> NDArray[np.float64]:
        columns = []
        for axis in _AXES:
            channel = getattr(outputs[axis], attribute)
            columns.append(
                np.full(len(series), np.nan, dtype=np.float64) if channel is None else channel
            )
        return np.stack(columns, axis=1)

    position = _stack("value")
    velocity = _stack("velocity")
    acceleration = _stack("acceleration")

    # A frame counts as valid only with all three coordinates present. A partial
    # position is not a position, and letting one through would give any vector
    # computed from it a silently wrong magnitude.
    valid = np.all(np.isfinite(position), axis=1)

    return FilteredLandmark(
        landmark=series.landmark,
        space=series.space,
        t=series.timestamps_s,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        valid=valid,
        visibility=outputs["x"].visibility,
        # Asked of the pipeline's *input*, not its output: after the fit,
        # `value` holds fitted numbers and the mask would report where the fit
        # succeeded rather than where the estimator saw the landmark.
        observed=gated_observations(signals["x"], resolved.gate),
        report=_landmark_report(series, reports, valid),
    )


def _sequence_warnings(reports: list[LandmarkFilterReport], config: FilterConfig) -> list[str]:
    """Facts about the run that a caller should not have to compute to notice."""
    warnings: list[str] = []
    if not reports:
        return warnings

    valid_fractions = [report.valid_fraction for report in reports]
    mean_valid = sum(valid_fractions) / len(valid_fractions)

    if mean_valid == 0.0:
        warnings.append(
            "No landmark produced a single filtered position. Nothing downstream can be "
            "computed from this clip."
        )
    elif mean_valid < 0.5:
        warnings.append(
            f"Only {mean_valid:.0%} of landmark-frames survived filtering, so most of this "
            "clip has no usable trajectory."
        )

    blocked = sum(report.blocked for report in reports)
    if blocked:
        longest = max(report.longest_gap_s for report in reports)
        warnings.append(
            f"{blocked} landmark-frames sit inside gaps longer than the {config.gaps.max_gap_s:g} s "
            f"policy allows (longest {longest:.3f} s) and were left unfilled rather than "
            "interpolated."
        )

    unsupported = sum(report.unsupported for report in reports)
    if unsupported:
        warnings.append(
            f"{unsupported} landmark-frames had too few observations within the "
            f"{config.smoothing.window_s:g} s window to fit a degree-"
            f"{config.smoothing.polyorder} polynomial, so no value was emitted there."
        )

    # Surfaced once for the clip rather than 33 times, but only from the stage
    # that measured it -- this is the "your frame rate cannot support this
    # window" case, and it makes every other number in the report zero.
    for note in reports[0].notes:
        warnings.append(note)

    return warnings


def sequence_clock(sequence: PoseSequence, slow_motion_factor: float = 1.0) -> NDArray[np.float64]:
    """A sequence's timestamps on the time base the filter will see.

    Slow motion divides them, which is the whole of what the factor does, so a
    conformed 8x clip samples eight times as densely in real seconds and needs no
    widening where a 30 fps one does. Exposed because `resolve_window` has to be
    given every clock that will be compared before any of them is filtered.
    """
    require_slow_motion_factor(slow_motion_factor)
    return np.array(
        [frame.timestamp_s / slow_motion_factor for frame in sequence.frames], dtype=np.float64
    )


def resolve_window(
    config: FilterConfig, *clocks: NDArray[np.float64]
) -> tuple[FilterConfig, float | None]:
    """Widen a smoothing window the clips' frame rates cannot support.

    Returns the config to filter with, and the window originally asked for when
    it was replaced.

    **Resolved once for everything that will be compared, from those clips' own
    timestamps.** Doing it per landmark would let two landmarks of one clip be
    smoothed at different widths -- their velocities would then not be
    comparable, and nothing downstream could tell, because each report would look
    internally consistent.

    The same argument reaches past one clip, which is why this takes several
    clocks and resolves to the widest any of them needs. Anything that measures a
    *difference* between two recordings -- sync, comparison -- must smooth them
    identically, because a wider window flattens and slightly shifts the speed
    features both key on, and the pair would be biased by an amount nothing
    measures. So the coarser clip sets the window for the pair, which is the rule
    `SyncClipsParams.filter` already states.

    Widening only ever happens where the alternative is nothing at all: below
    about 45 fps the shipped degree-4 window holds three samples where the fit
    needs five, and `local_polynomial_fit` emits no value at any sample of any
    landmark. That is not a coarser measurement to be traded against a finer one,
    so it is not offered as a trade.
    """
    smoothing = config.smoothing
    if not smoothing.auto_widen:
        return config, None

    widths = [
        narrowest_window_s(interval, smoothing.required_observations)
        for interval in (sampling_interval_s(clock) for clock in clocks)
        if np.isfinite(interval)
    ]
    needed = max((w for w in widths if np.isfinite(w)), default=float("nan"))
    if not np.isfinite(needed) or needed <= smoothing.window_s:
        return config, None

    widened = config.model_copy(
        update={"smoothing": smoothing.model_copy(update={"window_s": needed})}
    )
    return widened, smoothing.window_s


def _widening_warning(config: FilterConfig, requested_s: float, interval_s: float) -> str:
    """Say that the window moved, and what it costs, in one place the caller reads."""
    smoothing = config.smoothing
    return (
        f"The {requested_s:g} s smoothing window covers fewer than the "
        f"{smoothing.required_observations} samples a degree-{smoothing.polyorder} fit needs at "
        f"this clip's {1.0 / interval_s:.0f} fps, so it was widened to "
        f"{smoothing.window_s:.3f} s -- the narrowest that fits. Nothing would have been fitted "
        "at the requested width. The wider window averages over more of the swing, which flattens "
        "peak speed and blunts impact timing; the events' resolution confidence already accounts "
        "for it. Capturing at 120 fps or more is what makes the narrower window available."
    )


def filter_sequence(
    sequence: PoseSequence,
    config: FilterConfig | None = None,
    *,
    space: LandmarkSpace = LandmarkSpace.FRAME_WIDTHS,
    slow_motion_factor: float = 1.0,
    landmarks: tuple[Landmark, ...] | None = None,
    intrinsics: CameraIntrinsics | None = None,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
) -> FilteredSequence:
    """Filter every landmark of a stored pose sequence.

    `intrinsics` is passed straight down to `landmark_series`, which removes the
    lens before the frame-widths conversion and therefore before anything here
    fits a polynomial to the result. Undistorting afterwards would be a
    different and wrong operation: the fit would have smoothed the distorted
    trajectory, and its velocity and acceleration would describe that."""
    requested = config or FilterConfig()
    clock = sequence_clock(sequence, slow_motion_factor)
    resolved, requested_window_s = resolve_window(requested, clock)

    pipeline = default_pipeline(resolved)
    selected = landmarks if landmarks is not None else tuple(Landmark)

    tracker = ProgressTracker(reporter or NullReporter(), task=TASK_NAME, request_id=request_id)
    tracker.report("starting", 0, len(selected))

    started = time.perf_counter()
    filtered: dict[Landmark, FilteredLandmark] = {}
    for done, landmark in enumerate(selected, start=1):
        series = landmark_series(sequence, landmark, space, slow_motion_factor, intrinsics)
        filtered[landmark] = filter_landmark(series, resolved, pipeline=pipeline)
        tracker.report("filtering", done, len(selected))

    elapsed = time.perf_counter() - started
    reports = [entry.report for entry in filtered.values()]
    timestamps = next(iter(filtered.values())).t if filtered else clock

    tracker.report("done", len(selected), len(selected))

    warnings = _sequence_warnings(reports, resolved)
    if requested_window_s is not None:
        # First, because it is the reason for anything else the report says about
        # the window, and a reader who stops after one line should get that one.
        warnings.insert(
            0, _widening_warning(resolved, requested_window_s, sampling_interval_s(clock))
        )

    return FilteredSequence(
        space=space,
        t=timestamps,
        landmarks=filtered,
        geometry=sequence.geometry,
        slow_motion_factor=slow_motion_factor,
        intrinsics=intrinsics,
        report=SequenceFilterReport(
            config=resolved,
            requested_window_s=requested_window_s,
            space=space,
            slow_motion_factor=slow_motion_factor,
            samples=len(sequence.frames),
            landmarks=reports,
            elapsed_s=elapsed,
            warnings=warnings,
        ),
    )
