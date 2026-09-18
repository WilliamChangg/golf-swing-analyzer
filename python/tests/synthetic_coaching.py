"""Metric sets and detections built by hand, for testing what is reasoned from them.

The coaching layer's input is a `MetricSet` and a `SwingPhases`. It never looks
at a pixel, so the fixtures here construct those directly rather than driving a
synthetic body through pose estimation, filtering and detection to arrive at a
number that was chosen at the top anyway.

That is a deliberate departure from `tests/synthetic.py`, which builds a body
because the layers it tests measure one. Running the whole chain here would test
the chain -- again -- and would make a rule's threshold behaviour depend on
whether a local polynomial fit reproduced a constructed angle to three decimal
places. What these tests need is the ability to put a value exactly on a band
edge, and that is a property of the fixture rather than of the engine.
"""

from __future__ import annotations

from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricBasis,
    MetricConfidence,
    MetricConfig,
    MetricGroup,
    MetricName,
    MetricSet,
    MetricUnit,
    RefusedMetric,
    ViewEstimate,
)
from analyzer.contracts.phases import (
    DetectedEvent,
    EventConfidence,
    HandSignalInfo,
    HandSource,
    PhaseConfig,
    SwingEvent,
    SwingPhase,
    SwingPhases,
)
from analyzer.contracts.pose import FrameGeometry

FRAME_GEOMETRY = FrameGeometry(width=1080, height=1920)

# 30 fps, which is what a phone records by default and is the rate at which the
# tempo band stops being resolvable. Tests that want a finer clock pass their own.
DEFAULT_INTERVAL_S = 1.0 / 30.0

_GROUPS: dict[MetricName, MetricGroup] = {
    MetricName.TEMPO_RATIO: MetricGroup.TIMING,
    MetricName.BACKSWING_DURATION: MetricGroup.TIMING,
    MetricName.DOWNSWING_DURATION: MetricGroup.TIMING,
}


def metric(
    name: MetricName,
    value: float,
    *,
    unit: MetricUnit,
    basis: MetricBasis,
    event: SwingEvent | None = None,
    phase: SwingPhase | None = None,
    frames: tuple[int, ...] = (10, 20),
    confidence: float = 0.9,
    uncertainty: float | None = None,
    view: CameraView = CameraView.FACE_ON,
) -> Metric:
    """One measured quantity, with everything the contract requires and nothing real."""
    return Metric(
        name=name,
        group=_GROUPS.get(name, MetricGroup.ROTATION),
        label=f"{name.value.replace('_', ' ').capitalize()}",
        value=value,
        unit=unit,
        basis=basis,
        event=event,
        phase=phase,
        source_frames=list(frames),
        view=view,
        interpretation="A fixture. Means nothing anatomically.",
        uncertainty=uncertainty,
        confidence=MetricConfidence(
            overall=confidence, observation=1.0, anchor=confidence, method=1.0
        ),
        methodology="Constructed by a test.",
    )


def metric_set(
    metrics: list[Metric],
    *,
    view: CameraView = CameraView.FACE_ON,
    refused: list[RefusedMetric] | None = None,
    slow_motion_factor: float = 1.0,
    computed: bool = True,
) -> MetricSet:
    return MetricSet(
        computed=computed,
        metrics=metrics,
        refused=refused or [],
        view=ViewEstimate(
            view=view,
            confidence=1.0,
            shoulder_span_ratio=0.83,
            hip_span_ratio=0.42,
            openness=1.0,
            frames=[0, 1, 2],
            methodology="Constructed by a test.",
        ),
        torso_length=0.3,
        geometry=FRAME_GEOMETRY,
        frames=400,
        slow_motion_factor=slow_motion_factor,
        config=MetricConfig(),
    )


def phases(
    *,
    detected: bool = True,
    interval_s: float = DEFAULT_INTERVAL_S,
    takeaway_frame: int = 10,
    top_frame: int = 40,
    impact_frame: int = 50,
    finish_frame: int = 70,
) -> SwingPhases:
    """A detection carrying only what the coaching layer reads from one.

    Which is the events' frames and timestamps -- the clock resolution is derived
    from those, so a test that wants a 240 fps clip changes `interval_s` and
    nothing else. The phase intervals are left empty on purpose: nothing in the
    coaching engine reads them, and populating them would invite a rule that
    quietly started to.
    """
    frames = {
        SwingEvent.TAKEAWAY: takeaway_frame,
        SwingEvent.TOP: top_frame,
        SwingEvent.IMPACT: impact_frame,
        SwingEvent.FINISH: finish_frame,
    }
    return SwingPhases(
        detected=detected,
        events=[
            DetectedEvent(
                event=event,
                frame_index=frame,
                timestamp_s=frame * interval_s,
                confidence=EventConfidence(overall=0.9, margin=0.9, visibility=1.0, resolution=1.0),
                methodology="Constructed by a test.",
            )
            for event, frame in frames.items()
        ]
        if detected
        else [],
        hand=HandSignalInfo(
            source=HandSource.MIDPOINT,
            valid_frames=400,
            total_frames=400,
            peak_speed=1.0,
            travel=0.6,
            torso_length=0.3,
            travel_ratio=2.0,
        ),
        frames=400,
        config=PhaseConfig(),
    )


def tempo_metrics(
    backswing_s: float,
    downswing_s: float,
    *,
    view: CameraView = CameraView.FACE_ON,
    confidence: float = 0.9,
    slow_motion_factor: float = 1.0,
) -> MetricSet:
    """The three timing metrics a tempo rule needs, consistent with each other.

    The ratio is computed from the two durations rather than passed in, because
    a fixture where they disagreed would let a test pass against a bug that used
    the wrong one.
    """
    return metric_set(
        [
            metric(
                MetricName.BACKSWING_DURATION,
                backswing_s,
                unit=MetricUnit.SECONDS,
                basis=MetricBasis.TEMPORAL,
                phase=SwingPhase.BACKSWING,
                frames=(10, 39),
                confidence=confidence,
                view=view,
            ),
            metric(
                MetricName.DOWNSWING_DURATION,
                downswing_s,
                unit=MetricUnit.SECONDS,
                basis=MetricBasis.TEMPORAL,
                phase=SwingPhase.DOWNSWING,
                frames=(40, 49),
                confidence=confidence,
                view=view,
            ),
            metric(
                MetricName.TEMPO_RATIO,
                backswing_s / downswing_s,
                unit=MetricUnit.RATIO,
                basis=MetricBasis.TEMPORAL,
                frames=(10, 40, 50),
                confidence=confidence,
                view=view,
            ),
        ],
        view=view,
        slow_motion_factor=slow_motion_factor,
    )
