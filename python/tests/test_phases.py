"""Swing phase detection tests.

Detection is judged against synthetic signals whose event times are known by
construction rather than against real footage, for the same reason Phase 3 is
judged against analytical trajectories: a detector can be plausible on real
video and wrong, and there is no ground truth to catch it until Phase 12.

The synthetic signal is built from a *velocity profile* that is integrated into
a trajectory, so the instants the detector is supposed to find are inputs rather
than things read off a plot afterwards. It is a swing only in the shape of its
signal -- one speed minimum at the highest point, one speed maximum after it --
which is exactly and only what the rules key on.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.filtering import ConfidenceGate, FilterConfig, SmoothingConfig
from analyzer.contracts.phases import (
    HandSource,
    PhaseConfig,
    SwingEvent,
    SwingPhase,
)
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    Landmark,
    LandmarkPoint,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import SignalError, detect_phases, swing_signals

FPS = 120.0

# Event times the synthetic signal is built around.
TAKEAWAY_S = 0.50
TOP_S = 1.30
IMPACT_S = 1.70
# The descent and the follow-through are mirror images in the arc model below,
# so the hands come to rest as long after impact as the top was before it.
FINISH_S = TOP_S + 2 * (IMPACT_S - TOP_S)
DURATION_S = 2.60

# Radius and lowest point of the arc the hands travel, in frame widths.
ARC_RADIUS = 0.22
ARC_LOW_HEIGHT = 0.33
ARC_TOP_ANGLE = 2.2

# Tolerance for a recovered event. Half the default smoothing window: an event
# cannot be located more precisely than the filter that produced the signal.
TOLERANCE_S = SmoothingConfig().window_s / 2

# The takeaway gets its own, looser bound, and only on the late side. It is
# defined as the instant hand speed becomes *measurable* -- crossing a fraction
# of the swing's peak -- and motion starting from rest necessarily takes time to
# get there. The event is therefore never early and is expected to lag; what
# would be a defect is finding it before the hands moved at all.
TAKEAWAY_LAG_S = 0.20


def _arc_angle(t: np.ndarray) -> np.ndarray:
    """Angle of the hands along their arc, zero at the bottom.

    The hands are modelled on a circle rather than on a vertical line, because
    the vertical model gets impact wrong in a way that matters. On a real swing
    the hands are at the bottom of their arc at impact and moving horizontally,
    so their *vertical* speed there is near zero while their total speed peaks.
    A purely vertical fixture puts peak speed and lowest position at different
    instants, and would make the corroboration check below meaningless.

    Backswing eases in and out, so speed is zero at the takeaway and again at
    the top. The descent and follow-through are one cosine sweep through the
    bottom, which puts maximum angular speed exactly where the angle crosses
    zero -- the lowest point of the arc, and the instant impact is defined to be.
    """
    angle = np.zeros_like(t)

    rising = (t >= TAKEAWAY_S) & (t < TOP_S)
    u = (t[rising] - TAKEAWAY_S) / (TOP_S - TAKEAWAY_S)
    angle[rising] = ARC_TOP_ANGLE * np.sin(np.pi / 2 * u) ** 2

    sweeping = (t >= TOP_S) & (t <= FINISH_S)
    u = (t[sweeping] - TOP_S) / (FINISH_S - TOP_S)
    angle[sweeping] = ARC_TOP_ANGLE * np.cos(np.pi * u)

    after = t > FINISH_S
    angle[after] = -ARC_TOP_ANGLE
    return angle


def _hand_path(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The arc as an image-space path.

    Returns x and y with y in image convention (increasing downward), so the
    detector's conversion to an upward height is genuinely exercised rather than
    bypassed by handing it a signal that already points the right way.
    """
    angle = _arc_angle(t)
    height = ARC_LOW_HEIGHT + ARC_RADIUS * (1.0 - np.cos(angle))
    return 0.5 + ARC_RADIUS * np.sin(angle), 1.0 - height


# A static body for the synthetic subject, in image coordinates (y downward).
# The torso has to be real: detection judges hand travel in torso lengths, so a
# fixture with every landmark stacked on one point has no scale to measure
# against and every ratio it produces is meaningless.
TORSO_LENGTH = 0.25
_BODY: dict[int, tuple[float, float]] = {
    int(Landmark.LEFT_SHOULDER): (0.45, 0.25),
    int(Landmark.RIGHT_SHOULDER): (0.55, 0.25),
    int(Landmark.LEFT_HIP): (0.46, 0.25 + TORSO_LENGTH),
    int(Landmark.RIGHT_HIP): (0.54, 0.25 + TORSO_LENGTH),
}


def _sequence(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
    detected: np.ndarray | None = None,
) -> PoseSequence:
    """A pose sequence whose wrists follow the given path on a still body."""
    vis = np.full(t.size, 0.95) if visibility is None else visibility
    seen = np.ones(t.size, dtype=bool) if detected is None else detected
    wrists = {int(Landmark.LEFT_WRIST), int(Landmark.RIGHT_WRIST)}

    frames = []
    for index in range(t.size):
        if not seen[index]:
            frames.append(PoseFrame(frame_index=index, timestamp_s=float(t[index]), detected=False))
            continue
        points = []
        for landmark in range(LANDMARK_COUNT):
            if landmark in wrists:
                # Left and right a little apart, so the midpoint is a distinct
                # point from either and the hand-source choice is exercised.
                offset = 0.01 if landmark == int(Landmark.LEFT_WRIST) else -0.01
                px, py = float(x[index]) + offset, float(y[index])
            else:
                px, py = _BODY.get(landmark, (0.5, 0.2))
            points.append(
                LandmarkPoint(
                    x=px,
                    y=py,
                    z=0.0,
                    visibility=float(vis[index]),
                    presence=0.99,
                )
            )
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=float(t[index]),
                detected=True,
                image=points,
                hip_local=points,
            )
        )

    found = int(np.count_nonzero(seen))
    return PoseSequence(
        video_path="/data/synthetic.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="f" * 64, size_bytes=1
        ),
        model=PoseModelInfo(
            name="fake",
            variant="fake",
            precision="float32",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 16, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=t.size,
            frames_detected=found,
            detection_rate=found / t.size if t.size else 0.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
        ),
        frames=frames,
    )


def _detect(sequence: PoseSequence, *, filter_config=None, phase_config=None):
    filtered = filter_sequence(sequence, filter_config or FilterConfig())
    return detect_phases(filtered, phase_config)


def _swing(duration_s: float = DURATION_S, fps: float = FPS, **kwargs):
    t = np.arange(0.0, duration_s, 1.0 / fps)
    x, y = _hand_path(t)
    return _sequence(x, y, t, **kwargs)


@pytest.fixture(scope="module")
def result():
    """The reference synthetic swing, detected once for the whole module."""
    return _detect(_swing())


class TestSyntheticSwing:
    """Every event is recovered within half a smoothing window of where it was put."""

    def test_reports_a_swing(self, result) -> None:
        assert result.detected
        assert len(result.events) == 4

    @pytest.mark.parametrize(
        ("event", "expected_s"),
        [
            (SwingEvent.TOP, TOP_S),
            (SwingEvent.IMPACT, IMPACT_S),
            (SwingEvent.FINISH, FINISH_S),
        ],
    )
    def test_locates_each_event(self, result, event: SwingEvent, expected_s: float) -> None:
        found = result.event(event)
        assert found is not None
        assert found.timestamp_s == pytest.approx(expected_s, abs=TOLERANCE_S)

    def test_locates_the_takeaway_without_preceding_the_motion(self, result) -> None:
        found = result.event(SwingEvent.TAKEAWAY)
        assert found is not None
        assert TAKEAWAY_S <= found.timestamp_s <= TAKEAWAY_S + TAKEAWAY_LAG_S

    def test_the_takeaway_is_not_found_at_the_pause_at_the_top(self, result) -> None:
        """The failure the backswing-peak bound exists to prevent.

        The hands come almost to rest at the top, so a search for the last still
        stretch before the top finds that pause and reports the takeaway in the
        middle of the swing -- with a backswing of a few frames and a plausible
        looking result.
        """
        assert result.event(SwingEvent.TAKEAWAY).timestamp_s < TOP_S - 0.3

    def test_events_are_ordered_by_construction(self, result) -> None:
        """The searches are nested, so a later event cannot precede an earlier one."""
        times = [result.event(e).timestamp_s for e in SwingEvent]
        assert times == sorted(times)

    def test_every_event_carries_its_methodology(self, result) -> None:
        assert all(entry.methodology for entry in result.events)

    def test_impact_is_corroborated_by_the_lowest_hand_position(self, result) -> None:
        """A second, independent signal for the same instant."""
        impact = result.event(SwingEvent.IMPACT)
        assert impact.corroboration_frame is not None
        assert abs(impact.corroboration_delta_s) < TOLERANCE_S

    def test_phases_tile_the_clip_without_gaps_or_overlap(self, result) -> None:
        intervals = sorted(result.phases, key=lambda p: p.start_frame)
        assert intervals[0].start_frame == 0
        for earlier, later in itertools.pairwise(intervals):
            assert earlier.end_frame == later.start_frame

    def test_phase_order_is_the_swing_order(self, result) -> None:
        assert [entry.phase for entry in result.phases] == [
            SwingPhase.ADDRESS,
            SwingPhase.BACKSWING,
            SwingPhase.DOWNSWING,
            SwingPhase.FOLLOW_THROUGH,
        ]

    def test_phase_at_maps_a_frame_to_its_phase(self, result) -> None:
        top = result.event(SwingEvent.TOP)
        assert result.phase_at(top.frame_index) is SwingPhase.DOWNSWING
        assert result.phase_at(top.frame_index - 1) is SwingPhase.BACKSWING
        assert result.phase_at(0) is SwingPhase.ADDRESS

    def test_phase_at_returns_none_outside_the_detected_span(self, result) -> None:
        assert result.phase_at(10_000) is None

    def test_durations_come_from_timestamps(self, result) -> None:
        backswing = next(p for p in result.phases if p.phase is SwingPhase.BACKSWING)
        assert backswing.duration_s == pytest.approx(TOP_S - TAKEAWAY_S, abs=2 * TOLERANCE_S)


class TestRefusal:
    """What produces no events, and whether it says why."""

    def test_a_still_subject_is_not_a_swing(self) -> None:
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        rng = np.random.default_rng(0)
        jitter = 0.0005 * rng.standard_normal(t.size)
        result = _detect(_sequence(0.5 + jitter, 0.5 + jitter, t))

        assert not result.detected
        assert result.events == []
        assert result.phases == []
        assert any("No swing detected" in w for w in result.warnings)

    def test_a_refusal_still_reports_what_was_measured(self) -> None:
        """So a caller can see the reason rather than only the verdict."""
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        result = _detect(_sequence(np.full(t.size, 0.5), np.full(t.size, 0.5), t))

        assert not result.detected
        assert result.hand.total_frames == t.size
        assert result.frames == t.size

    def test_untracked_hands_produce_no_events(self) -> None:
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        x, y = _hand_path(t)
        result = _detect(_sequence(x, y, t, detected=np.zeros(t.size, dtype=bool)))

        assert not result.detected
        assert any("never tracked" in w for w in result.warnings)

    def test_a_single_brief_movement_is_not_a_swing(self) -> None:
        """One quick motion has no backswing, whatever its speed."""
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        y = np.full(t.size, 0.5)
        moving = (t > 1.0) & (t < 1.08)
        y[moving] = 0.5 - 0.2 * np.sin(np.pi * (t[moving] - 1.0) / 0.08)
        result = _detect(_sequence(np.full(t.size, 0.5), y, t))

        assert not result.detected
        # Rejected for having no real motion either side of the burst, rather
        # than for its durations: the rules happily find a "top" on whichever
        # still frame noise made highest, and the backswing that leads to it is
        # nearly a second long and completely motionless.
        assert any("rather than noise" in w for w in result.warnings)

    def test_low_confidence_landmarks_are_gated_before_detection(self) -> None:
        """Gating happens in the filter, so detection never sees the guesses."""
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        result = _detect(
            _sequence(x, y, t, visibility=np.full(t.size, 0.2)),
            filter_config=FilterConfig(gate=ConfidenceGate(min_visibility=0.5)),
        )
        assert not result.detected


class TestTruncatedClip:
    def test_reports_the_finish_at_the_last_frame_and_says_so(self) -> None:
        result = _detect(_swing(duration_s=IMPACT_S + 0.10))

        finish = result.event(SwingEvent.FINISH)
        assert finish is not None
        assert any("had not come to rest" in w for w in result.warnings)

    def test_a_truncated_finish_earns_no_confidence(self) -> None:
        """It is the end of the recording, not the end of the motion."""
        result = _detect(_swing(duration_s=IMPACT_S + 0.10))
        assert result.event(SwingEvent.FINISH).confidence.overall == 0.0


class TestDoubleSwing:
    @staticmethod
    def _two_swings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two swings with the club lowered back to address in between.

        The settle matters. Splicing two swings together directly leaves the
        hands at the finish of one and the address of the next in consecutive
        frames, and the differentiator reads that step as a velocity spike
        dwarfing both swings -- after which only one peak clears the prominence
        bar and the test passes or fails for the wrong reason.
        """
        settle_s = 1.5
        gap = FINISH_S + settle_s
        t = np.arange(0.0, gap + DURATION_S, 1.0 / FPS)

        angle = _arc_angle(t)
        settling = (t > FINISH_S) & (t <= FINISH_S + settle_s)
        u = (t[settling] - FINISH_S) / settle_s
        angle[settling] = -ARC_TOP_ANGLE * (1.0 - np.sin(np.pi / 2 * u) ** 2)

        second = t > FINISH_S + settle_s
        angle[second] = _arc_angle(t[second] - gap)

        height = ARC_LOW_HEIGHT + ARC_RADIUS * (1.0 - np.cos(angle))
        return 0.5 + ARC_RADIUS * np.sin(angle), 1.0 - height, t

    def test_two_swings_are_noticed_and_named(self) -> None:
        """Analysing the fastest silently would attribute metrics to whichever won."""
        x, y, t = self._two_swings()
        result = _detect(_sequence(x, y, t))
        assert any("swing-like motions" in w for w in result.warnings)


class TestConfidence:
    def test_overall_is_the_product_of_its_factors(self) -> None:
        result = _detect(_swing())
        for entry in result.events:
            factors = entry.confidence
            assert factors.overall == pytest.approx(
                factors.margin * factors.visibility * factors.resolution, abs=1e-9
            )

    def test_resolution_collapses_when_the_window_swallows_the_downswing(self) -> None:
        """The measured 24 fps failure, reproduced deliberately.

        A window as long as the event being measured leaves a curve that is an
        artefact of the filter, so the event keeps its frame and loses its
        confidence.
        """
        wide = FilterConfig(smoothing=SmoothingConfig(window_s=0.40, polyorder=4))
        narrow = FilterConfig(smoothing=SmoothingConfig(window_s=0.05, polyorder=4))

        blunt = _detect(_swing(), filter_config=wide)
        sharp = _detect(_swing(), filter_config=narrow)

        assert blunt.event(SwingEvent.IMPACT).confidence.resolution < 0.15
        assert sharp.event(SwingEvent.IMPACT).confidence.resolution == 1.0
        # The event is still found; it is its worth that collapses, not its frame.
        assert blunt.detected

    def test_a_narrow_window_resolves_the_downswing_fully(self) -> None:
        narrow = FilterConfig(smoothing=SmoothingConfig(window_s=0.05, polyorder=4))
        result = _detect(_swing(), filter_config=narrow)
        assert result.event(SwingEvent.IMPACT).confidence.resolution == 1.0

    def test_poor_visibility_lowers_confidence_without_moving_the_event(self) -> None:
        clear = _detect(_swing())
        dim = _detect(_swing(visibility=np.full(int(DURATION_S * FPS) + 1, 0.55)))

        assert dim.detected
        assert (
            dim.event(SwingEvent.IMPACT).frame_index == clear.event(SwingEvent.IMPACT).frame_index
        )
        assert (
            dim.event(SwingEvent.IMPACT).confidence.visibility
            < clear.event(SwingEvent.IMPACT).confidence.visibility
        )

    def test_a_phase_is_only_as_confident_as_its_weaker_boundary(self) -> None:
        result = _detect(_swing(duration_s=IMPACT_S + 0.10))
        follow = next((p for p in result.phases if p.phase is SwingPhase.FOLLOW_THROUGH), None)
        if follow is not None:
            assert follow.confidence == 0.0


class TestSignals:
    def test_height_increases_upward_despite_image_coordinates(self) -> None:
        """The sign convention that would invert the top into the bottom.

        Image y grows downward. If the conversion were missing, the highest hand
        position would be found where the hands are lowest, and every event after
        it would be wrong without anything failing.
        """
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        filtered = filter_sequence(_sequence(x, y, t), FilterConfig())
        signals = swing_signals(filtered)

        highest = int(np.nanargmax(signals.height))
        lowest_y = int(np.nanargmin(filtered[Landmark.RIGHT_WRIST].position[:, 1]))
        assert highest == lowest_y

    def test_the_hand_point_is_chosen_once_for_the_clip(self) -> None:
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        filtered = filter_sequence(_sequence(x, y, t), FilterConfig())
        assert swing_signals(filtered).hand.source in set(HandSource)

    def test_prefers_the_midpoint_when_both_wrists_are_equally_tracked(self) -> None:
        """Ties go to the midpoint, which averages two independent estimates."""
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        filtered = filter_sequence(_sequence(x, y, t), FilterConfig())
        assert swing_signals(filtered).hand.source is HandSource.MIDPOINT

    def test_refuses_hip_local_space(self) -> None:
        """Hand height there is measured from an origin that moves with the body."""
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        filtered = filter_sequence(
            _sequence(x, y, t), FilterConfig(), space=LandmarkSpace.HIP_LOCAL
        )
        with pytest.raises(SignalError, match="HIP_LOCAL"):
            swing_signals(filtered)

    def test_refuses_a_sequence_missing_a_wrist(self) -> None:
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        filtered = filter_sequence(_sequence(x, y, t), FilterConfig(), landmarks=(Landmark.NOSE,))
        with pytest.raises(SignalError, match="both wrist"):
            swing_signals(filtered)

    def test_travel_is_measured_in_the_subject_s_own_torso_lengths(self) -> None:
        """Scale comes from the body, so framing cannot change the verdict."""
        t = np.arange(0.0, DURATION_S, 1.0 / FPS)
        x, y = _hand_path(t)
        signals = swing_signals(filter_sequence(_sequence(x, y, t), FilterConfig()))

        assert signals.torso_length == pytest.approx(TORSO_LENGTH, abs=0.01)
        assert signals.travel_ratio > 1.0

    def test_a_still_subject_travels_a_fraction_of_a_torso(self) -> None:
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        flat = np.full(t.size, 0.5)
        signals = swing_signals(filter_sequence(_sequence(flat, flat, t), FilterConfig()))
        assert signals.travel_ratio < 0.1


class TestConfigValidation:
    def test_rejects_a_moving_fraction_outside_zero_to_one(self) -> None:
        with pytest.raises(ValueError):
            PhaseConfig(moving_fraction=1.5)

    def test_rejects_a_signal_to_noise_bar_below_one(self) -> None:
        with pytest.raises(ValueError):
            PhaseConfig(min_signal_to_noise=0.5)

    def test_rejects_an_unknown_field(self) -> None:
        with pytest.raises(ValueError):
            PhaseConfig(min_backswing=0.2)
